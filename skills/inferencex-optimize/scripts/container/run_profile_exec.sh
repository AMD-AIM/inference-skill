#!/usr/bin/env bash
# Run a profiling benchmark inside a Docker container with heartbeats and trace flush wait.
#
# Usage: bash run_profile_exec.sh \
#   --container <name> --benchmark-script <path> \
#   --model <model> --tp <tp> --ep <ep> --conc <conc> \
#   --isl <isl> --osl <osl> --max-model-len <len> \
#   --precision <prec> --framework <fw> --exp-name <name> \
#   --result-filename <name> --repo-dir <dir> \
#   [--gpus <gpu-ids>] [--docker-log <path>]
set -euo pipefail

CONTAINER=""
BENCHMARK_SCRIPT=""
MODEL=""
TP=""
EP=""
CONC=""
ISL=""
OSL=""
MAX_MODEL_LEN=""
PRECISION=""
FRAMEWORK=""
EXP_NAME=""
RESULT_FILENAME=""
REPO_DIR=""
GPUS=""
DOCKER_LOG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)        CONTAINER="$2"; shift 2 ;;
        --benchmark-script) BENCHMARK_SCRIPT="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --tp)               TP="$2"; shift 2 ;;
        --ep)               EP="$2"; shift 2 ;;
        --conc)             CONC="$2"; shift 2 ;;
        --isl)              ISL="$2"; shift 2 ;;
        --osl)              OSL="$2"; shift 2 ;;
        --max-model-len)    MAX_MODEL_LEN="$2"; shift 2 ;;
        --precision)        PRECISION="$2"; shift 2 ;;
        --framework)        FRAMEWORK="$2"; shift 2 ;;
        --exp-name)         EXP_NAME="$2"; shift 2 ;;
        --result-filename)  RESULT_FILENAME="$2"; shift 2 ;;
        --repo-dir)         REPO_DIR="$2"; shift 2 ;;
        --gpus)             GPUS="$2"; shift 2 ;;
        --docker-log)       DOCKER_LOG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

DOCKER_LOG="${DOCKER_LOG:-/tmp/${CONTAINER}_profile.log}"
GPU_ENV=""
if [ -n "$GPUS" ] && [ "$GPUS" != "auto" ]; then
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=$GPUS"
    echo "Using manually specified GPUs: $GPUS"
else
    echo "Using all available GPUs"
fi

# Pre-run cleanup: kill stale servers
docker exec "$CONTAINER" bash -c '
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "trtllm-serve" 2>/dev/null || true
    sleep 2
    if command -v rocm-smi &>/dev/null; then
        echo "GPU memory before profiling:"
        rocm-smi --showmeminfo vram 2>/dev/null | grep -E "Used|Total" | head -4
    elif command -v nvidia-smi &>/dev/null; then
        echo "GPU memory before profiling:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
    fi
' || true

echo "Starting profile run: RESULT_FILENAME=$RESULT_FILENAME"
echo "Streaming live output to terminal and $DOCKER_LOG"

(
    set -o pipefail
    docker exec \
        $GPU_ENV \
        -e MODEL="$MODEL" \
        -e TP="$TP" \
        -e EP_SIZE="$EP" \
        -e CONC="$CONC" \
        -e ISL="$ISL" \
        -e OSL="$OSL" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e RANDOM_RANGE_RATIO=0.5 \
        -e RESULT_FILENAME="$RESULT_FILENAME" \
        -e PRECISION="$PRECISION" \
        -e FRAMEWORK="$FRAMEWORK" \
        -e EXP_NAME="$EXP_NAME" \
        "$CONTAINER" \
        /bin/bash "/workspace/$BENCHMARK_SCRIPT" \
        2>&1 | tee -a "$DOCKER_LOG"
) &
EXEC_PID=$!

while kill -0 "$EXEC_PID" 2>/dev/null; do
    sleep 30
    if kill -0 "$EXEC_PID" 2>/dev/null; then
        echo "[heartbeat] Profile run still running for $RESULT_FILENAME"
        echo "[heartbeat] Full log: $DOCKER_LOG"
    fi
done

wait "$EXEC_PID"
EXIT_CODE=$?
echo "Profile exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker log ==="
    tail -n 50 "$DOCKER_LOG"
fi

# Wait for trace flush
echo "Waiting for trace files to finish writing..."
PROF_DIR="$REPO_DIR/profiles"
STABLE_COUNT=0
PREV_SIZE=""
MAX_WAIT=60
WAITED=0
while [ $STABLE_COUNT -lt 3 ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 5
    WAITED=$((WAITED + 5))
    CURR_SIZE=$(du -sb "$PROF_DIR" 2>/dev/null | cut -f1)
    if [ -z "$CURR_SIZE" ]; then
        echo "[flush-wait] No profiles directory found, skipping"
        break
    fi
    if [ "$CURR_SIZE" = "$PREV_SIZE" ]; then
        STABLE_COUNT=$((STABLE_COUNT + 1))
        echo "[flush-wait] Size stable at $(numfmt --to=iec $CURR_SIZE 2>/dev/null || echo ${CURR_SIZE}B), count=$STABLE_COUNT/3"
    else
        STABLE_COUNT=0
        echo "[flush-wait] Still writing: $(numfmt --to=iec ${PREV_SIZE:-0} 2>/dev/null || echo ${PREV_SIZE:-0}B) -> $(numfmt --to=iec $CURR_SIZE 2>/dev/null || echo ${CURR_SIZE}B)"
    fi
    PREV_SIZE="$CURR_SIZE"
done
if [ $WAITED -ge $MAX_WAIT ]; then
    echo "WARNING: Trace may still be writing after ${MAX_WAIT}s — proceeding anyway"
fi

exit $EXIT_CODE
