#!/usr/bin/env bash
# Start a persistent Docker container with GPUs, configurable mounts, and env vars.
#
# Usage: bash start_profile_container.sh \
#   --name <container-name> --image <docker-image> --runner <runner> \
#   --repo-dir <repo-dir> --hf-cache <hf-cache> --profile-dir <profile-dir> \
#   [--mode benchmark|profile|optimize] \
#   [--mount src:dst] ... \
#   [--env KEY=VAL] ...
#
# Modes:
#   profile   (default) -- sets PROFILE=1 and all SGLANG_*/VLLM_* profiler env vars
#   benchmark -- sets only base HF/timeout env vars; no profiler vars
#   optimize  -- same base env as benchmark; use --env/--mount for extras
#
# Outputs: CONTAINER_NAME=<name> on success.
set -euo pipefail

CONTAINER_NAME=""
IMAGE=""
RUNNER=""
REPO_DIR=""
HF_CACHE=""
PROFILE_DIR=""
MODE="profile"
EXTRA_MOUNTS=()
EXTRA_ENVS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)        CONTAINER_NAME="$2"; shift 2 ;;
        --image)       IMAGE="$2"; shift 2 ;;
        --runner)      RUNNER="$2"; shift 2 ;;
        --repo-dir)    REPO_DIR="$2"; shift 2 ;;
        --hf-cache)    HF_CACHE="$2"; shift 2 ;;
        --profile-dir) PROFILE_DIR="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --mount)       EXTRA_MOUNTS+=("$2"); shift 2 ;;
        --env)         EXTRA_ENVS+=("$2"); shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$MODE" in
    benchmark|profile|optimize) ;;
    *) echo "Invalid mode: $MODE (expected benchmark|profile|optimize)" >&2; exit 1 ;;
esac

if [[ "$RUNNER" == mi* ]]; then
    GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"
else
    GPU_FLAGS="--gpus all"
fi

# Build extra mount args
MOUNT_ARGS=()
for m in "${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"}"; do
    MOUNT_ARGS+=(-v "$m")
done

# Build extra env args
ENV_ARGS=()
for e in "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}"; do
    ENV_ARGS+=(-e "$e")
done

# Common env vars present in all modes
COMMON_ENV_ARGS=(
    -e HF_HOME=/root/.cache/huggingface
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub
    -e VLLM_RPC_TIMEOUT=1800000
    -e SGLANG_WARMUP_TIMEOUT=10000
)

# Profiler env vars -- only in profile mode
PROFILE_ENV_ARGS=()
if [[ "$MODE" == "profile" ]]; then
    PROFILE_ENV_ARGS=(
        -e PROFILE=1
        -e SGLANG_TORCH_PROFILER_DIR=/workspace/profiles
        -e VLLM_TORCH_PROFILER_DIR=/workspace/profiles
        -e SGLANG_PROFILE_WITH_STACK=True
        -e SGLANG_PROFILE_RECORD_SHAPE=True
        -e SGLANG_ENABLE_PROFILER_METADATA=1
    )
fi

DOCKER_RUN_LOG="${PROFILE_DIR}/${CONTAINER_NAME}_docker_run.log"
echo "DOCKER_RUN_LOG: $DOCKER_RUN_LOG"

# Clean up stale container with the same name from prior runs
if docker inspect "$CONTAINER_NAME" &>/dev/null; then
    echo "Removing stale container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

set +e
docker run -d \
    --name "$CONTAINER_NAME" \
    --label inferencex-pipeline=true \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 64g \
    --ipc=host \
    --network=host \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    "${MOUNT_ARGS[@]+"${MOUNT_ARGS[@]}"}" \
    -w /workspace \
    "${COMMON_ENV_ARGS[@]}" \
    "${PROFILE_ENV_ARGS[@]+"${PROFILE_ENV_ARGS[@]}"}" \
    "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}" \
    "$IMAGE" \
    -c "sleep infinity" \
    > "$DOCKER_RUN_LOG" 2>&1
EXIT_CODE=$?
set -e

echo "Container start exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker run log ==="
    tail -n 50 "$DOCKER_RUN_LOG"
    exit $EXIT_CODE
fi

echo "CONTAINER_NAME=$CONTAINER_NAME"
