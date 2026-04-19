# Phase 1: vLLM Server Setup

**Phase name**: `server`

## Objective
Kill any stale vLLM processes, download the model if needed, start the vLLM server with profiler enabled, verify the `/start_profile` API is reachable.

**The server started here runs through Phase 5. Do NOT restart it between phases.**

---

**Constraint 3**: Never modify `/opt/`, `/usr/`, or pip-installed packages. This phase starts vLLM normally; kernel injection happens via PYTHONPATH in Phase 5.

## Step 1: Kill Stale vLLM + Verify GPU Memory

**WHY**: `pgrep -f` can hang on uninterruptible processes. `VLLM::EngineCore` subprocesses hold GPU VRAM even after the API server exits — they must be found via `/proc/*/maps` (GPU device file handles), not by name.

```bash
# ── Find ALL processes holding GPU device files (catches EngineCore name changes) ───
GPU_PIDS=$(grep -rl "kfd\|renderD" /proc/*/maps 2>/dev/null \
           | grep -oP '(?<=/proc/)\d+' | sort -u)

if [ -n "$GPU_PIDS" ]; then
    echo "Killing GPU-holding PIDs: $GPU_PIDS"
    echo "$GPU_PIDS" | xargs -r kill -9 2>/dev/null || true
    sleep 3
else
    echo "No GPU-holding processes found."
fi

# ── Verify GPU memory is fully released ──────────────────────────────────
python3 - << 'PYEOF'
import subprocess, sys, time

for attempt in range(6):   # max 30s wait
    r = subprocess.run(['rocm-smi','--showmemuse'], capture_output=True, text=True, timeout=10)
    occupied = [l for l in r.stdout.splitlines() if 'VRAM%' in l and ': 0' not in l]
    if not occupied:
        print("All GPUs free."); break
    print(f"  [{attempt+1}/6] Waiting for VRAM release: {occupied}")
    if attempt == 5:
        print("WARNING: VRAM still occupied — may be a different workload. Proceeding anyway.")
    time.sleep(5)
PYEOF
```

## Step 2: Select GPU(s)

```bash
if [ -n "{{GPUS}}" ]; then
    export CUDA_VISIBLE_DEVICES="{{GPUS}}"
    echo "Using specified GPUs: $CUDA_VISIBLE_DEVICES"
else
    SELECTED=$(python3 {{SCRIPTS_DIR}}/select_gpus.py {{TP}})
    export CUDA_VISIBLE_DEVICES="$SELECTED"
    echo "Auto-selected GPUs: $CUDA_VISIBLE_DEVICES"
fi
```

## Step 3: Environment Variables

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=1800000
mkdir -p "$HF_HOME"
```

## Step 4: Ensure Model is Available Locally

Do NOT rely on vLLM to download. Use explicit `hf download` with progress output.

```bash
python3 << 'PYEOF'
import os, sys, subprocess, json

model = "{{MODEL}}"
print(f"Checking model: {model}")

if os.path.isdir(model):
    cfg = os.path.join(model, "config.json")
    if not os.path.isfile(cfg):
        print(f"ERROR: config.json not found in {model}"); sys.exit(1)
    weights = [f for f in os.listdir(model) if f.endswith(('.safetensors', '.bin'))]
    if not weights:
        print("ERROR: no weight files found"); sys.exit(1)
    print(f"Local model OK: {len(weights)} weight file(s)")
    # Write resolved path for subsequent steps
    with open("{{OUTPUT_DIR}}/resolved_model_path.txt", "w") as f:
        f.write(model)
else:
    name = model.split("/")[-1]
    local = f"/app/{name}"
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
        ws = [f for f in os.listdir(local) if f.endswith(('.safetensors', '.bin'))]
        if ws:
            print(f"Model already at {local} ({len(ws)} weight file(s))")
            with open("{{OUTPUT_DIR}}/resolved_model_path.txt", "w") as f:
                f.write(local)
            sys.exit(0)
    print(f"Downloading {model} to {local}...")
    r = subprocess.run(["hf", "download", model, "--local-dir", local],
                       capture_output=False, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"ERROR: download failed (exit {r.returncode})"); sys.exit(1)
    if not os.path.isfile(os.path.join(local, "config.json")):
        print("ERROR: download incomplete"); sys.exit(1)
    ws = [f for f in os.listdir(local) if f.endswith(('.safetensors', '.bin'))]
    print(f"Downloaded: {local} ({len(ws)} weight file(s))")
    with open("{{OUTPUT_DIR}}/resolved_model_path.txt", "w") as f:
        f.write(local)
PYEOF

# Use resolved path for all subsequent steps
if [ -f "{{OUTPUT_DIR}}/resolved_model_path.txt" ]; then
    MODEL=$(cat "{{OUTPUT_DIR}}/resolved_model_path.txt")
    echo "Model path: $MODEL"
fi
```

## Step 5: Start vLLM Server with Profiler

**CRITICAL**: The `--profiler-config` flag requires a JSON string. Without it, `/start_profile` returns 404.

```bash
VLLM_LOG="{{OUTPUT_DIR}}/vllm_server.log"

PROFILER_JSON=$(python3 -c "
import json
print(json.dumps({
    'profiler': 'torch',
    'torch_profiler_dir': '{{PROFILE_DIR}}',
    'torch_profiler_record_shapes': True,
    'torch_profiler_with_stack': False,
    'torch_profiler_use_gzip': True,
    'ignore_frontend': True,
    'max_iterations': {{PROFILE_ITERATIONS}},
}))
")

echo "Starting vLLM server..."
echo "Log: $VLLM_LOG"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype {{DTYPE}} \
    --tensor-parallel-size {{TP}} \
    --trust-remote-code \
    --enforce-eager \
    --api-key dummy \
    --max-model-len {{MAX_MODEL_LEN}} \
    --gpu-memory-utilization {{GPU_MEM_UTIL}} \
    --profiler-config "$PROFILER_JSON" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > "{{OUTPUT_DIR}}/vllm.pid"
echo "vLLM PID: $VLLM_PID"

# Wait for server ready
# KEY: check kill -0 $VLLM_PID first — if process died, stop immediately and show log.
# Fail-fast on fatal errors: CUDA out of memory, ValidationError, ModuleNotFoundError, OOMKilled.
# Do NOT keep looping after process death (wastes minutes).
for i in $(seq 1 36); do
    sleep 10
    kill -0 $VLLM_PID 2>/dev/null || {
        echo "ERROR: vLLM process died at ${i}0s!"
        tail -20 "$VLLM_LOG"
        exit 1
    }
    # Check for fatal startup errors before trying HTTP
    if grep -qE 'CUDA out of memory|OOMKilled|ValidationError|ModuleNotFoundError|Address already in use' "$VLLM_LOG" 2>/dev/null; then
        echo "ERROR: fatal error detected in log:"; grep -E 'CUDA out of memory|OOMKilled|ValidationError|ModuleNotFoundError|Address already in use' "$VLLM_LOG" | tail -3; exit 1
    fi
    HTTP=$(curl -s -o/dev/null -w '%{http_code}' \
        -H "Authorization: Bearer dummy" http://localhost:8000/v1/models 2>/dev/null)
    [ "$HTTP" = "200" ] && echo "Server ready at ${i}0s" && break
    [ $i -eq 36 ] && { echo "ERROR: timeout 360s"; tail -10 "$VLLM_LOG"; exit 1; }
    echo "  ${i}0s | $(tail -1 "$VLLM_LOG")"
done
```

## Step 6: Verify Server and Profiler API

```bash
# Verify model loaded
curl -s -H "Authorization: Bearer dummy" http://localhost:8000/v1/models | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

# Verify profiler API
RESP=$(curl -s -X POST http://localhost:8000/start_profile -H "Authorization: Bearer dummy")
echo "start_profile response: $RESP"
if echo "$RESP" | grep -q "Not Found"; then
    echo "ERROR: Profiler API not available (server started without --profiler-config)"; exit 1
fi

# Stop immediately — profiling happens in Phase 2
curl -s -X POST http://localhost:8000/stop_profile -H "Authorization: Bearer dummy" > /dev/null
echo "Profiler API verified."
```

## Completion

Update `{{PROGRESS_FILE}}`:
```json
{"phases_completed": ["env", "server"], "details": {"vllm_pid": "<PID>", "model": "{{MODEL}}", "profiler_enabled": true}}
```
