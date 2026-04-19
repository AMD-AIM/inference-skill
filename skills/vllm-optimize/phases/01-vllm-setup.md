# Phase 1: vLLM Server Setup

## Objective
Start vLLM server with required configuration for benchmark and profiling.

## CRITICAL: Profiler Enablement

**IMPORTANT**: To use torch profiler, you MUST start vLLM with `--profiler-config.*` flags. Without these flags, the `/start_profile` and `/stop_profile` API endpoints will NOT be registered, and you will get "404 Not Found" errors.

## Steps

### 1. Kill Any Existing vLLM Processes

**Kill both the API server AND the EngineCore worker processes.** The EngineCore holds GPU memory — if only the API server is killed, EngineCore becomes a zombie and leaks VRAM.

```bash
# Kill EngineCore workers first (they hold GPU memory)
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
# Then kill the API server
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
echo "Cleaned vLLM processes"
```

### 2. Environment Validation

```bash
# Detect GPU vendor
GPU_VENDOR=""
if command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
    echo "Detected AMD GPU"
elif command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    echo "Detected NVIDIA GPU"
else
    echo "ERROR: No GPU detection tool found" >&2
    exit 1
fi

# Check vLLM installation
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" || { echo "ERROR: vLLM not installed"; exit 1; }

# Check PyTorch GPU detection
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### 3. Auto-Select Idle GPU(s)

Use `select_gpus.py` to pick the least-utilized GPU(s) for this run. This avoids conflicts with other processes.

```bash
SELECTED_GPUS=$(python3 {{SCRIPTS_DIR}}/select_gpus.py {{TP}})
export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
echo "Auto-selected GPU(s): CUDA_VISIBLE_DEVICES=$SELECTED_GPUS"
```

If the user explicitly specified GPUs via `{{GPUS}}`, use that value instead:
```bash
if [ -n "{{GPUS}}" ]; then
    export CUDA_VISIBLE_DEVICES="{{GPUS}}"
    echo "Using user-specified GPUs: $CUDA_VISIBLE_DEVICES"
fi
```

### 4. Set Environment Variables

```bash
# HuggingFace proxy (required in restricted network environments)
export HF_ENDPOINT="${HF_ENDPOINT:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-}"

# HuggingFace cache
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"

# vLLM runtime
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=1800000
```

### 5. Create Directories

```bash
mkdir -p "{{OUTPUT_DIR}}" "{{PROFILE_DIR}}" "{{REPORT_DIR}}" "{{SCRIPTS_DIR}}"
```

### 6. Ensure Model is Available Locally

If `{{MODEL}}` is a local directory, validate it has config.json and weight files.
If it is a HuggingFace model ID (e.g., `org/model-name`), **download it first** using `hf download`, then update the model path to the local directory.

**Do NOT rely on vLLM to download the model at startup** — that gives no progress feedback and fails silently on network issues.

```bash
python3 << 'PYEOF'
import os, sys, json, subprocess

model = "{{MODEL}}"
print(f"Checking model: {model}")

if os.path.isdir(model):
    # Local path — validate
    config_path = os.path.join(model, "config.json")
    if not os.path.isfile(config_path):
        print(f"ERROR: config.json not found in {model}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"  model_type: {cfg.get('model_type', 'unknown')}")
    print(f"  architectures: {cfg.get('architectures', [])}")
    weights = [f for f in os.listdir(model) if f.endswith(('.safetensors', '.bin'))]
    print(f"  weight files: {len(weights)}")
    if not weights:
        print("ERROR: No weight files found")
        sys.exit(1)
    print(f"Model ready: {model}")
else:
    # Remote HF model ID — download to local directory
    # Derive local path: /app/<model-name> or {{OUTPUT_DIR}}/<model-name>
    model_name = model.split("/")[-1]
    local_dir = os.path.join("/app", model_name)
    
    if os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "config.json")):
        weights = [f for f in os.listdir(local_dir) if f.endswith(('.safetensors', '.bin'))]
        if weights:
            print(f"Model already downloaded at {local_dir} ({len(weights)} weight files)")
            # Write the resolved local path for subsequent steps
            with open("{{OUTPUT_DIR}}/resolved_model_path.txt", "w") as f:
                f.write(local_dir)
            sys.exit(0)
    
    print(f"Downloading {model} to {local_dir}...")
    result = subprocess.run(
        ["hf", "download", model, "--local-dir", local_dir],
        capture_output=False, text=True, timeout=1800  # 30 min timeout
    )
    if result.returncode != 0:
        print(f"ERROR: Download failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Validate download
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        print(f"ERROR: Download completed but config.json not found in {local_dir}")
        sys.exit(1)
    
    weights = [f for f in os.listdir(local_dir) if f.endswith(('.safetensors', '.bin'))]
    print(f"Model downloaded: {local_dir} ({len(weights)} weight files)")
    
    # Write the resolved local path for subsequent steps
    with open("{{OUTPUT_DIR}}/resolved_model_path.txt", "w") as f:
        f.write(local_dir)
PYEOF

# If model was downloaded, update MODEL to local path
if [ -f "{{OUTPUT_DIR}}/resolved_model_path.txt" ]; then
    MODEL=$(cat "{{OUTPUT_DIR}}/resolved_model_path.txt")
    echo "Using resolved model path: $MODEL"
fi
```

**IMPORTANT**: If the model was a remote ID and got downloaded, all subsequent steps (including the vLLM start command) should use the resolved local path from `resolved_model_path.txt`.

### 7. Start vLLM Server and Wait for Ready

**CRITICAL**: The server start and readiness check MUST run in the same shell so that `VLLM_PID` is available for health checking. Execute the entire block below as ONE script.

**For benchmark-only mode (NO profiling)**, use this single script:
```bash
VLLM_LOG="{{OUTPUT_DIR}}/vllm_server.log"

python3 -m vllm.entrypoints.openai.api_server \
    --model "{{MODEL}}" \
    --dtype {{DTYPE}} \
    --tensor-parallel-size {{TP}} \
    --trust-remote-code \
    --enforce-eager \
    --api-key dummy \
    --max-model-len {{MAX_MODEL_LEN}} \
    --gpu-memory-utilization {{GPU_MEM_UTIL}} \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID, log: $VLLM_LOG"

MAX_WAIT=300; ELAPSED=0; INTERVAL=10; SERVER_READY=false
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM exited!"; tail -50 "$VLLM_LOG"; exit 1
    fi
    if grep -qE '(ValidationError|CUDA out of memory|OOMKilled|Address already in use|ModuleNotFoundError)' "$VLLM_LOG" 2>/dev/null; then
        echo "ERROR: fatal error in log!"; grep -A5 'Error' "$VLLM_LOG" | tail -20; kill $VLLM_PID 2>/dev/null; exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer dummy" http://localhost:8000/v1/models 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server ready! (took ${ELAPSED}s)"; SERVER_READY=true; break
    fi
    sleep $INTERVAL; ELAPSED=$((ELAPSED + INTERVAL))
    echo "[wait ${ELAPSED}/${MAX_WAIT}s] vLLM starting... (HTTP: $HTTP_CODE)"
done
if [ "$SERVER_READY" != "true" ]; then
    echo "ERROR: timeout after ${MAX_WAIT}s"; tail -50 "$VLLM_LOG"; kill $VLLM_PID 2>/dev/null; exit 1
fi
echo $VLLM_PID > "{{OUTPUT_DIR}}/vllm.pid"
```

**For profiling mode (REQUIRED for profiler API)**, use this single script:

**IMPORTANT**: The `--profiler-config` flag requires a JSON string (not dot-notation). Construct it as shown below.
```bash
VLLM_LOG="{{OUTPUT_DIR}}/vllm_server.log"

# Build profiler config as JSON string (required format for vLLM >= 0.16)
PROFILER_JSON=$(python3 -c "
import json
print(json.dumps({
    'profiler': 'torch',
    'torch_profiler_dir': '{{PROFILE_DIR}}',
    'torch_profiler_record_shapes': True,
    'torch_profiler_with_stack': True,
    'torch_profiler_use_gzip': True,
    'ignore_frontend': True,
    'max_iterations': {{PROFILE_ITERATIONS}},
}))
")

python3 -m vllm.entrypoints.openai.api_server \
    --model "{{MODEL}}" \
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
echo "vLLM PID: $VLLM_PID, log: $VLLM_LOG"

MAX_WAIT=300; ELAPSED=0; INTERVAL=10; SERVER_READY=false
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM exited!"; tail -50 "$VLLM_LOG"; exit 1
    fi
    if grep -qE '(ValidationError|CUDA out of memory|OOMKilled|Address already in use|ModuleNotFoundError)' "$VLLM_LOG" 2>/dev/null; then
        echo "ERROR: fatal error in log!"; grep -A5 'Error' "$VLLM_LOG" | tail -20; kill $VLLM_PID 2>/dev/null; exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer dummy" http://localhost:8000/v1/models 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server ready! (took ${ELAPSED}s)"; SERVER_READY=true; break
    fi
    sleep $INTERVAL; ELAPSED=$((ELAPSED + INTERVAL))
    echo "[wait ${ELAPSED}/${MAX_WAIT}s] vLLM starting... (HTTP: $HTTP_CODE)"
done
if [ "$SERVER_READY" != "true" ]; then
    echo "ERROR: timeout after ${MAX_WAIT}s"; tail -50 "$VLLM_LOG"; kill $VLLM_PID 2>/dev/null; exit 1
fi
echo $VLLM_PID > "{{OUTPUT_DIR}}/vllm.pid"
```

### 9. Verify Server

```bash
curl -s -H "Authorization: Bearer dummy" http://localhost:8000/v1/models | \
    python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('data', [])
if models:
    print(f'Status: Ready')
    print(f'Model: {models[0][\"id\"]}')
else:
    print('Status: FAILED - no models loaded')
    sys.exit(1)
"
```

### 10. Verify Profiler API (if started with profiling enabled)

```bash
echo "Testing /start_profile..."
START_RESP=$(curl -s -X POST http://localhost:8000/start_profile -H "Authorization: Bearer dummy")
echo "Response: $START_RESP"

if echo "$START_RESP" | grep -q "Not Found"; then
    echo "ERROR: Profiler API not available!"
    echo "The server was NOT started with --profiler-config flags."
    echo "You must restart with profiling flags."
    exit 1
fi

echo "Testing /stop_profile..."
curl -s -X POST http://localhost:8000/stop_profile -H "Authorization: Bearer dummy"
echo ""
echo "Profiler API verified."
```

## Troubleshooting

### vLLM process exits immediately
- Check the log file at `{{OUTPUT_DIR}}/vllm_server.log`
- Common causes: model architecture not supported, OOM, port conflict

### Profiler API returns 404 Not Found
- **Cause**: Server started without `--profiler-config.*` flags
- **Solution**: Restart server with profiling flags (see Step 7 profiling mode)

### CUDA out of memory
- Reduce `--gpu-memory-utilization` (default 0.9)
- Reduce `--max-model-len`
- Use a GPU with more VRAM

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | HuggingFace model name or local path |
| TP | 1 | Tensor parallelism |
| DTYPE | bfloat16 | Model precision (bfloat16, half, float16) |
| HF_HOME | /root/.cache/huggingface | HuggingFace cache directory |
| OUTPUT_DIR | ./vllm_results | Results directory |
| PROFILE_DIR | $OUTPUT_DIR/profiles | Profiler output |
| PROFILE_ITERATIONS | 128 | Profiler iterations |
| MAX_MODEL_LEN | 4096 | Maximum model context length |
| GPU_MEM_UTIL | 0.9 | GPU memory utilization fraction |
| GPUS | (auto) | Explicit GPU IDs; empty = auto-select via select_gpus.py |

## Completion

Server running at http://localhost:8000 with OpenAI-compatible API.
Save the `VLLM_PID` and `VLLM_LOG` path for later phases (cleanup, restart for profiling).

Update progress.json:
```json
{
  "phase": "vllm-setup",
  "phases_completed": ["env", "vllm-setup"],
  "current_step": "vLLM server running",
  "details": {
    "model": "{{MODEL}}",
    "gpu": "<selected GPU IDs>",
    "pid": "<VLLM_PID>",
    "log": "{{OUTPUT_DIR}}/vllm_server.log",
    "profiler_enabled": true
  }
}
```

Next: Proceed to Phase 2 (Benchmark)
