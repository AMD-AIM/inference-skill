# Phase 1: vLLM Server Setup

**Phase name**: `server`  **Phase number**: 1

## Objective
Kill previous vLLM (by PID file), select GPU, set env vars (including TunableOps), start vLLM with profiler, verify server and profiler API ready.

**CRITICAL**: `PYTORCH_TUNABLEOP_*` env vars must be set here — before server starts — so EngineCore inherits them at fork.

## Kill vLLM pattern (used here and in Phase 5)
vLLM's recommended shutdown is **SIGTERM** (triggers uvicorn graceful shutdown + engine cleanup).
`kill -9` skips cleanup and leaves orphan `multiprocessing.resource_tracker` children holding
`/dev/kfd` open, which prevents ROCm from releasing VRAM even after the main process dies.

```bash
# vLLM graceful shutdown: SIGTERM → wait 30s → SIGKILL fallback
_kill_vllm() {
    local pid=$1
    [ -z "$pid" ] && return
    kill -0 $pid 2>/dev/null || return   # already gone
    echo "  Stopping PID=$pid (SIGTERM)..."
    kill -SIGTERM $pid 2>/dev/null || true
    for _w in $(seq 1 30); do
        kill -0 $pid 2>/dev/null || { echo "  Exited cleanly after ${_w}s."; return; }
        sleep 1
    done
    echo "  Still alive after 30s — sending SIGKILL"
    kill -9 $pid 2>/dev/null || true
    sleep 2
}
_kill_vllm $(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")
```

## Execution

```bash
mkdir -p "{{OUTPUT_DIR}}/logs" "{{PROFILE_DIR}}" "{{RESULTS_DIR}}"
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_1_server.log"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'server','status':'running','phase_1_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail

echo "[$(date +%T)] === Phase 1/6: server — STARTING ==="

# ── Step 1: Kill previous vLLM ───────────────────────────────────────────
echo "[$(date +%T)] [Step 1] Stopping any previous vLLM (PID file only)..."
# Constraint 3: never touch /opt/, /usr/, pip packages.
# Kill only the PID this skill started — saved to vllm.pid on startup.
# VLLM::EngineCore is a child of that PID and dies with it.
OLD_PID=$(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")
if [ -n "$OLD_PID" ]; then
    echo "  Stopping previous PID=$OLD_PID..."
    # SIGTERM first — allows vLLM uvicorn + EngineCore to clean up ROCm/CUDA contexts
    kill -SIGTERM $OLD_PID 2>/dev/null || true
    for _w in $(seq 1 30); do
        kill -0 $OLD_PID 2>/dev/null || { echo "  Exited cleanly after ${_w}s."; break; }
        sleep 1
    done
    # SIGKILL fallback only if still alive
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "  Still alive — SIGKILL"
        kill -9 $OLD_PID 2>/dev/null || true
        sleep 2
    fi
    rm -f "{{OUTPUT_DIR}}/vllm.pid"
    echo "  Done."
else
    echo "  No previous vLLM PID found (first run)."
fi

# ── Step 2: Select GPU ────────────────────────────────────────────────────
echo "[$(date +%T)] [Step 2] Selecting GPU..."
SELECTED=$(python3 {{SCRIPTS_DIR}}/select_gpus.py {{TP}})
export CUDA_VISIBLE_DEVICES="$SELECTED"
# Save GPU selection to file — used by Phase 4 (tuning) and Phase 5 (restart).
# This avoids reading /proc/{PID}/environ which is fragile and permission-sensitive.
echo "$SELECTED" > "{{OUTPUT_DIR}}/gpu_selection.txt"
python3 -u - << 'PYEOF'
import subprocess, os
selected = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
# Show utilization for selected GPU(s)
r = subprocess.run(['rocm-smi','--showuse'], capture_output=True, text=True, timeout=10)
for line in r.stdout.splitlines():
    for gpu_id in selected.split(','):
        if f'GPU[{gpu_id}]' in line:
            print(f"  Selected GPU {gpu_id}: {line.split(':')[-1].strip()}")
PYEOF
echo "  CUDA_VISIBLE_DEVICES=$SELECTED"
echo "  GPU selection saved → {{OUTPUT_DIR}}/gpu_selection.txt"

# ── Step 3: Environment variables ─────────────────────────────────────────
echo "[$(date +%T)] [Step 3] Setting env vars..."
export HF_ENDPOINT="${HF_ENDPOINT:-http://134.199.133.77}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=1800000
mkdir -p "$HF_HOME"

# TunableOps MUST be set before server start (EngineCore inherits at fork).
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
export PYTORCH_TUNABLEOP_UNTUNED_FILENAME="{{RESULTS_DIR}}/untuned_shapes.csv"
echo "  TunableOps RECORD_UNTUNED → {{RESULTS_DIR}}/untuned_shapes.csv"

# ── Step 4: Check model ────────────────────────────────────────────────────
echo "[$(date +%T)] [Step 4] Checking model..."
python3 -u - << 'PYEOF'
import os, sys, subprocess, json

model = "{{MODEL}}"
if os.path.isdir(model):
    ws = [f for f in os.listdir(model) if f.endswith(('.safetensors','.bin'))]
    cfg_ok = os.path.isfile(os.path.join(model,"config.json"))
    if not ws or not cfg_ok:
        print(f"  ERROR: model incomplete at {model}"); sys.exit(1)
    # Print key config
    try:
        cfg = json.load(open(os.path.join(model,"config.json")))
        tc = cfg.get('text_config', cfg)
        h = tc.get('hidden_size','?')
        l = tc.get('num_hidden_layers','?')
        kv = tc.get('num_key_value_heads','?')
        dtype = tc.get('torch_dtype','?')
        print(f"  Model:       {model}")
        print(f"  Config:      hidden={h}  layers={l}  kv_heads={kv}  dtype={dtype}")
        print(f"  Weights:     {len(ws)} file(s)")
    except Exception as e:
        print(f"  Model: {model} ({len(ws)} weight file(s))")
    open("{{OUTPUT_DIR}}/resolved_model_path.txt","w").write(model)
else:
    import glob as _glob
    name = model.split("/")[-1]
    # Case-insensitive search in /app/ — prevents re-download when cache exists
    # with different capitalization (e.g. Qwen3.5-4b vs Qwen3.5-4B).
    local = None
    for candidate in sorted(_glob.glob("/app/*/")):
        dirname = os.path.basename(candidate.rstrip('/'))
        if dirname.lower() == name.lower():
            cfg = os.path.join(candidate.rstrip('/'), "config.json")
            if os.path.isfile(cfg):
                ws = [f for f in os.listdir(candidate.rstrip('/'))
                      if f.endswith(('.safetensors', '.bin'))]
                if ws:
                    local = candidate.rstrip('/')
                    break
    if local:
        ws = [f for f in os.listdir(local) if f.endswith(('.safetensors', '.bin'))]
        print(f"  Model cached: {local} ({len(ws)} files)")
        open("{{OUTPUT_DIR}}/resolved_model_path.txt","w").write(local)
        sys.exit(0)
    # Not found locally — download
    local = f"/app/{name}"
    print(f"  Downloading: {model} → {local}...")
    r = subprocess.run(["hf","download",model,"--local-dir",local], timeout=3600)
    if r.returncode != 0: print("  ERROR: download failed"); sys.exit(1)
    print(f"  Downloaded: {local}")
    open("{{OUTPUT_DIR}}/resolved_model_path.txt","w").write(local)
PYEOF
MODEL=$(cat "{{OUTPUT_DIR}}/resolved_model_path.txt")

# ── Step 5: Start vLLM with profiler ──────────────────────────────────────
echo "[$(date +%T)] [Step 5] Starting vLLM server..."
VLLM_LOG="{{OUTPUT_DIR}}/vllm_server.log"

PROFILER_JSON=$(python3 -c "
import json
print(json.dumps({'profiler':'torch','torch_profiler_dir':'{{PROFILE_DIR}}',
    'torch_profiler_record_shapes':True,
    'torch_profiler_with_stack':False,'torch_profiler_use_gzip':True,
    'ignore_frontend':True,'max_iterations':{{PROFILE_ITERATIONS}}}))")

echo "  Config: model=$MODEL dtype={{DTYPE}} tp={{TP}} max_len={{MAX_MODEL_LEN}} gpu_util={{GPU_MEM_UTIL}}"
echo "  Profiler: dir={{PROFILE_DIR}} record_shapes=True max_iter={{PROFILE_ITERATIONS}}"
echo "  Log: $VLLM_LOG"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype {{DTYPE}} \
    --tensor-parallel-size {{TP}} --trust-remote-code --enforce-eager \
    --api-key dummy --max-model-len {{MAX_MODEL_LEN}} \
    --gpu-memory-utilization {{GPU_MEM_UTIL}} \
    --profiler-config "$PROFILER_JSON" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo $VLLM_PID > "{{OUTPUT_DIR}}/vllm.pid"
echo "  Started: PID=$VLLM_PID"

# Wait — check alive every 10s, show last relevant log line
echo "[$(date +%T)] [Step 5] Waiting for server ready..."
for i in $(seq 1 36); do
    sleep 10
    kill -0 $VLLM_PID 2>/dev/null || {
        echo "  ERROR: vLLM died at ${i}0s — last log lines:"
        tail -15 "$VLLM_LOG"; exit 1
    }
    if grep -qE 'CUDA out of memory|OOMKilled|ValidationError|ModuleNotFoundError|Address already in use' \
            "$VLLM_LOG" 2>/dev/null; then
        echo "  FATAL error in vLLM log:"
        grep -E 'out of memory|OOMKilled|ValidationError|ModuleNotFoundError|Address already in use' \
            "$VLLM_LOG" | tail -3
        kill -SIGTERM $VLLM_PID 2>/dev/null || true; sleep 3
        kill -0 $VLLM_PID 2>/dev/null && kill -9 $VLLM_PID 2>/dev/null || true; exit 1
    fi
    HTTP=$(curl -s -o/dev/null -w '%{http_code}' \
        -H "Authorization: Bearer dummy" http://localhost:8000/v1/models 2>/dev/null)
    if [ "$HTTP" = "200" ]; then
        echo "  [${i}0s] HTTP=200 — server ready!"
        # Extract key info from vLLM log
        KV_INFO=$(grep -oE 'Maximum concurrency for [0-9,]+ tokens.*' "$VLLM_LOG" | tail -1 || true)
        MEM_INFO=$(grep -oE 'Model loading took [0-9.]+ GiB' "$VLLM_LOG" | tail -1 || true)
        [ -n "$MEM_INFO" ] && echo "  Model: $MEM_INFO"
        [ -n "$KV_INFO"  ] && echo "  KV:    $KV_INFO"
        break
    fi
    [ $i -eq 36 ] && { echo "  ERROR: timeout 360s"; tail -10 "$VLLM_LOG"; exit 1; }
    # Show only meaningful log lines, not the full config dump
    LAST=$(tail -1 "$VLLM_LOG" | sed 's/.*\] //' | cut -c1-120)
    echo "  [${i}0s] $LAST"
done

# ── Step 6: Verify profiler API ────────────────────────────────────────────
echo "[$(date +%T)] [Step 6] Verifying profiler API..."
PROF_RESP=$(curl -s -X POST http://localhost:8000/start_profile \
    -H "Authorization: Bearer dummy" -w "\nHTTP:%{http_code}")
HTTP_CODE=$(echo "$PROF_RESP" | grep -oE 'HTTP:[0-9]+' | cut -d: -f2)
BODY=$(echo "$PROF_RESP" | grep -v '^HTTP:')
echo "  start_profile: HTTP=$HTTP_CODE  body=${BODY:-<empty>}"
if [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "204" ]; then
    echo "  ERROR: profiler API not available (HTTP $HTTP_CODE)"
    echo "  Check that --profiler-config was passed correctly"; exit 1
fi
curl -s -X POST http://localhost:8000/stop_profile -H "Authorization: Bearer dummy" > /dev/null
echo "  Profiler API: OK"

MODEL_ID=$(curl -s -H "Authorization: Bearer dummy" http://localhost:8000/v1/models \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
echo "  Serving model: $MODEL_ID"

# ── Completion ─────────────────────────────────────────────────────────────
echo "[$(date +%T)] === Phase 1/6: server — DONE  pid=$VLLM_PID ==="

} 2>&1 | tee -a "$PHASE_LOG"

export VLLM_PID=$(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':list(dict.fromkeys(p.get('phases_completed',[])+['server'])),
          'current_phase':None,'status':'idle','phase_1_done':datetime.datetime.now().isoformat(),
          'vllm_pid':'$(cat {{OUTPUT_DIR}}/vllm.pid 2>/dev/null)'})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```
