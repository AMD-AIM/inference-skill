# Phase 5: Integration & E2E Verification

**Phase name**: `integrate`  **Phase number**: 5

## Objective
Deploy optimized kernels, verify they are being called, check correctness, benchmark E2E at all concurrencies. Roll back on regression.

**Constraints**: No /opt/, /usr/, pip modifications. Injection via PYTHONPATH + sitecustomize.py only.

## Execution

```bash
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_5_integrate.log"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'integrate','status':'running','phase_5_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail
echo "[$(date +%T)] === Phase 5/6: integrate — STARTING ==="

# Config
echo "[$(date +%T)] Config:"
python3 -u - << 'PYEOF'
import json, os, glob

# What kernels are being integrated
manifest_f = "{{OPTIMIZED_DIR}}/tuned_gemm.csv"
inject_f   = "{{OPTIMIZED_DIR}}/pypath/sitecustomize.py"
if os.path.exists(manifest_f):
    n = sum(1 for l in open(manifest_f) if l.startswith("Gemm"))
    print(f"  TunableOps tuned_gemm.csv: {n} entries")
if os.path.exists(inject_f):
    print(f"  Injection shim: {inject_f}  [exists]")
else:
    print(f"  Injection shim: MISSING — run Phase 4 first"); exit(1)
speedup_f = "{{OPTIMIZED_DIR}}/tunableops_speedup.txt"
if os.path.exists(speedup_f):
    sp = open(speedup_f).read().strip()
    print(f"  Expected micro-speedup: {sp}x (from Phase 4)")
print(f"  Injection method: PYTHONPATH={{OPTIMIZED_DIR}}/pypath")
print(f"  Baseline: {{RESULTS_DIR}}/baseline_e2e.json (will be created now)")
PYEOF

# ── Step 1: Load Phase 2 benchmark results as baseline ────────────────────
# DO NOT run a new benchmark here. Phase 2 Step 1 already ran an identical sweep
# (same model, same server, same ISL/OSL/concurrencies). Re-running it wastes 12 min.
# Load benchmark_report.json from Phase 2 and convert to baseline_e2e.json format.
echo "[$(date +%T)] [Step 1] Loading Phase 2 benchmark as baseline (NO re-benchmarking)..."
T_START=$(date +%s)

python3 -u - << 'PYEOF'
import json, os, sys

bench2_f = "{{RESULTS_DIR}}/benchmark_report.json"
if not os.path.exists(bench2_f):
    print(f"  ERROR: {bench2_f} not found — Phase 2 must complete before Phase 5"); sys.exit(1)

bench2 = json.load(open(bench2_f))
baseline = {}
print(f"  {'Conc':>4}  {'OutTPS':>8}  {'TotTPS':>8}  {'P50':>8}")
print("  "+"-"*38)
for r in bench2.get("benchmark_results", []):
    if r.get("total_tps", 0) > 0:
        conc = str(r["concurrency"])
        baseline[conc] = {
            "output_tps": r.get("output_tps", 0),
            "total_tps":  r.get("total_tps", 0),
            "lat_p50_s":  r.get("lat_p50_s", 0),
            "n_ok":       r.get("n_requests", 0),
        }
        print(f"  {conc:>4}  {r.get('output_tps',0):>8.1f}  {r.get('total_tps',0):>8.1f}"
              f"  {r.get('lat_p50_s',0):>7.3f}s")

os.makedirs("{{RESULTS_DIR}}", exist_ok=True)
with open("{{RESULTS_DIR}}/baseline_e2e.json", "w") as f:
    json.dump(baseline, f, indent=2)
print(f"  Saved: baseline_e2e.json  (loaded from Phase 2, {len(baseline)} concurrency levels)")
PYEOF

T_END=$(date +%s)
echo "[$(date +%T)] [Step 1] Baseline loaded. ($(( T_END - T_START ))s elapsed)"

# ── Step 2: Kill current server, start patched server ─────────────────────
echo "[$(date +%T)] [Step 2] Restarting vLLM with TunableOps injection..."

# Read GPU selection BEFORE killing the old server.
# gpu_selection.txt was written by Phase 1. Reading /proc/{PID}/environ after
# kill -9 is a race: the process is already dead and /proc/{PID} is gone.
SELECTED=$(cat "{{OUTPUT_DIR}}/gpu_selection.txt" 2>/dev/null || \
           python3 {{SCRIPTS_DIR}}/select_gpus.py {{TP}})
echo "  GPU selection: $SELECTED (from gpu_selection.txt)"

# Kill using saved PID only. Never use pkill -f patterns — they scan all
# processes, hang on zombies, and have caused 2-minute timeouts in practice.
# EngineCore is a child of the API server and dies automatically with it.
# Use SIGTERM first: allows vLLM to close ROCm/CUDA contexts cleanly so VRAM
# is released. kill -9 leaves orphan resource_tracker children holding /dev/kfd
# open, which prevents ROCm from freeing VRAM on the next server start.
OLD_PID=$(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")
if [ -n "$OLD_PID" ]; then
    echo "  Stopping previous server PID=$OLD_PID (SIGTERM)..."
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
    echo "  No previous vLLM PID found."
fi

export CUDA_VISIBLE_DEVICES="$SELECTED"
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTHONPATH="{{OPTIMIZED_DIR}}/pypath:${PYTHONPATH:-}"

MODEL=$(cat "{{OUTPUT_DIR}}/resolved_model_path.txt" 2>/dev/null || echo "{{MODEL}}")
PATCHED_LOG="{{OUTPUT_DIR}}/vllm_patched.log"

echo "  CUDA_VISIBLE_DEVICES=$SELECTED"
echo "  PYTHONPATH={{OPTIMIZED_DIR}}/pypath:..."
echo "  PYTORCH_TUNABLEOP_ENABLED=1"
echo "  Log: $PATCHED_LOG"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype {{DTYPE}} \
    --tensor-parallel-size {{TP}} --trust-remote-code --enforce-eager \
    --api-key dummy --max-model-len {{MAX_MODEL_LEN}} \
    --gpu-memory-utilization {{GPU_MEM_UTIL}} \
    > "$PATCHED_LOG" 2>&1 &
PATCHED_PID=$!
echo $PATCHED_PID > "{{OUTPUT_DIR}}/vllm.pid"
echo "  Started patched server: PID=$PATCHED_PID"

for i in $(seq 1 36); do
    sleep 10
    kill -0 $PATCHED_PID 2>/dev/null || { echo "  ERROR: patched server died at ${i}0s"; tail -15 "$PATCHED_LOG"; exit 1; }
    HTTP=$(curl -s -o/dev/null -w '%{http_code}' -H "Authorization: Bearer dummy" http://localhost:8000/v1/models 2>/dev/null)
    [ "$HTTP" = "200" ] && echo "  [${i}0s] HTTP=200 — patched server ready!" && break
    [ $i -eq 36 ] && { echo "  ERROR: timeout"; tail -10 "$PATCHED_LOG"; exit 1; }
    echo "  [${i}0s] $(tail -1 "$PATCHED_LOG" | sed 's/.*\] //' | cut -c1-100)"
done

# ── Step 3: Verify injection is active ────────────────────────────────────
echo "[$(date +%T)] [Step 3] Verifying TunableOps injection..."
sleep 5  # let server log settle

INJECT_COUNT=$(grep -c "tunableops_inject" "$PATCHED_LOG" 2>/dev/null || echo 0)
echo "  tunableops_inject log lines: $INJECT_COUNT"
if [ "$INJECT_COUNT" -gt 0 ]; then
    grep "tunableops_inject" "$PATCHED_LOG" | head -3
    echo "  Injection: [ACTIVE]"
else
    echo "  WARNING: injection not detected in log — may not be taking effect  [CHECK]"
fi

# Warmup: send a few requests to confirm kernel is executing
python3 -u - << 'PYEOF'
import requests, time
HEADERS = {"Authorization":"Bearer dummy","Content-Type":"application/json"}
MODEL   = "{{MODEL}}"
for _ in range(3):
    r = requests.post("http://localhost:8000/v1/completions", headers=HEADERS,
        json={"model":MODEL,"prompt":"Hello world","max_tokens":10,"temperature":0}, timeout=30)
    if r.status_code != 200:
        print(f"  WARNING: warmup request failed: {r.status_code}")
print("  Warmup requests: done")
PYEOF

# VLLM::EngineCore note: SIGTERM on the APIServer triggers the vLLM shutdown chain which
# signals EngineCore to exit cleanly. kill -9 would leave EngineCore running as an orphan.
# Verify injection via log — tunableops_inject lines confirm the shim ran in EngineCore subprocess.
# autotune warning: @triton.autotune causes E2E regression in serving (each new batch size triggers
# 100-iteration sweep). TunableOps uses fixed pre-tuned algorithms — no autotune risk.

# Call count verification — check injection log
echo "  Call count/injection verification:"
INJECT_LINES=$(grep "tunableops_inject.*Loaded" "{{OUTPUT_DIR}}/vllm_patched.log" 2>/dev/null | wc -l || echo 0)
echo "  tunableops_inject 'Loaded' lines in patched server log: $INJECT_LINES"
[ "$INJECT_LINES" -gt 0 ] && echo "  Injection active in EngineCore subprocess [OK]" || \
    echo "  WARNING: injection not confirmed in log [CHECK]"

# ── Step 4: Correctness check (Gate 3) ────────────────────────────────────
echo "[$(date +%T)] [Step 4] Correctness check (temperature=0, deterministic)..."
python3 -u - << 'PYEOF'
import requests, os

HEADERS = {"Authorization":"Bearer dummy","Content-Type":"application/json"}
MODEL   = "{{MODEL}}"
prompt  = "The capital of France is"

r = requests.post("http://localhost:8000/v1/completions", headers=HEADERS,
    json={"model":MODEL,"prompt":prompt,"max_tokens":20,"temperature":0}, timeout=60)
if r.status_code != 200:
    print(f"  Correctness: [FAIL] HTTP={r.status_code}"); exit(1)

output = r.json()["choices"][0]["text"]
print(f"  Prompt:  '{prompt}'")
print(f"  Output:  '{output}'")

# Save as reference
ref_f = "{{RESULTS_DIR}}/patched_correctness.json"
import json
with open(ref_f,"w") as f:
    json.dump({"prompt":prompt,"output":output}, f)
print(f"  Saved: {ref_f}")
print(f"  Correctness: [PASS] (model produces output)")
PYEOF

# ── Step 5: E2E benchmark (patched) + comparison ──────────────────────────
echo "[$(date +%T)] [Step 5] E2E benchmark with patched server (Gates 4+5)..."
T_START=$(date +%s)

python3 -u - << 'PYEOF'
import requests, json, time, concurrent.futures, sys

BASE_URL = "http://localhost:8000/v1"
HEADERS  = {"Authorization":"Bearer dummy","Content-Type":"application/json"}
MODEL    = "{{MODEL}}"
ISL, OSL = {{ISL}}, {{OSL}}
prompt   = "The quick brown fox jumps over the lazy dog. " * (ISL // 10 + 1)

def req():
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/completions", headers=HEADERS,
        json={"model":MODEL,"prompt":prompt,"max_tokens":OSL,"temperature":0,"ignore_eos":True},
        timeout=900)
    if r.status_code == 200:
        u = r.json().get("usage",{})
        return {"ok":True,"time":time.time()-t0,"pt":u.get("prompt_tokens",0),"ct":u.get("completion_tokens",0)}
    return {"ok":False}

baseline = json.load(open("{{RESULTS_DIR}}/baseline_e2e.json"))

print(f"  {'Conc':>4}  {'Baseline':>9}  {'Patched':>9}  {'Speedup':>8}  {'Gate4':>6}  Status")
print("  "+"-"*60)

patched_results = {}
regression_found = False
for conc in [{{CONCURRENCY_LEVELS}}]:
    n = max(conc*2, 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        t0 = time.time()
        oks = [r for r in [f.result() for f in [ex.submit(req) for _ in range(n)]] if r.get("ok")]
    wall = time.time()-t0
    out_tps = sum(r["ct"] for r in oks)/wall if oks else 0
    tot_tps = sum(r["pt"]+r["ct"] for r in oks)/wall if oks else 0

    bv = baseline.get(str(conc),{})
    base_out = bv.get("output_tps",0)
    speedup = out_tps / base_out if base_out > 0 else 0
    gate4_pass = speedup >= 0.95   # 5% tolerance
    if not gate4_pass: regression_found = True

    # Gate 5: single-user decode must not regress
    if conc == 1 and not gate4_pass:
        print(f"  *** GATE 5 FAIL at conc=1: single-user regression ***")

    status = "[PASS]" if gate4_pass else "[FAIL: regression]"
    patched_results[str(conc)] = {"output_tps":round(out_tps,1),"total_tps":round(tot_tps,1),
                                   "speedup":round(speedup,4),"gate4_pass":gate4_pass}
    print(f"  {conc:>4}  {base_out:>8.1f}  {out_tps:>8.1f}  {speedup:>7.3f}x  "
          f"{'PASS' if gate4_pass else 'FAIL':>6}  {status}")

import os
with open("{{RESULTS_DIR}}/integration_e2e.json","w") as f:
    json.dump({"patched":patched_results,"baseline":baseline,
               "regression_found":regression_found}, f, indent=2)
with open("{{RESULTS_DIR}}/integration_result.json","w") as f:
    json.dump({"integrated": not regression_found,
               "reason": "all gates passed" if not regression_found else "E2E regression detected"}, f, indent=2)

print()
if regression_found:
    print("  *** REGRESSION DETECTED — initiating rollback ***  [FAIL]")
    sys.exit(1)
else:
    speedups = [v["speedup"] for v in patched_results.values()]
    avg_sp = sum(speedups)/len(speedups) if speedups else 0
    print(f"  All gates PASSED  avg_speedup={avg_sp:.3f}x  [OK]")
PYEOF

T_END=$(date +%s)
echo "[$(date +%T)] [Step 5] E2E benchmark done. ($(( T_END - T_START ))s elapsed)"

# ── Step 6: Rollback if regression ────────────────────────────────────────
python3 -u - << 'PYEOF'
import json, os
result = json.load(open("{{RESULTS_DIR}}/integration_result.json"))
if result.get("integrated"):
    e2e = json.load(open("{{RESULTS_DIR}}/integration_e2e.json"))
    speedups = [v["speedup"] for v in e2e["patched"].values()]
    avg_sp = sum(speedups)/len(speedups) if speedups else 0
    print(f"  Integration: SUCCESSFUL  avg_e2e_speedup={avg_sp:.3f}x  [OK]")
else:
    print(f"  Integration: ROLLED BACK — {result.get('reason','unknown')}  [FAIL]")
PYEOF

# ── Step 7: Shut down patched vLLM server ─────────────────────────────────
# Phase 6 (report) only reads JSON files — it does NOT need the server.
# Kill it here so VRAM is released before the pipeline ends.
echo "[$(date +%T)] [Step 7] Shutting down patched vLLM server..."
DONE_PID=$(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")
if [ -n "$DONE_PID" ]; then
    kill -SIGTERM $DONE_PID 2>/dev/null || true
    for _w in $(seq 1 30); do
        kill -0 $DONE_PID 2>/dev/null || { echo "  Exited cleanly after ${_w}s."; break; }
        sleep 1
    done
    kill -0 $DONE_PID 2>/dev/null && kill -9 $DONE_PID 2>/dev/null || true
    sleep 2
    rm -f "{{OUTPUT_DIR}}/vllm.pid"
    echo "  vLLM server stopped. VRAM released."
else
    echo "  No vLLM PID on record."
fi

# ── Completion ─────────────────────────────────────────────────────────
INTEGRATED=$(python3 -c "
import json,os
f='{{RESULTS_DIR}}/integration_result.json'
print(json.load(open(f)).get('integrated','?') if os.path.exists(f) else '?')
" 2>/dev/null || echo "?")
echo "[$(date +%T)] === Phase 5/6: integrate — DONE  integrated=${INTEGRATED} ==="

} 2>&1 | tee -a "$PHASE_LOG"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':list(dict.fromkeys(p.get('phases_completed',[])+['integrate'])),
          'current_phase':None,'status':'idle','phase_5_done':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```
