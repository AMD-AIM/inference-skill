# Phase 2: Benchmark & Profile

**Phase name**: `bench-profile`  **Phase number**: 2

## Objective
Concurrency sweep benchmark (GEMM shapes collected automatically), then decode-only profiler traces at each concurrency level.

**Default: decode-only** — profiler starts after prefill clears. Set `PROFILE_INCLUDE_PREFILL=1` to capture prefill too.

**GEMM shapes collected via `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`** set in Phase 1 before server start.

## Execution

```bash
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_2_bench.log"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'bench-profile','status':'running','phase_2_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail

echo "[$(date +%T)] === Phase 2/6: bench-profile — STARTING ==="

# Config summary
echo "[$(date +%T)] Config:"
echo "  Model:       {{MODEL}}"
echo "  ISL x OSL:   {{ISL}} x {{OSL}}"
echo "  Concurrency: [{{CONCURRENCY_LEVELS}}]"
echo "  Dtype:       {{DTYPE}}  TP={{TP}}"
echo "  TunableOps recording: PYTORCH_TUNABLEOP_RECORD_UNTUNED=${PYTORCH_TUNABLEOP_RECORD_UNTUNED:-NOT SET}"
[ -z "${PYTORCH_TUNABLEOP_RECORD_UNTUNED:-}" ] && \
    echo "  WARNING: shape recording not active — run Phase 1 in the same shell session"

# ── Step 1: Benchmark sweep ────────────────────────────────────────────────
echo "[$(date +%T)] [Step 1] Benchmark sweep..."
T_BENCH_START=$(date +%s)

python3 -u - << 'PYEOF'
import requests, json, time, os, concurrent.futures

BASE_URL = "http://localhost:8000/v1"
HEADERS  = {"Authorization":"Bearer dummy","Content-Type":"application/json"}
MODEL    = "{{MODEL}}"
ISL, OSL = {{ISL}}, {{OSL}}
prompt   = "The quick brown fox jumps over the lazy dog. " * (ISL // 10 + 1)

def req():
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/completions", headers=HEADERS,
            json={"model":MODEL,"prompt":prompt,"max_tokens":OSL,"temperature":0,"ignore_eos":True},
            timeout=900)
        if r.status_code == 200:
            u = r.json().get("usage",{})
            return {"ok":True,"time":time.time()-t0,
                    "pt":u.get("prompt_tokens",0),"ct":u.get("completion_tokens",0)}
        return {"ok":False,"status":r.status_code}
    except Exception as e:
        return {"ok":False,"error":str(e)}

results = []
for conc in [{{CONCURRENCY_LEVELS}}]:
    n = max(conc*2, 8)
    print(f"  conc={conc:3d}: sending {n} requests...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        t0 = time.time()
        oks = [r for r in [f.result() for f in [ex.submit(req) for _ in range(n)]] if r.get("ok")]
    wall = time.time()-t0
    if not oks:
        print(f"  conc={conc:3d}: ALL FAILED  [FAIL]", flush=True)
        results.append({"concurrency":conc,"total_tps":0,"failures":n}); continue

    out_tps = sum(r["ct"] for r in oks)/wall
    tot_tps = sum(r["pt"]+r["ct"] for r in oks)/wall
    lats    = sorted(r["time"] for r in oks)
    fails   = n - len(oks)
    entry   = {"concurrency":conc,"n_requests":len(oks),"failures":fails,
               "wall_time_s":round(wall,2),"output_tps":round(out_tps,1),
               "total_tps":round(tot_tps,1),
               "lat_p50_s":round(lats[len(lats)//2],3),
               "lat_p90_s":round(lats[int(len(lats)*.9)],3)}
    results.append(entry)
    status = "[FAIL: partial]" if fails > 0 else "[OK]"
    print(f"  conc={conc:3d}: output_tps={out_tps:7.1f}  total_tps={tot_tps:7.1f}"
          f"  p50={entry['lat_p50_s']:.2f}s  p90={entry['lat_p90_s']:.2f}s"
          f"  ok={len(oks)}/{n}  {status}", flush=True)

os.makedirs("{{RESULTS_DIR}}", exist_ok=True)
with open("{{RESULTS_DIR}}/benchmark_report.json","w") as f:
    json.dump({"config":{"ISL":ISL,"OSL":OSL,"model":MODEL},"benchmark_results":results},f,indent=2)
print(f"  Saved: {{RESULTS_DIR}}/benchmark_report.json")
PYEOF

T_BENCH_END=$(date +%s)
echo "[$(date +%T)] [Step 1] Benchmark done. ($(( T_BENCH_END - T_BENCH_START ))s elapsed)"

# ── Step 2: Profile at each concurrency ───────────────────────────────────
echo "[$(date +%T)] [Step 2] Profiling (decode-only by default)..."
T_PROF_START=$(date +%s)

PREFILL_WAIT=$(python3 -c "print(max(5, int({{ISL}}/500)+5))")
INCLUDE_PREFILL="${PROFILE_INCLUDE_PREFILL:-0}"
echo "  prefill_wait=${PREFILL_WAIT}s  include_prefill=${INCLUDE_PREFILL}"

python3 -u - << 'PYEOF'
import requests, json, time, concurrent.futures, os, glob, shutil

BASE_URL = "http://localhost:8000/v1"
HEADERS  = {"Authorization":"Bearer dummy","Content-Type":"application/json"}
MODEL    = "{{MODEL}}"
ISL, OSL = {{ISL}}, {{OSL}}
PROFILE_DIR = "{{PROFILE_DIR}}"
RESULTS_DIR = "{{RESULTS_DIR}}"
PREFILL_WAIT    = int(os.environ.get("PREFILL_WAIT","5"))
INCLUDE_PREFILL = os.environ.get("PROFILE_INCLUDE_PREFILL","0") == "1"
prompt = "The quick brown fox jumps over the lazy dog. " * (ISL // 10 + 1)

def clear_base_traces():
    """Remove only files directly in PROFILE_DIR (not subdirectories)."""
    for f in glob.glob(f"{PROFILE_DIR}/*.json*"): os.remove(f)

def start_profiler():
    r = requests.post("http://localhost:8000/start_profile",
                      headers={"Authorization":"Bearer dummy"})
    ok = r.status_code in (200,204)
    print(f"    start_profile: HTTP={r.status_code}  "
          f"body={r.text[:80] if r.text else '<empty>'}  {'[OK]' if ok else '[FAIL]'}", flush=True)
    return ok

def stop_profiler():
    r = requests.post("http://localhost:8000/stop_profile",
                      headers={"Authorization":"Bearer dummy"})
    print(f"    stop_profile:  HTTP={r.status_code}", flush=True)

def do_req():
    r = requests.post(f"{BASE_URL}/completions", headers=HEADERS,
        json={"model":MODEL,"prompt":prompt,"max_tokens":OSL,"temperature":0,"ignore_eos":True},
        timeout=900)
    return r.status_code

def wait_for_traces(dest_dir, min_bytes=50_000, max_wait=60):
    """Wait for trace files to appear in PROFILE_DIR, then return them."""
    for _ in range(max_wait//3):
        time.sleep(3)
        ts = glob.glob(f"{PROFILE_DIR}/*.json.gz")
        if ts and sum(os.path.getsize(t) for t in ts) > min_bytes:
            return ts
    return glob.glob(f"{PROFILE_DIR}/*.json.gz")

meta = {}
for conc in [{{CONCURRENCY_LEVELS}}]:
    n = max(conc*2, 4)
    dest = os.path.join(PROFILE_DIR, f"conc_{conc}")
    os.makedirs(dest, exist_ok=True)
    clear_base_traces()

    mode = "prefill+decode" if INCLUDE_PREFILL else "decode-only"
    print(f"\n  Profiling conc={conc} n={n} [{mode}]...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futures = [ex.submit(do_req) for _ in range(n)]
        if not INCLUDE_PREFILL:
            print(f"    waiting {PREFILL_WAIT}s for prefill to clear...", flush=True)
            time.sleep(PREFILL_WAIT)
        ok = start_profiler()
        if not ok:
            print(f"    WARNING: profiler start failed for conc={conc}", flush=True)
        codes = [f.result() for f in futures]

    time.sleep(2)
    stop_profiler()
    print(f"    waiting for trace flush...", flush=True)

    traces = wait_for_traces(dest)
    sz = sum(os.path.getsize(t) for t in traces)
    oks_req = sum(1 for c in codes if c==200)
    print(f"    requests: {oks_req}/{n} ok  |  traces: {len(traces)} file(s)  {sz/1e6:.1f}MB",
          flush=True)

    if not traces:
        print(f"    WARNING: No trace files found for conc={conc} — profiling may have failed",
              flush=True)

    # Move to per-concurrency subdirectory
    moved = []
    for t in traces:
        dst = os.path.join(dest, os.path.basename(t))
        shutil.move(t, dst)
        moved.append(dst)
    print(f"    Traces saved → {dest}/", flush=True)

    meta[str(conc)] = {"conc":conc,"n_requests":n,"n_ok":oks_req,
                       "decode_only":not INCLUDE_PREFILL,
                       "trace_dir":dest,
                       "trace_count":len(moved),
                       "trace_size_mb":round(sz/1e6,1)}

with open(f"{RESULTS_DIR}/profile_meta.json","w") as f:
    json.dump(meta, f, indent=2)
print(f"\n  profile_meta.json saved.")
PYEOF

T_PROF_END=$(date +%s)
echo "[$(date +%T)] [Step 2] Profiling done. ($(( T_PROF_END - T_PROF_START ))s elapsed)"

# ── Step 3: Validate results ───────────────────────────────────────────────
echo "[$(date +%T)] [Step 3] Validating traces and GEMM shapes..."

python3 -u - << 'PYEOF'
import glob, os, gzip, json, sys

all_ok = True

# Validate traces — check the MOVED locations (conc_N subdirs)
meta = json.load(open("{{RESULTS_DIR}}/profile_meta.json"))
print("  Trace validation:")
for cs, m in sorted(meta.items(), key=lambda x: int(x[0])):
    dest = m["trace_dir"]
    ts = glob.glob(dest + "/*.json.gz")
    sz = sum(os.path.getsize(t) for t in ts)
    if not ts:
        print(f"    conc={cs:>3}: [FAIL] No traces in {dest}/")
        all_ok = False
        continue
    # Check traceEvents — read enough of the file to find the key
    # Note: trace files are large (10-20MB gzipped); key appears early in metadata
    t = max(ts, key=os.path.getsize)
    with gzip.open(t, "rt") as f:
        content = f.read(65536)  # 64KB — enough to find "traceEvents" in header
    has_events = '"traceEvents"' in content
    # File size is the primary sanity check; kernel count needs full file scan (too slow here)
    size_ok = sz > 1_000_000  # at least 1MB
    mode = "decode-only" if m.get("decode_only") else "prefill+decode"
    status = "[OK]" if has_events and size_ok else \
             "[WARN: no traceEvents]" if not has_events else "[WARN: file too small]"
    print(f"    conc={cs:>3}: {len(ts)} trace(s) {sz/1e6:.1f}MB  "
          f"traceEvents={'YES' if has_events else 'NO'}  [{mode}]  {status}")
    if not has_events:
        all_ok = False

# Validate untuned shapes
# PyTorch appends process ordinal suffix: .csv → 0.csv
cands = sorted(glob.glob("{{RESULTS_DIR}}/untuned_shapes*.csv"), key=os.path.getsize, reverse=True)
if cands:
    best = cands[0]
    n_shapes = sum(1 for l in open(best) if l.startswith("Gemm"))
    final = "{{RESULTS_DIR}}/untuned_shapes_final.csv"
    if best != final:
        os.rename(best, final)
    m_values = set()
    for line in open(final):
        if line.startswith("Gemm"):
            parts = line.strip().split(",")
            if len(parts) >= 2:
                # Format: GemmTunableOp_BFloat16_TN,tn_N_M_K_...
                shape_str = parts[1]
                nums = [x for x in shape_str.split('_') if x.isdigit()]
                if len(nums) >= 2:
                    m_values.add(int(nums[1]))  # M is second number
    status = "[OK]" if n_shapes > 0 else "[FAIL: 0 shapes — check TunableOps env vars in Phase 1]"
    print(f"\n  GEMM shape collection: {n_shapes} unique shapes  M_values={sorted(m_values)}  {status}")
    print(f"  First 3 shapes:")
    count = 0
    for l in open(final):
        if l.startswith("Gemm") and count < 3:
            print(f"    {l.strip()[:100]}")
            count += 1
    if n_shapes == 0:
        all_ok = False
else:
    print("  GEMM shapes: [FAIL] No untuned_shapes CSV found")
    print("  Fix: PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 must be set BEFORE server start (Phase 1)")
    all_ok = False

if not all_ok:
    print("\n  Some validations failed — check warnings above before proceeding to Phase 3")
    sys.exit(1)
else:
    print("\n  All validations passed [OK]")
PYEOF

# ── Completion ─────────────────────────────────────────────────────────────
N_SHAPES=$(python3 -c "
import os,glob
f='{{RESULTS_DIR}}/untuned_shapes_final.csv'
print(sum(1 for l in open(f) if l.startswith('Gemm')) if os.path.exists(f) else 0)
" 2>/dev/null || echo "?")

echo "[$(date +%T)] === Phase 2/6: bench-profile — DONE  untuned_shapes=${N_SHAPES} ==="

} 2>&1 | tee -a "$PHASE_LOG"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':p.get('phases_completed',[])+['bench-profile'],
          'current_phase':None,'status':'idle','phase_2_done':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```
