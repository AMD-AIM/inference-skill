# Phase 2: Benchmark & Profile

**Phase name**: `bench-profile`

## Objective
Run the concurrency sweep benchmark AND collect profiler traces under the SAME workload configuration. This is a hard constraint: profiling at a different concurrency than benchmarking produces misleading kernel time percentages.

**Why combined**: At conc=64 attention is ~50% of GPU time. At conc=1 it is ~1.5%. Profiling at conc=1 while optimizing for conc=64 targets the wrong kernel.

---

## Step 1: Benchmark Sweep with GEMM Shape Collection

Run benchmark AND collect real GEMM shapes for TunableOps tuning (Phase 4 Step 0).
`PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` records every unique GEMM shape to a CSV with **zero overhead** — it does not tune, only records.

**WHY collect here**: vLLM uses `BFloat16_TN` format with fused projections (QKV merged, gate+up merged). These shapes cannot be guessed from model config — they must be observed from actual inference. The untuned CSV from this step feeds directly into Phase 4 offline tuning.

```bash
# Untuned shapes will be written to: {{RESULTS_DIR}}/untuned_shapes0.csv
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
export PYTORCH_TUNABLEOP_UNTUNED_FILENAME={{RESULTS_DIR}}/untuned_shapes.csv
echo "TunableOps shape recording enabled (zero overhead)"
```

## Step 2: Benchmark Sweep (no profiler — measure clean throughput)

```bash
python3 << 'PYEOF'
import requests, json, time, os, concurrent.futures, statistics

BASE_URL = "http://localhost:8000/v1"
HEADERS  = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
MODEL    = "{{MODEL}}"
ISL, OSL = {{ISL}}, {{OSL}}

# Generate approximately ISL-token prompt
prompt = "The quick brown fox jumps over the lazy dog. " * (ISL // 10 + 1)

def single_request():
    t0 = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/completions", headers=HEADERS,
            json={"model": MODEL, "prompt": prompt, "max_tokens": OSL, "temperature": 0},
            timeout=600)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            u = resp.json().get("usage", {})
            return {"ok": True, "time": elapsed,
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0)}
        return {"ok": False, "status": resp.status_code, "time": elapsed}
    except Exception as e:
        return {"ok": False, "error": str(e), "time": time.time() - t0}

results = []
for conc in [{{CONCURRENCY_LEVELS}}]:
    n = max(conc * 2, 8)   # at least 2 batches of requests for stable measurement
    print(f"\nBenchmarking conc={conc} ({n} requests)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        t_wall_start = time.time()
        futures = [ex.submit(single_request) for _ in range(n)]
        resps = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall = time.time() - t_wall_start

    oks  = [r for r in resps if r.get("ok")]
    if not oks:
        results.append({"concurrency": conc, "failures": n, "total_tps": 0})
        print(f"  ALL FAILED at conc={conc}")
        continue

    total_in  = sum(r["prompt_tokens"]     for r in oks)
    total_out = sum(r["completion_tokens"] for r in oks)
    latencies = sorted(r["time"] for r in oks)
    n_ok = len(oks)

    def pct(lst, p): idx = max(0, int(len(lst) * p / 100) - 1); return round(lst[idx], 3)

    entry = {
        "concurrency": conc,
        "n_requests": n_ok, "failures": n - n_ok,
        "wall_time_s": round(wall, 2),
        "input_tps":   round(total_in  / wall, 1),
        "output_tps":  round(total_out / wall, 1),
        "total_tps":   round((total_in + total_out) / wall, 1),
        "lat_avg_s":   round(statistics.mean(latencies), 3),
        "lat_p50_s":   pct(latencies, 50),
        "lat_p90_s":   pct(latencies, 90),
        "lat_p99_s":   pct(latencies, 99),
    }
    results.append(entry)
    print(f"  TPS={entry['total_tps']}  output_TPS={entry['output_tps']}  "
          f"lat_avg={entry['lat_avg_s']}s  lat_p90={entry['lat_p90_s']}s")

os.makedirs("{{RESULTS_DIR}}", exist_ok=True)
out = {"config": {"ISL": ISL, "OSL": OSL, "model": MODEL},
       "benchmark_results": results}
with open("{{RESULTS_DIR}}/benchmark_report.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nBenchmark saved: {{RESULTS_DIR}}/benchmark_report.json  ({len(results)} levels)")
PYEOF
```

## Step 2: Profile at Each Concurrency Level (Decode-Only by Default)

Profile at **every** benchmark concurrency level, not just the middle one. Bottleneck shifts with concurrency:
- conc=1: may have GPU idle from CPU scheduling gaps
- conc=16: typically GEMM-bound
- conc=64: GEMM + Attention both significant as KV cache grows

**Default: decode-only** — start profiler AFTER prefill completes so the trace captures steady-state decode performance (most relevant for throughput optimization).

**Advanced (include prefill)**: set `PROFILE_INCLUDE_PREFILL=1` to start profiler simultaneously with requests.

```bash
# Heuristic prefill warmup: ISL tokens / ~500 tok/s + 5s buffer
PREFILL_WAIT=$(python3 -c "print(max(5, int({{ISL}} / 500) + 5))")
echo "Prefill wait time: ${PREFILL_WAIT}s"

# Profile each concurrency level
python3 << 'PYEOF'
import requests, json, time, concurrent.futures, os, glob, shutil

BASE_URL = "http://localhost:8000/v1"
HEADERS  = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
MODEL    = "{{MODEL}}"
ISL, OSL = {{ISL}}, {{OSL}}
PROFILE_DIR = "{{PROFILE_DIR}}"
RESULTS_DIR = "{{RESULTS_DIR}}"
PREFILL_WAIT = int(os.environ.get("PREFILL_WAIT", "5"))
INCLUDE_PREFILL = os.environ.get("PROFILE_INCLUDE_PREFILL", "0") == "1"

prompt = "The quick brown fox jumps over the lazy dog. " * (ISL // 10 + 1)

def clear_traces():
    for f in glob.glob(f"{PROFILE_DIR}/*.json*"):
        os.remove(f)

def start_profiler():
    r = requests.post(f"http://localhost:8000/start_profile",
                      headers={"Authorization": "Bearer dummy"})
    if r.status_code not in (200, 204):
        print(f"  ERROR: start_profile failed: {r.status_code} {r.text[:100]}")
        return False
    return True

def stop_profiler():
    requests.post(f"http://localhost:8000/stop_profile",
                  headers={"Authorization": "Bearer dummy"})

def wait_for_traces(min_size_bytes=100_000, max_wait=60):
    for _ in range(max_wait // 3):
        time.sleep(3)
        traces = glob.glob(f"{PROFILE_DIR}/*.json.gz")
        if traces and sum(os.path.getsize(t) for t in traces) > min_size_bytes:
            return traces
    return glob.glob(f"{PROFILE_DIR}/*.json.gz")

def single_req():
    r = requests.post(f"{BASE_URL}/completions", headers=HEADERS,
        json={"model": MODEL, "prompt": prompt, "max_tokens": OSL, "temperature": 0},
        timeout=600)
    return r.status_code

profile_meta = {}
for conc in [{{CONCURRENCY_LEVELS}}]:
    n = max(conc * 2, 4)
    dest = f"{PROFILE_DIR}/conc_{conc}"
    os.makedirs(dest, exist_ok=True)
    clear_traces()

    print(f"\nProfiling conc={conc} (n={n}, decode_only={not INCLUDE_PREFILL})...")

    if INCLUDE_PREFILL:
        # Advanced: capture prefill+decode
        start_profiler()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            codes = [f.result() for f in [ex.submit(single_req) for _ in range(n)]]
    else:
        # Default: decode-only — fire requests, wait for prefill to clear, then profile
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            futures = [ex.submit(single_req) for _ in range(n)]
            time.sleep(PREFILL_WAIT)      # wait for prefill to finish
            start_profiler()
            codes = [f.result() for f in futures]

    time.sleep(2)
    stop_profiler()

    traces = wait_for_traces()
    sz = sum(os.path.getsize(t) for t in traces)
    oks = sum(1 for c in codes if c == 200)
    print(f"  {oks}/{n} ok, {len(traces)} trace(s), {sz/1e6:.1f}MB")

    for t in traces:
        shutil.move(t, dest)

    profile_meta[conc] = {
        "conc": conc, "n_requests": n, "n_ok": oks,
        "decode_only": not INCLUDE_PREFILL,
        "trace_dir": dest,
    }

import json as _json
with open(f"{RESULTS_DIR}/profile_meta.json", "w") as f:
    _json.dump(profile_meta, f, indent=2)
print(f"\nProfile meta saved: {RESULTS_DIR}/profile_meta.json")
PYEOF

# Validate traces
python3 - << 'PYEOF'
import glob, os, json, gzip

meta = json.load(open('{{RESULTS_DIR}}/profile_meta.json'))
for conc, m in sorted(meta.items(), key=lambda x: int(x[0])):
    traces = glob.glob(m['trace_dir'] + '/*.json.gz')
    sz = sum(os.path.getsize(t) for t in traces)
    print(f'  conc={conc}: {len(traces)} trace(s), {sz/1e6:.1f}MB  ({m["n_ok"]}/{m["n_requests"]} ok)')
    if not traces:
        print(f'    WARNING: no traces for conc={conc}')
        continue
    # Validate: check traceEvents present and kernel events exist
    t = max(traces, key=os.path.getsize)
    with gzip.open(t, 'rt') as f:
        peek = f.read(8192)
    has_trace_events = '"traceEvents"' in peek
    print(f'    {os.path.basename(t)}: traceEvents={has_trace_events}')
    if not has_trace_events:
        print(f'    WARNING: trace may be empty or invalid')
PYEOF
```

## Step 4 (after profile stop): Validate Untuned Shapes Were Collected

```bash
python3 << 'PYEOF'
import glob, os

# PyTorch appends process ordinal to filename: untuned_shapes.csv → untuned_shapes0.csv
results_dir = "{{RESULTS_DIR}}"
candidates = glob.glob(f"{results_dir}/untuned_shapes*.csv")
if candidates:
    f = max(candidates, key=os.path.getsize)
    n = sum(1 for l in open(f) if l.startswith("Gemm"))
    print(f"Untuned GEMM shapes collected: {n} unique shapes in {os.path.basename(f)}")
    import os as _os; _os.rename(f, f"{results_dir}/untuned_shapes_final.csv")
    print(f"Saved as: untuned_shapes_final.csv")
else:
    print("WARNING: No untuned shapes CSV found — PYTORCH_TUNABLEOP_RECORD_UNTUNED may not be set")
PYEOF

# Unset RECORD_UNTUNED for profiling step (no double-counting)
unset PYTORCH_TUNABLEOP_RECORD_UNTUNED
```

## Completion

Update `{{PROGRESS_FILE}}`:
```json
{"phases_completed": ["env", "server", "bench-profile"],
 "details": {"benchmark_levels": [{{CONCURRENCY_LEVELS}}], "profile_concurrency": "<middle>",
             "trace_dir": "{{PROFILE_DIR}}",
             "untuned_shapes": "{{RESULTS_DIR}}/untuned_shapes_final.csv"}}
```
