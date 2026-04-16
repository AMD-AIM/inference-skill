# Phase 2: Benchmark & Profile {{SKIP_LABEL}}

## Objective
Run benchmark AND collect profiler traces under the SAME workload. This ensures kernel time percentages match the actual target workload (ISL, OSL, concurrency).

## Why combined
Separated benchmark and profiling produces **inconsistent data**: benchmark runs at concurrency=4,16,64 but profiling runs at concurrency=1. At concurrency=64, attention can be 50% of GPU time vs 1.5% at concurrency=1. Optimizing based on conc=1 profile data would target the wrong bottleneck.

## Steps

### 1. Prepare Test Prompts
Generate prompts of approximately `{{ISL}}` tokens:
```python
prompt = "The quick brown fox jumps over the lazy dog. " * ({{ISL}} // 10)
```

### 2. Benchmark WITHOUT profiler (baseline throughput)
Run the concurrency sweep to measure clean throughput (profiler adds overhead):

```bash
python3 << 'PYEOF'
import requests, json, time, os, concurrent.futures

BASE_URL = "http://localhost:8000/v1"
MODEL = "{{MODEL}}"
HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
ISL = {{ISL}}
OSL = {{OSL}}

prompt = "The quick brown fox jumps over the lazy dog. " * (ISL // 10)

def make_request():
    t0 = time.time()
    try:
        resp = requests.post(f"{BASE_URL}/completions", headers=HEADERS,
            json={"model": MODEL, "prompt": prompt, "max_tokens": OSL, "temperature": 0},
            timeout=600)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            usage = resp.json().get("usage", {})
            return {"ok": True, "time": elapsed,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)}
        return {"ok": False, "code": resp.status_code, "time": elapsed}
    except Exception as e:
        return {"ok": False, "error": str(e), "time": time.time() - t0}

results = []
for conc in [{{CONCURRENCY_LEVELS}}]:
    n_req = max(conc, 4)
    print(f"Benchmarking concurrency={conc}, {n_req} requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        resps = [f.result() for f in [ex.submit(make_request) for _ in range(n_req)]]
    oks = [r for r in resps if r.get("ok")]
    fails = len(resps) - len(oks)
    if oks:
        wall = max(r["time"] for r in oks)
        total_in = sum(r["prompt_tokens"] for r in oks)
        total_out = sum(r["completion_tokens"] for r in oks)
        entry = {
            "concurrency": conc, "n_requests": len(oks), "failures": fails,
            "wall_time_s": round(wall, 2),
            "input_tps": round(total_in / wall, 1),
            "output_tps": round(total_out / wall, 1),
            "total_tps": round((total_in + total_out) / wall, 1),
            "avg_latency_s": round(sum(r["time"] for r in oks) / len(oks), 2),
            "avg_prompt_tokens": round(total_in / len(oks), 1),
            "avg_completion_tokens": round(total_out / len(oks), 1),
        }
        results.append(entry)
        print(f"  {entry['total_tps']} tok/s, {entry['output_tps']} out tok/s, lat={entry['avg_latency_s']}s")

with open("{{OUTPUT_DIR}}/results/benchmark_report.json", "w") as f:
    json.dump({"benchmark_results": results, "config": {"ISL": ISL, "OSL": OSL}}, f, indent=2)
print(f"Benchmark saved: {len(results)} concurrency levels")
PYEOF
```

### 3. Profile AT representative concurrency

Start profiler, run requests at the **middle concurrency level** (representative of real workload), stop profiler:

```bash
# Start profiler
curl -s -X POST http://localhost:8000/start_profile -H "Authorization: Bearer dummy"
echo "Profiler started"

# Run profiling workload at representative concurrency
python3 << 'PYEOF'
import requests, json, time, concurrent.futures

BASE_URL = "http://localhost:8000/v1"
MODEL = "{{MODEL}}"
HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
ISL = {{ISL}}
OSL = {{OSL}}

prompt = "The quick brown fox jumps over the lazy dog. " * (ISL // 10)

# Use the MIDDLE concurrency level for representative profiling
CONC_LEVELS = [{{CONCURRENCY_LEVELS}}]
profile_conc = CONC_LEVELS[len(CONC_LEVELS) // 2] if len(CONC_LEVELS) > 1 else CONC_LEVELS[0]
n_req = max(profile_conc, 4)

print(f"Profiling at concurrency={profile_conc}, {n_req} requests, ISL={ISL}, OSL={OSL}...")

def make_request():
    resp = requests.post(f"{BASE_URL}/completions", headers=HEADERS,
        json={"model": MODEL, "prompt": prompt, "max_tokens": OSL, "temperature": 0},
        timeout=600)
    return resp.status_code

with concurrent.futures.ThreadPoolExecutor(max_workers=profile_conc) as ex:
    results = [f.result() for f in [ex.submit(make_request) for _ in range(n_req)]]
oks = sum(1 for r in results if r == 200)
print(f"  {oks}/{len(results)} requests succeeded")
PYEOF

# Stop profiler
sleep 2
curl -s -X POST http://localhost:8000/stop_profile -H "Authorization: Bearer dummy"
echo "Profiler stopped"

# Wait for trace flush
sleep 10
echo "Trace files:"
ls -lh {{PROFILE_DIR}}/*.json* 2>/dev/null
```

### 4. Validate trace
```bash
python3 << 'PYEOF'
import gzip, json, glob, os

traces = sorted(glob.glob("{{PROFILE_DIR}}/*.json*"), key=lambda f: -os.path.getsize(f))
for t in traces[:2]:
    size = os.path.getsize(t) / 1e6
    print(f"  {os.path.basename(t)}: {size:.1f}MB")
    if size < 0.1:
        print("    WARNING: trace too small")

if not traces:
    print("ERROR: No trace files found")
PYEOF
```

## Completion
Update progress.json:
```json
{
  "phase": "benchmark-and-profile",
  "phases_completed": ["env", "vllm-setup", "benchmark-and-profile"],
  "current_step": "benchmark and profiling complete",
  "details": {
    "benchmark_concurrencies": [{{CONCURRENCY_LEVELS}}],
    "profile_concurrency": "<middle_conc>",
    "isl": {{ISL}},
    "osl": {{OSL}}
  }
}
```
