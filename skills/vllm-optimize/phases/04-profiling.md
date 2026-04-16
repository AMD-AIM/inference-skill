# Phase 4: Profiling {{SKIP_LABEL}}

## Objective
Generate torch profiler traces for GPU kernel analysis. The profiling workload **MUST match the benchmark configuration** (ISL, OSL, concurrency) to ensure kernel time percentages reflect the real target workload.

## Steps

### 1. Start Profiler

```bash
curl -s -X POST http://localhost:8000/start_profile \
    -H "Authorization: Bearer dummy" \
    -H "Content-Type: application/json"
```

### 2. Run Inference Requests

**CRITICAL**: Use `{{ISL}}` and `{{OSL}}` to generate the profiling workload. DO NOT use hardcoded short prompts — the kernel time distribution depends heavily on sequence length.

```bash
python3 << 'PYEOF'
import requests, os, json

HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
MODEL = "{{MODEL}}"
BASE_URL = "http://localhost:8000/v1"

ISL = {{ISL}}   # Input sequence length from config
OSL = {{OSL}}   # Output sequence length from config

# Generate a prompt of approximately ISL tokens
# "word " is roughly 1 token, so repeat ISL times
prompt = "The quick brown fox jumps over the lazy dog. " * (ISL // 10)

# Send multiple requests to get a representative trace
# At least 2 requests: one for JIT warmup, one for measurement
NUM_REQUESTS = 3

for i in range(NUM_REQUESTS):
    print(f"Profiling request {i+1}/{NUM_REQUESTS} (ISL≈{ISL}, OSL={OSL})...")
    try:
        resp = requests.post(
            f"{BASE_URL}/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": OSL,
                "temperature": 0,
            },
            timeout=600,  # Long timeout for large OSL
        )
        if resp.status_code == 200:
            usage = resp.json().get("usage", {})
            print(f"  OK: prompt_tokens={usage.get('prompt_tokens', '?')}, "
                  f"completion_tokens={usage.get('completion_tokens', '?')}")
        else:
            print(f"  Error: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
PYEOF
```

### 3. Stop Profiler

```bash
curl -s -X POST http://localhost:8000/stop_profile \
    -H "Authorization: Bearer dummy" \
    -H "Content-Type: application/json"
```

### 4. Wait for Trace Flush

```bash
sleep 10
ls -la {{PROFILE_DIR}}/*.json.gz 2>/dev/null || ls -la {{PROFILE_DIR}}/*.json 2>/dev/null
```

### 5. Validate Trace

```python
import gzip, json, os, glob

trace_dir = "{{PROFILE_DIR}}"
traces = sorted(glob.glob(os.path.join(trace_dir, "*.json*")),
                key=lambda f: -os.path.getsize(f))

for trace_file in traces[:2]:
    size_mb = os.path.getsize(trace_file) / 1e6
    print(f"Trace: {os.path.basename(trace_file)} ({size_mb:.1f}MB)")
    
    opener = gzip.open if trace_file.endswith('.gz') else open
    with opener(trace_file, 'rt') as f:
        data = json.load(f)
    
    events = data.get('traceEvents', [])
    gpu_events = [e for e in events if e.get('cat', '') in ('kernel', 'cuda', 'gpu')]
    print(f"  Total events: {len(events)}")
    print(f"  GPU kernel events: {len(gpu_events)}")
    
    if len(events) < 1000:
        print("  WARNING: trace seems too small — profiling may not have captured the inference")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| MODEL | Model name or path |
| PROFILE_DIR | Where trace files are saved |
| ISL | Input sequence length — MUST match benchmark config |
| OSL | Output sequence length — MUST match benchmark config |

## Completion

Trace file(s) generated in `{{PROFILE_DIR}}/`.

Update progress.json:
```json
{
  "phase": "profiling",
  "phases_completed": ["env", "vllm-setup", "benchmark", "benchmark-analyze", "profiling"],
  "current_step": "profiling complete",
  "details": {
    "isl": "{{ISL}}",
    "osl": "{{OSL}}",
    "num_requests": 3,
    "trace_dir": "{{PROFILE_DIR}}"
  }
}
```

Next: Proceed to Phase 5 (Profile Analysis)
