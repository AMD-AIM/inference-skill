# Phase 3: Profiling

## Objective
Generate torch profiler traces for GPU kernel analysis.

## Steps

### 1. Start Profiler

```bash
curl -s -X POST http://localhost:8000/start_profile \
    -H "Authorization: Bearer dummy" \
    -H "Content-Type: application/json"
```

### 2. Run Inference Requests

```python
import requests

HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}

for i in range(3):
    resp = requests.post(
        "http://localhost:8000/v1/completions",
        headers=HEADERS,
        json={
            "model": os.environ.get('MODEL'),
            "prompt": "Hello " * 128,
            "max_tokens": 512,
            "temperature": 0
        },
        timeout=180
    )
    print(f"Request {i+1}: {resp.status_code}")
```

### 3. Stop Profiler

```bash
curl -s -X POST http://localhost:8000/stop_profile \
    -H "Authorization: Bearer dummy" \
    -H "Content-Type: application/json"
```

### 4. Wait for Trace Flush

```bash
sleep 5
ls -la ${PROFILE_DIR:-./vllm_results/profiles}/*.json.gz
```

### 5. Validate Trace

```python
import gzip
import json

trace_file = os.environ.get('PROFILE_DIR', './vllm_results/profiles') + '/'
trace_file += [f for f in os.listdir(trace_file) if f.endswith('.json.gz')][0]

with gzip.open(trace_file, 'rt') as f:
    data = json.load(f)

events = data.get('traceEvents', [])
print(f"Events: {len(events)}")
print(f"Keys: {list(data.keys())[:5]}")
```

Valid trace should have:
- `traceEvents`: list of ~10M+ events
- `deviceProperties`: GPU info
- `traceName`: profiler configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | Model name |
| PROFILE_DIR | ./vllm_results/profiles | Profiler output |
| PROFILE_ITERATIONS | 128 | Number of iterations |

## Completion

Trace file generated in `PROFILE_DIR/`.

Next: Proceed to Phase 4 (Analysis)