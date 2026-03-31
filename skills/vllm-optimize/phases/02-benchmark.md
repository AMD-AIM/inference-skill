# Phase 2: Benchmark Execution

## Objective
Run vLLM benchmark with configurable concurrency sweep and measure performance metrics.

## Steps

### 1. Prepare Test Prompts

Generate prompts of appropriate length:
```python
def generate_prompt(target_length=1024):
    """Generate prompt with approximately target_length tokens"""
    word = "Hello"
    # Average token is ~0.75 words for English
    words_needed = int(target_length * 0.75)
    return " ".join([word] * (words_needed // len(word)))
```

### 2. Run Benchmark

```python
import requests
import time
import concurrent.futures
import json
import os

BASE_URL = "http://localhost:8000/v1"
HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}

ISL = int(os.environ.get('ISL', 1024))
OSL = int(os.environ.get('OSL', 1024))
CONCURRENCY_LEVELS = [4, 8, 16, 32, 64, 128]

def make_request():
    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/completions",
            headers=HEADERS,
            json={
                "model": os.environ.get('MODEL'),
                "prompt": "Hello " * (ISL // 5),
                "max_tokens": OSL,
                "temperature": 0
            },
            timeout=300
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            output = data.get("choices", [{}])[0].get("text", "")
            return {"success": True, "time": elapsed, 
                    "prompt_tokens": ISL, "output_tokens": len(output.split())}
        return {"success": False, "error": resp.status_code}
    except Exception as e:
        return {"success": False, "error": str(e)}

results = []

for CONC in CONCURRENCY_LEVELS:
    print(f"Testing concurrency {CONC}...", end=" ", flush=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as executor:
        start = time.time()
        futures = [executor.submit(make_request) for _ in range(CONC)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start
    
    successes = [r for r in responses if r.get("success")]
    if successes:
        rps = len(successes) / total_time
        total_input = sum(r.get("prompt_tokens", 0) for r in successes)
        total_output = sum(r.get("output_tokens", 0) for r in successes)
        input_tps = total_input / total_time
        output_tps = total_output / total_time
        latencies = sorted([r["time"] for r in successes])
        
        results.append({
            "conc": CONC,
            "success": len(successes),
            "rps": round(rps, 3),
            "input_tps": round(input_tps, 2),
            "output_tps": round(output_tps, 2),
            "total_tps": round(input_tps + output_tps, 2),
            "lat_p50": round(latencies[len(latencies)//2], 2),
            "lat_p90": round(latencies[int(len(latencies)*0.9)], 2),
            "lat_p99": round(latencies[int(len(latencies)*0.99)], 2)
        })
        print(f"RPS: {rps:.3f}, TPS: {input_tps+output_tps:.2f}")
    else:
        results.append({"conc": CONC, "success": 0, "rps": 0, "total_tps": 0})
        print("Failed!")

# Save results
output = {
    "meta": {
        "model": os.environ.get('MODEL'),
        "input_tokens": ISL,
        "output_tokens": OSL,
        "framework": "vLLM",
        "precision": os.environ.get('DTYPE', 'half')
    },
    "results": results
}

output_path = os.environ.get('OUTPUT_PATH', './vllm_results/benchmark_report.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")
```

### 3. Output Format

```json
{
  "meta": {
    "model": "Qwen/Qwen3.5-35B-A3B",
    "input_tokens": 1024,
    "output_tokens": 1024,
    "framework": "vLLM",
    "precision": "half"
  },
  "results": [
    {"conc": 4, "rps": 0.070, "total_tps": 142.64, "lat_p99": 57.43},
    {"conc": 8, "rps": 0.122, "total_tps": 250.49, "lat_p99": 65.41},
    ...
  ]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | Model name |
| ISL | 1024 | Input sequence length |
| OSL | 1024 | Output sequence length |
| OUTPUT_PATH | ./vllm_results/benchmark_report.json | Output file |

## Completion

Benchmark complete. Results saved to `OUTPUT_PATH`.

Next: Proceed to Phase 3 (Profiling) if enabled.