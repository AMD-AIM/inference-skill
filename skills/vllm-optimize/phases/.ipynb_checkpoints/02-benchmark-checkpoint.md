# Phase 2: Benchmark Execution

## Objective
Run vLLM benchmark with configurable concurrency sweep and measure performance metrics.

## CRITICAL: Model Name Must Match User Input

**⚠️ IMPORTANT**: The benchmark script MUST use the exact model name that was validated in Phase 1. Do NOT hardcode or guess model names. The model name should be passed via environment variable `MODEL` which is set from user input.

```bash
# The model name MUST come from user input, not be guessed
MODEL="THUDM/glm-4-9b-chat"  # Example: exact HuggingFace model ID from user
```

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

Run benchmark with full metrics including input/output TPS, latency percentiles, and success/fail counts.

```python
import requests
import time
import concurrent.futures
import json
import os
import statistics

BASE_URL = "http://localhost:8000/v1"
HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}

ISL = int(os.environ.get('ISL', 1024))
OSL = int(os.environ.get('OSL', 1024))
CONCURRENCY_LEVELS = [4, 8, 16, 32, 64, 128]

def make_request(model_name):
    start = time.time()
    try:
        # Use chat completions API
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Write a short story."}],
                "max_tokens": OSL,
                "temperature": 0
            },
            timeout=300
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "success": True, 
                "time": elapsed, 
                "prompt_tokens": usage.get("prompt_tokens", ISL),
                "output_tokens": usage.get("completion_tokens", len(output.split()) if output else 0)
            }
        return {"success": False, "error": resp.status_code, "time": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}

results = []

for CONC in CONCURRENCY_LEVELS:
    print(f"\n=== Testing concurrency {CONC} ===")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as executor:
        start = time.time()
        futures = [executor.submit(make_request, os.environ.get('MODEL', 'default')) for _ in range(CONC)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start
    
    successes = [r for r in responses if r.get("success")]
    failures = [r for r in responses if not r.get("success")]
    
    if successes:
        rps = len(successes) / total_time
        total_input = sum(r.get("prompt_tokens", 0) for r in successes)
        total_output = sum(r.get("output_tokens", 0) for r in successes)
        input_tps = total_input / total_time
        output_tps = total_output / total_time
        latencies = sorted([r["time"] for r in successes])
        
        # Calculate percentiles
        n = len(latencies)
        p50 = latencies[int(n * 0.50)] if n > 0 else 0
        p90 = latencies[int(n * 0.90)] if n > 0 else 0
        p99 = latencies[int(n * 0.99)] if n > 0 else 0
        avg_lat = sum(latencies) / n if n > 0 else 0
        
        result = {
            "conc": CONC,
            "success": len(successes),
            "fail": len(failures),
            "rps": round(rps, 3),
            "input_tps": round(input_tps, 2),
            "output_tps": round(output_tps, 2),
            "total_tps": round(input_tps + output_tps, 2),
            "lat_avg": round(avg_lat, 2),
            "lat_p50": round(p50, 2),
            "lat_p90": round(p90, 2),
            "lat_p99": round(p99, 2)
        }
        results.append(result)
        
        print(f"  RPS: {rps:.3f}")
        print(f"  Input TPS: {input_tps:.2f}")
        print(f"  Output TPS: {output_tps:.2f}")
        print(f"  Total TPS: {input_tps + output_tps:.2f}")
        print(f"  Latency - Avg: {avg_lat:.2f}s, P50: {p50:.2f}s, P90: {p90:.2f}s, P99: {p99:.2f}s")
    else:
        results.append({"conc": CONC, "success": 0, "fail": CONC, "rps": 0, "total_tps": 0})
        print("  FAILED!")

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
    "model": "${MODEL}",  <!-- Must be exact HuggingFace model ID from user input -->
    "model_name": "${MODEL_DISPLAY_NAME}",  <!-- Optional: friendly name from user -->
    "input_tokens": 1024,
    "output_tokens": 1024,
    "framework": "vLLM",
    "precision": "fp16",
    "gpu": "AMD MI300X",
    "tp": 1,
    "backend": "ROCM_ATTN"
  },
  "results": [
    {
      "conc": 4,
      "success": 4,
      "fail": 0,
      "rps": 0.419,
      "input_tps": 4.19,
      "output_tps": 333.84,
      "total_tps": 338.03,
      "lat_avg": 9.55,
      "lat_p50": 9.55,
      "lat_p90": 9.55,
      "lat_p99": 9.55
    },
    ...
  ]
}
```

## Required Output Fields

| Field | Description |
|-------|-------------|
| conc | Concurrency level |
| success | Number of successful requests |
| fail | Number of failed requests |
| rps | Requests per second |
| input_tps | Input tokens per second |
| output_tps | Output tokens per second |
| total_tps | Total tokens per second (input + output) |
| lat_avg | Average latency (seconds) |
| lat_p50 | P50 latency (seconds) |
| lat_p90 | P90 latency (seconds) |
| lat_p99 | P99 latency (seconds) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | Model name |
| ISL | 1024 | Input sequence length |
| OSL | 1024 | Output sequence length |
| OUTPUT_PATH | ./vllm_results/benchmark_report.json | Output file |
| DTYPE | half | Model precision (half/fp16, bf16, fp8) |

## Completion

Benchmark complete. Results saved to `OUTPUT_PATH`.

Next: Proceed to Phase 3 (Profiling) if enabled.