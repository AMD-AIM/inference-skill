# Phase 2: Benchmark Execution

## Objective
Run vLLM benchmark with configurable concurrency sweep and measure performance metrics.

## CRITICAL: Model Name Must Match User Input
The benchmark MUST use the exact model name validated in Phase 1. Never hardcode or guess.

## Steps

### 1. Run Benchmark

Run: `python3 scripts/vllm_benchmark.py --model "$MODEL" --isl ${ISL:-1024} --osl ${OSL:-1024} --concurrency "${CONCURRENCY_LEVELS:-4,8,16,32,64,128}" --output "$OUTPUT_DIR/benchmark_report.json" --base-url "http://localhost:8000/v1"`

The script runs concurrent requests at each level and measures: RPS, input/output/total TPS, and latency percentiles (P50/P90/P99).

### 2. Output Format

Results saved as JSON with structure:
```json
{
  "meta": {"model": "...", "input_tokens": 1024, "output_tokens": 1024, "framework": "vLLM"},
  "results": [
    {"conc": 4, "success": 4, "fail": 0, "rps": 0.419, "input_tps": 4.19, "output_tps": 333.84, "total_tps": 338.03, "lat_avg": 9.55, "lat_p50": 9.55, "lat_p90": 9.55, "lat_p99": 9.55}
  ]
}
```

## Required Output Fields
| Field | Description |
|-------|-------------|
| conc | Concurrency level |
| success / fail | Request counts |
| rps | Requests per second |
| input_tps / output_tps / total_tps | Token throughput |
| lat_avg / lat_p50 / lat_p90 / lat_p99 | Latency (seconds) |

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | Model name |
| ISL | 1024 | Input sequence length |
| OSL | 1024 | Output sequence length |

## Completion
Results saved to `$OUTPUT_DIR/benchmark_report.json`. Next: Phase 3 (Profiling).
