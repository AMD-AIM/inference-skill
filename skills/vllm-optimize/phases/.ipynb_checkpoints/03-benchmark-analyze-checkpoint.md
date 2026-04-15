# Phase 3: Benchmark Analysis {{SKIP_LABEL}}

## Objective
Analyze benchmark results to identify performance characteristics, scaling behavior, and bottlenecks.

## Steps

### 1. Collect All Benchmark Results
Gather result files from the output directory: `{{OUTPUT_DIR}}/results/`.
Look for JSON result files matching the benchmark naming pattern (e.g., `benchmark_report.json`).

### 2. Parse Benchmark Metrics
For each result file, extract key metrics:
- **Request Throughput** (requests/second)
- **Input Token Throughput** (tokens/second)
- **Output Token Throughput** (tokens/second)
- **Total Token Throughput** (tokens/second)
- **Time to First Token (TTFT)**
- **Inter-Token Latency (ITL)**
- **Total latency**
- **Request success rate**

### 3. Compute Derived Metrics
For each benchmark result, compute the following derived values and present them in **exactly ONE table** (do **NOT** split into two separate tables):
- `Token Thpt/GPU = total_token_throughput / tp` (total throughput per GPU)
- `In Thpt/GPU = input_token_throughput / tp`
- `Out Thpt/GPU = output_token_throughput / tp`
- `Interactivity (tok/s/user) = 1 / TPOT` where `TPOT = mean_itl_ms / 1000` (Time Per Output Token in seconds)
- `End-to-end Latency (s) = mean_e2el_ms / 1000`

**IMPORTANT**: Print ALL derived metrics in a single table with this exact format:

```
Throughput per GPU, Interactivity & Latency:
--------------------------------------------------------------------------------
Conc | ISLxOSL   | TP | Token Thpt/GPU | In Thpt/GPU | Out Thpt/GPU | Interactivity(tok/s/user) | End-to-end Latency (s)
---- | --------- | -- | -------------- | ----------- | ------------ | ------------------------- | ----------------------
...  | ...       | .. | ...            | ...         | ...          | ...                       | ...
```

### 4. Build Comparison Table
Create a table comparing performance across:
- Different concurrency levels
- Different sequence lengths (ISL×OSL)

Save as `{{OUTPUT_DIR}}/results/benchmark_summary.json`.

### 5. Identify Benchmark Bottlenecks
Analyze benchmark metrics to identify performance bottlenecks:
- Identify scaling bottlenecks (how throughput changes with concurrency)
- Detect throughput saturation points across concurrency levels
- Analyze TTFT and ITL degradation under load
- Note memory pressure points
- Flag any anomalous results (e.g., throughput drops, high error rates)

Save to `{{OUTPUT_DIR}}/results/bottlenecks.json`.

### 6. Generate Benchmark Report
Create `{{REPORT_DIR}}/benchmark_report.md` with the configuration, throughput summary, latency summary, and scaling analysis.

## Completion
Update progress.json:
```json
{
  "phase": "benchmark-analyze",
  "phases_completed": ["env", "vllm-setup", "benchmark", "benchmark-analyze"],
  "current_step": "benchmark analysis and report complete",
  "details": {
    "results_analyzed": "<N>",
    "bottlenecks_found": "<M>",
    "final_report": "{{REPORT_DIR}}/benchmark_report.md"
  }
}
```