# Phase 3: Benchmark Analysis {{SKIP_LABEL}}

## Objective
Analyze benchmark results to identify performance characteristics, scaling behavior, and bottlenecks.

## Steps

### 1. Collect All Benchmark Results
Gather result files **only** from the output directory: `{{OUTPUT_DIR}}/results/`.
Do **NOT** search `{{REPO_DIR}}/` — it contains pre-existing results from other experiments that are not part of this run.
Look for JSON result files matching the experiment naming pattern (e.g., `*_conc*.json`).

### 2. Parse Benchmark Metrics
For each result file, extract key metrics:
- **Request Throughput** (requests/second)
- **Input Token Throughput** (tokens/second) — if not in raw data, compute as `total_token_throughput - output_token_throughput` or `total_input_tokens / duration_s`
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

**IMPORTANT**: Print ALL derived metrics in a single table with this exact format. Do NOT create separate "Throughput vs Interactivity" and "Throughput vs Latency" tables.

```
Throughput per GPU, Interactivity & Latency:
--------------------------------------------------------------------------------
Conc | ISLxOSL   | TP | Token Thpt/GPU | In Thpt/GPU | Out Thpt/GPU | Interactivity(tok/s/user) | End-to-end Latency (s)
---- | --------- | -- | -------------- | ----------- | ------------ | ------------------------- | ----------------------
...  | ...       | .. | ...            | ...         | ...          | ...                       | ...
```

Include `tok_per_s_per_gpu`, `in_tok_per_s_per_gpu`, `out_tok_per_s_per_gpu`, `interactivity_tok_per_s`, and `e2el_s` as derived fields in the benchmark summary JSON alongside the raw metrics.

### 4. Build Comparison Table
Create a table comparing performance across:
- Different concurrency levels
- Different sequence lengths (ISL×OSL)
- Framework/precision combinations

Save as `{{OUTPUT_DIR}}/results/benchmark_summary.json`.

### 5. Identify Benchmark Bottlenecks
Analyze benchmark metrics to identify performance bottlenecks:
- Identify scaling bottlenecks (how throughput changes with concurrency)
- Detect throughput saturation points across concurrency levels
- Analyze TTFT and ITL degradation under load
- Compare throughput/latency trade-offs across TP configurations and sequence lengths
- Note memory pressure points
- Flag any anomalous results (e.g., throughput drops, high error rates)

Save to `{{OUTPUT_DIR}}/results/bottlenecks.json`.

### 6. Generate Benchmark Report
Create `{{REPORT_DIR}}/benchmark_report.md` using the **exact template below**. If an old report already exists, **overwrite it completely** — do NOT copy or replicate the old report's structure.

```markdown
# InferenceX Benchmark Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU>
- **Framework**: <framework from config>
- **Model**: <model name>
- **Precision**: <precision>
- **Docker Image**: <image from sweep config>

## Benchmark Results

### Throughput Summary
| Metric                  | Value    |
|-------------------------|----------|
| Request Throughput      | <req/s>  |
| Input Token Throughput  | <tok/s>  |
| Output Token Throughput | <tok/s>  |
| Total Token Throughput  | <tok/s>  |

Note: If the raw benchmark data does not include `input_throughput`, compute it as `total_token_throughput - output_token_throughput`.

### Latency Summary

| Concurrency | ISL×OSL | TTFT Mean (ms) | TTFT P99 (ms) | ITL Mean (ms) | ITL P99 (ms) | End-to-end Latency (s) |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

### Throughput per GPU, Interactivity & Latency

**IMPORTANT**: This MUST be exactly ONE table. Do NOT split into separate "Throughput vs Interactivity" and "Throughput vs Latency" tables.

Shows throughput efficiency per GPU, per-user interactivity, and end-to-end latency across concurrency levels.
- `Token Throughput per GPU = total_token_throughput / tp` (total throughput per GPU)
- `Input Token Throughput per GPU = input_token_throughput / tp`
- `Output Token Throughput per GPU = output_token_throughput / tp`
- `Interactivity (tok/s/user) = 1 / TPOT` where `TPOT = mean_itl_ms / 1000` (Time Per Output Token in seconds)
- `End-to-end Latency (s) = mean_e2el_ms / 1000`

Conc | ISLxOSL   | TP | Token Thpt/GPU | In Thpt/GPU | Out Thpt/GPU | Interactivity(tok/s/user) | End-to-end Latency (s)
---- | --------- | -- | -------------- | ----------- | ------------ | ------------------------- | ----------------------
...  | ...       | .. | ...            | ...         | ...          | ...                       | ...

### Scaling Analysis
- How throughput scales with concurrency
- Optimal concurrency point
- Memory utilization patterns


## Raw Data
- Location of result files
```

### 7. Generate JSON Summary
Create `{{REPORT_DIR}}/report_summary.json` with machine-readable results.

## Completion
Update progress.json:
```json
{
  "phase": "benchmark-analyze",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze"],
  "current_step": "benchmark analysis and report complete",
  "details": {
    "results_analyzed": <N>,
    "bottlenecks_found": <M>,
    "final_report": "{{REPORT_DIR}}/benchmark_report.md"
  }
}
```
