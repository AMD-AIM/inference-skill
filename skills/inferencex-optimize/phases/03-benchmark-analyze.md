> **ARCHIVE**: This file is a reference copy of the original phase runbook. The active
> agent docs are in `agents/phase-NN-*.md`. Script paths in this file reference the
> pre-reorganization flat layout (`scripts/*.py`); the actual scripts are now under
> `scripts/{env,container,profiling,optimize,plugin,report}/`.

# Phase 3: Benchmark Analysis {{SKIP_LABEL}}

## Objective
Analyze benchmark results to identify performance characteristics, scaling behavior, and bottlenecks.

## Steps

### 1. Collect All Benchmark Results
Gather result files from `{{OUTPUT_DIR}}/results/` only. Do NOT search `{{REPO_DIR}}/`.

### 2. Parse Benchmark Metrics
For each result file, extract: Request Throughput, Input/Output/Total Token Throughput, TTFT, ITL, Total latency, Request success rate.

### 3. Compute Derived Metrics
Present in ONE table:
- `Token Thpt/GPU = total_token_throughput / tp`
- `In Thpt/GPU = input_token_throughput / tp`
- `Out Thpt/GPU = output_token_throughput / tp`
- `Interactivity (tok/s/user) = 1 / TPOT` where `TPOT = mean_itl_ms / 1000`
- `End-to-end Latency (s) = mean_e2el_ms / 1000`

Include derived fields in `benchmark_summary.json` alongside raw metrics.

### 4. Build Comparison Table
Compare across concurrency levels, sequence lengths, framework/precision combinations. Save as `{{OUTPUT_DIR}}/results/benchmark_summary.json`.

### 5. Identify Benchmark Bottlenecks
Analyze: scaling bottlenecks, throughput saturation, TTFT/ITL degradation, memory pressure, anomalous results. Save to `{{OUTPUT_DIR}}/results/bottlenecks.json`.

### 6. Generate Benchmark Report
Create `{{REPORT_DIR}}/benchmark_report.md` using template from `{{TEMPLATES_DIR}}/benchmark_report.md`. Overwrite any existing report.

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
    "results_analyzed": 1,
    "bottlenecks_found": 0,
    "final_report": "{{REPORT_DIR}}/benchmark_report.md"
  }
}
```
