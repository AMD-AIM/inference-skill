# Phase 3: Benchmark Analysis

## Instructions

You are a phase agent responsible for analyzing benchmark results. You read exactly 2 files: this document and your handoff at `handoff/to-phase-03.md`.

**Tools**: Python, file I/O.
**Outputs**: Write `agent-results/phase-03-result.md`. Write `{{OUTPUT_DIR}}/results/benchmark_summary.json`, `{{OUTPUT_DIR}}/results/bottlenecks.json`, `{{REPORT_DIR}}/benchmark_report.md`, and `{{REPORT_DIR}}/report_summary.json`.
**Sub-agents**: May spawn an analyzer subagent for complex data analysis per `protocols/analyzer-manifest.schema.md`.

If result files use mixed formats (some JSON, some CSV) across older runs, normalize in a small parser script under the allowed tooling rather than hand-editing summaries, and record which source file produced each row inside `benchmark_summary.json` for traceability.

## Runbook

### 1. Collect Results
Gather result files from `{{OUTPUT_DIR}}/results/` only. Do NOT search `{{REPO_DIR}}/`.

Ignore transient editor files or unrelated logs; include only benchmark outputs your phase-02 runs produced (often CSV/JSON/text with a consistent prefix or directory layout described in the handoff).

### 2. Parse Metrics
For each result file, extract at least:

- **Request Throughput**
- **Input Token Throughput**, **Output Token Throughput**, **Total Token Throughput**
- **TTFT** (time to first token)
- **ITL** (inter-token latency)
- **Total latency** (end-to-end / request latency as reported by the benchmark)
- **Request success rate**

Map these from the benchmark output fields your artifacts use (CSV/JSON/logs) consistently across runs. When building tables for the report, prefer column headers that match operator language from the runbooks, e.g. **Request Throughput**, **Input Token Throughput**, **Output Token Throughput**, **Total Token Throughput**, **TTFT**, **ITL**, **Total latency**, **Request success rate**.

### 3. Compute Derived Metrics
Present raw + derived values in **one table**. Use these **field names** in `benchmark_summary.json` (adjust only if the source schema uses different keys, but keep the formulas):

- `token_thpt_per_gpu` = `total_token_throughput / tp`
- `in_thpt_per_gpu` = `input_token_throughput / tp`
- `out_thpt_per_gpu` = `output_token_throughput / tp`
- `interactivity_tok_s_per_user` = `1 / TPOT` where `TPOT = mean_itl_ms / 1000` (human label: **Interactivity (tok/s/user)**)
- `e2e_latency_s` = `mean_e2el_ms / 1000` (human label: **End-to-end Latency (s)**)

**Per-GPU throughput columns** in tables should read as **Token Thpt/GPU**, **In Thpt/GPU**, and **Out Thpt/GPU**, matching the numerators above divided by `tp`.

Include these derived fields in `{{OUTPUT_DIR}}/results/benchmark_summary.json` alongside the raw metrics.

### 4. Build Comparison Table
Compare results **across concurrency levels**, **sequence lengths (ISL×OSL)**, and **framework / precision** combinations. Highlight where throughput scales, plateaus, or regresses.

Structure the comparison so a reader can answer: (a) best token/s at each concurrency, (b) whether TTFT/ITL blow up as concurrency or OSL increases, and (c) how FP8/FP4 (or other precisions) trade accuracy of throughput vs latency in the same table. Save the consolidated view as part of `{{OUTPUT_DIR}}/results/benchmark_summary.json` (for example a `comparisons` array or nested object) or a clearly linked structure within that file so later steps and reports consume **one** primary analysis artifact.

Example (illustrative columns — align to your real metrics):

| Config (framework/precision) | conc | ISL×OSL | Req/s | Tot tok/s | Tok/s/GPU | TTFT (ms) | ITL (ms) | Success % |
|-----------------------------|------|---------|-------|-----------|-----------|-----------|----------|-----------|
| vllm / fp8                  | 8    | 1024×1024 | …   | …         | …         | …         | …        | …         |

Repeat rows for each concurrency and sequence point so scaling curves are obvious at a glance.

### 5. Identify Bottlenecks
Analyze and document:

- **Scaling bottlenecks** (e.g. throughput not growing with concurrency or TP)
- **Throughput saturation** (flat token/s despite higher load)
- **TTFT / ITL degradation** (latency spikes under load or long outputs)
- **Memory pressure** (OOM, KV growth, or batch collapse)
- **Anomalous results** (missing runs, zero throughput, success rate under 100%, outliers vs neighboring points)

Save findings to `{{OUTPUT_DIR}}/results/bottlenecks.json`.

### 6. Generate Benchmark Report
Create `{{REPORT_DIR}}/benchmark_report.md` using the template from `{{TEMPLATES_DIR}}/benchmark_report.md`. Overwrite any existing report.

The Markdown report should embed or reference the comparison table from step 4 and summarize bottleneck themes from `bottlenecks.json` in prose (what changed across concurrency / sequence / precision), not only raw numbers.

### 7. Generate JSON Summary
Create `{{REPORT_DIR}}/report_summary.json` with machine-readable headline metrics, key comparisons, and references to paths of `benchmark_summary.json` / `bottlenecks.json` as appropriate for downstream tooling.

Suggested contents (adapt keys to your pipeline): top-level `config_key`, `results_analyzed`, `primary_bottleneck_ids`, `best_throughput_point` (with `conc`, `isl`, `osl`, `framework`, `precision`), and `artifact_paths` pointing at the JSON/Markdown you produced in this phase.

### Completion
Write `agent-results/phase-03-result.md` with results_analyzed count, bottlenecks_found count, report path, and `report_summary.json` path.

Sanity-check JSON outputs before handoff: `benchmark_summary.json` includes both raw throughput/latency numbers **and** the derived fields from step 3; `bottlenecks.json` entries reference specific configs (concurrency / sequence / framework) they apply to; `report_summary.json` is valid JSON and its paths exist on disk.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
