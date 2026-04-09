# Phase 5: Profile Analysis {{SKIP_LABEL}}

## Objective
Analyze profiling traces to identify GPU kernel-level performance bottlenecks and optimization opportunities.

{{PROFILE_ANALYSIS_NOTE}}

## IMPORTANT: Always Re-run Analysis From Scratch
When this phase is entered (including via `--from-phase profile-analyze`), always delete stale results first:
```bash
rm -rf "{{OUTPUT_DIR}}/results/gap_analysis" "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
       "{{OUTPUT_DIR}}/results/tracelens_collective_csvs" "{{OUTPUT_DIR}}/results/phase_split" \
       "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs" "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
rm -f  "{{OUTPUT_DIR}}/results/profile_analysis.json" "{{OUTPUT_DIR}}/results/tracelens_rank0.log" \
       "{{OUTPUT_DIR}}/results/tracelens_collective.log" "{{OUTPUT_DIR}}/results/gpu_arch.json"
```

Verify that profile trace files exist in `{{PROFILE_DIR}}/`. If none exist, skip to step 4.

## Steps

### 1. Discover and Validate Trace Files

Run: `python3 "{{SCRIPTS_DIR}}/validate_traces.py" --trace-dir "{{PROFILE_DIR}}"`

Outputs key=value pairs: `TRACE_COUNT`, `WORLD_SIZE`, `TRACELENS_PRIMARY_TRACE`, `TRACELENS_PRIMARY_ROLE`, `PHASE_SPLIT_INPUT_TRACE`, `PHASE_SPLIT_INPUT_ROLE`, `COLLECTIVE_TRACE_MODE`, `MERGED_TRACE`, `RANK0_FULL_TRACE`, `RANK0_EXTEND_TRACE`, `RANK0_DECODE_TRACE`.

Capture all outputs — they are used as arguments for later scripts.

If `TRACE_COUNT=0`, skip to step 4 (bottleneck analysis using benchmark data only).

Trace role selection rules:
- For TraceLens primary input: prefer `RANK0_EXTEND_TRACE` > `RANK0_FULL_TRACE` > `MERGED_TRACE` > first valid trace
- For phase splitting: prefer `MERGED_TRACE` > `RANK0_FULL_TRACE`; do NOT split already-phase-specific traces
- For collective analysis: prefer one `TP-N-EXTEND` trace per rank; fall back to one `rank-N` trace per rank
- `WORLD_SIZE` = number of unique per-rank IDs, not number of trace files

### 2. Gap Analysis (Kernel Profiling) — Primary

Run: `python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "{{PROFILE_DIR}}" --gap-analysis --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" --start-pct 0 --end-pct 100 --top-k 20`

Timeout: 600s minimum. This processes large trace files (potentially millions of events).

Produces a ranked list of the most expensive GPU kernels. The pipeline:
1. Filter by category — include only `kernel` and `gpu` events, exclude `gpu_user_annotation`
2. Aggregate per kernel — group by name, sum CUDA time, count calls
3. Merge across ranks — combine stats into single ranking
4. Rank by total duration descending

Typical bottleneck categories: GEMM kernels (`ck_fmha_*`, `hipblas*`), communication (`ncclAllReduce*`), custom attention (`paged_attention_*`), quantization (`dequant*`, `mxfp4_*`).

### 3. TraceLens Analysis

**3a. Install TraceLens:**

Run: `bash "{{SCRIPTS_DIR}}/install_tracelens.sh" "{{RESOURCES_DIR}}"`

Timeout: 300s. If output contains `TRACELENS_INSTALL_FAILED=true` after retry, proceed to step 4 using gap analysis data only.

**3b. Detect GPU architecture (for roofline):**

Run: `python3 "{{SCRIPTS_DIR}}/detect_gpu_arch.py" --output "{{OUTPUT_DIR}}/results/gpu_arch.json"`

**3c. Run TraceLens reports:**

Run:
```bash
bash "{{SCRIPTS_DIR}}/run_tracelens.sh" \
    --primary-trace "$TRACELENS_PRIMARY_TRACE" \
    --primary-role "$TRACELENS_PRIMARY_ROLE" \
    --output-dir "{{OUTPUT_DIR}}" \
    --profile-dir "{{PROFILE_DIR}}" \
    --world-size "$WORLD_SIZE" \
    --collective-mode "$COLLECTIVE_TRACE_MODE" \
    --phase-split-trace "$PHASE_SPLIT_INPUT_TRACE" \
    --phase-split-role "$PHASE_SPLIT_INPUT_ROLE" \
    --gpu-arch-json "{{OUTPUT_DIR}}/results/gpu_arch.json"
```

Substitute variables from step 1 output. This runs: primary single-trace report, multi-rank collective report (if `WORLD_SIZE > 1`), and phase-split roofline analysis.

**3d. Display results:**

Run: `bash "{{SCRIPTS_DIR}}/display_tracelens_results.sh" "{{OUTPUT_DIR}}" "$TRACELENS_PRIMARY_ROLE"`

**3e. Parse TraceLens CSV outputs:**

Read CSVs from `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` and (if present) `tracelens_collective_csvs/`:
- `gpu_timeline.csv` — GPU activity timeline
- `ops_summary.csv` — operator-level time breakdown
- `ops_summary_by_category.csv` — time grouped by op category
- `kernel_summary.csv` — GPU kernel execution statistics
- `coll_analysis.csv` — collective communication analysis

Also read phase-specific CSVs from `tracelens_prefill_decode_csvs/` and `tracelens_decode_only_csvs/`:
- `unified_perf_summary.csv` — per-op roofline (FLOPS/byte, TFLOPS/s, bound type)
- `GEMM.csv` — per-GEMM-shape roofline data (include ALL shapes in report)
- `SDPA_fwd.csv` / `FLASH_ATTN_fwd.csv` — attention roofline per phase

Save to `{{OUTPUT_DIR}}/results/profile_analysis.json` using schema from `{{TEMPLATES_DIR}}/profile_analysis_schema.json`.

### 4. Identify Profile Bottlenecks

From **gap analysis** (step 2):
- Top-K most expensive steady-state kernels from `gap_analysis.csv`
- Kernel time distribution across ranks (load imbalance)
- Whether bottleneck is compute-bound (GEMM-heavy) or communication-bound (collective-heavy)

From **TraceLens** (step 3, fall back to gap analysis only if install failed):
- GPU kernels by cumulative time from `kernel_summary.csv`
- Collective communication overhead from `coll_analysis.csv`
- GPU idle gaps from `gpu_timeline.csv` (pipeline bubbles, CPU-bound phases)
- Time distribution from `ops_summary_by_category.csv`
- Preserve `trace_roles` semantics: only call it `rank-0` if role is `rank0-extend` or `rank0-full`

From **phase-split roofline** (step 3, when available):
- Compare GPU utilization between prefill-decode and decode-only phases
- Identify bound type per phase (prefill typically compute-bound, decode typically memory-bound)
- Flag ops far from roofline ceiling as optimization opportunities

Merge findings into `{{OUTPUT_DIR}}/results/profile_analysis.json`.

### 5. Generate Profiling Report

Generate `{{REPORT_DIR}}/profiling_report.md` using template from `{{TEMPLATES_DIR}}/profiling_report.md`.

Populate from `{{OUTPUT_DIR}}/results/profile_analysis.json` and `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json`. Preserve trace-role semantics from step 1.

Print the final report path:
```bash
echo "Report: {{REPORT_DIR}}/profiling_report.md"
```

## Completion
Update progress.json:
```json
{
  "phase": "profile-analyze",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  "current_step": "profile analysis complete",
  "details": {
    "gap_analysis": true,
    "tracelens_analysis": true,
    "phase_split_roofline": true,
    "gpu_arch_detected": "<GPU model name or null>",
    "report": "{{REPORT_DIR}}/profiling_report.md"
  }
}
```
