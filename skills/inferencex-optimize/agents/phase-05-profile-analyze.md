# Phase 5: Profile Analysis

## Instructions

You are a phase agent responsible for analyzing profiling traces to identify GPU kernel bottlenecks. You read exactly 2 files: this document and your handoff at `handoff/to-phase-05.md`.

**Tools**: Shell commands, Python, file I/O.
**Outputs**: Write `agent-results/phase-05-result.md`. Write `profile_analysis.json`, gap analysis data, profiling report.
**Sub-agents**: May spawn an analyzer subagent for trace data analysis per `protocols/analyzer-manifest.schema.md`.
**Errors**: If no traces exist, skip trace analysis and report based on benchmark data only.

## Runbook

### 0. Clean Stale Results
```bash
rm -rf "{{OUTPUT_DIR}}/results/gap_analysis" "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
       "{{OUTPUT_DIR}}/results/tracelens_collective_csvs" "{{OUTPUT_DIR}}/results/phase_split" \
       "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs" "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
rm -f  "{{OUTPUT_DIR}}/results/profile_analysis.json" "{{OUTPUT_DIR}}/results/gpu_arch.json"
```

### 1. Discover and Validate Traces
```bash
python3 "{{SCRIPTS_DIR}}/profiling/validate_traces.py" --trace-dir "{{PROFILE_DIR}}"
```
Capture outputs: `TRACE_COUNT`, `WORLD_SIZE`, `TRACELENS_PRIMARY_TRACE`, etc. If `TRACE_COUNT=0`, skip to step 4.

### 2. Gap Analysis
```bash
python3 "{{SCRIPTS_DIR}}/profiling/trace_analyzer.py" "{{PROFILE_DIR}}" \
    --gap-analysis --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" \
    --start-pct 0 --end-pct 100 --top-k 20
```
Timeout: 600s minimum.

### 3. TraceLens Analysis
**3a. Install:**
```bash
bash "{{SCRIPTS_DIR}}/profiling/install_tracelens.sh" "{{RESOURCES_DIR}}"
```

The install script receives `{{RESOURCES_DIR}}` so TraceLens wheels or archives bundled with the skill can be found for offline or pinned installs. Timeout: 300s minimum. If, after retry, the script prints `TRACELENS_INSTALL_FAILED=true`, treat TraceLens as unavailable: skip steps 3b–3e and continue with step 4 using gap analysis and any benchmark context only.

**3b. Detect GPU arch:**
```bash
python3 "{{SCRIPTS_DIR}}/env/detect_gpu_arch.py" --output "{{OUTPUT_DIR}}/results/gpu_arch.json"
```

**3c. Run TraceLens:**
```bash
bash "{{SCRIPTS_DIR}}/profiling/run_tracelens.sh" \
    --primary-trace "$TRACELENS_PRIMARY_TRACE" \
    --primary-role "$TRACELENS_PRIMARY_ROLE" \
    --output-dir "{{OUTPUT_DIR}}" --profile-dir "{{PROFILE_DIR}}" \
    --world-size "$WORLD_SIZE" --collective-mode "$COLLECTIVE_TRACE_MODE" \
    --phase-split-trace "$PHASE_SPLIT_INPUT_TRACE" \
    --phase-split-role "$PHASE_SPLIT_INPUT_ROLE" \
    --gpu-arch-json "{{OUTPUT_DIR}}/results/gpu_arch.json"
```

Substitute shell variables from step 1 output. Full `run_tracelens.sh` flag set: `--primary-trace` (primary roofline/trace input path), `--primary-role` (role string for that trace), `--output-dir` (workspace/results root), `--profile-dir` (directory containing trace files), `--world-size` (number of unique rank IDs), `--collective-mode` (how collective traces are interpreted), `--phase-split-trace` and `--phase-split-role` (merged or full trace used for prefill vs decode split), `--gpu-arch-json` (hardware roofline inputs from step 3b). Together this runs the primary single-trace report, optional multi-rank collective analysis when `WORLD_SIZE > 1`, and phase-split roofline when inputs allow.

**3d. Display results:**
```bash
bash "{{SCRIPTS_DIR}}/profiling/display_tracelens_results.sh" "{{OUTPUT_DIR}}" "$TRACELENS_PRIMARY_ROLE"
```

Run after TraceLens finishes to print readable summaries and paths for generated artifacts (helps confirm outputs before parsing).

**3e. Parse CSVs** — Read CSVs under `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` and, when present, `{{OUTPUT_DIR}}/results/tracelens_collective_csvs/`:

| File | Purpose |
|------|---------|
| `gpu_timeline.csv` | GPU activity timeline |
| `ops_summary.csv` | Operator-level time breakdown |
| `ops_summary_by_category.csv` | Time grouped by op category |
| `kernel_summary.csv` | GPU kernel execution statistics |
| `coll_analysis.csv` | Collective communication analysis |

Also read phase-specific roofline CSVs from `{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs/` and `{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs/`:

| File | Purpose |
|------|---------|
| `unified_perf_summary.csv` | Per-op roofline (FLOPS/byte, TFLOPS/s, bound type) |
| `GEMM.csv` | Per-GEMM-shape roofline (retain **all** shapes for reporting and downstream shape extraction) |
| `SDPA_fwd.csv` / `FLASH_ATTN_fwd.csv` | Attention roofline per phase |

Persist merged structured results to `{{OUTPUT_DIR}}/results/profile_analysis.json` following `{{TEMPLATES_DIR}}/profile_analysis_schema.json`.

### 4. Identify Bottlenecks
Merge gap analysis, TraceLens, and phase-split roofline findings into `{{OUTPUT_DIR}}/results/profile_analysis.json`.

**Merge strategy:**
- **Gap analysis (step 2):** Top-K most expensive steady-state kernels from `gap_analysis.csv`; kernel time distribution across ranks (load imbalance); whether the workload is compute-bound (GEMM-heavy) vs communication-bound (collective-heavy).
- **TraceLens (step 3, or gap-only if install failed):** Kernels by cumulative time from `kernel_summary.csv`; collective overhead from `coll_analysis.csv`; GPU idle gaps / pipeline bubbles from `gpu_timeline.csv`; time distribution from `ops_summary_by_category.csv`. Preserve `trace_roles` semantics: call it `rank-0` only when the role is `rank0-extend` or `rank0-full`.
- **Phase-split roofline (step 3 when phase CSV dirs exist):** Compare GPU utilization between prefill-decode and decode-only phases; identify bound type per phase (prefill often compute-bound, decode often memory-bound); flag operators far from the roofline ceiling as optimization opportunities.

### 5. Generate Report
Create `{{REPORT_DIR}}/profiling_report.md` using `{{TEMPLATES_DIR}}/profiling_report.md`.

### Completion
Write `agent-results/phase-05-result.md` with gap_analysis status, tracelens status, GPU arch, report path, and top kernel findings (for sticky: `top_kernels`, `kernel_time_pct`).

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
