# Phase 9: Final Optimization Report

## Instructions

You are a phase agent responsible for generating the final optimization report. You read exactly 2 files: this document and your handoff at `handoff/to-phase-09.md`.

**Tools**: Python, file I/O.
**Outputs**: Write `agent-results/phase-09-result.md`. Write `optimization_report.md` and `optimization_summary.json`.
**Sub-agents**: May spawn an analyzer subagent per `protocols/analyzer-manifest.schema.md`.

## Runbook

### 1. Gather Artifacts
Read from `{OUTPUT_DIR}`:
- `results/benchmark_summary.json` (Phase 3)
- `results/gap_analysis/gap_analysis.json` (Phase 5)
- `results/profile_analysis.json` (Phase 5)
- `problems/optimization_manifest.json` (Phase 6)
- `problems/kernel_type_classification.json` (Phase 6)
- `problems/fusion_opportunities.json` (Phase 6)
- `problems/geak_results.json` (Phase 7)
- `results/optimization_comparison.json` (Phase 8)
- `report/benchmark_report.md` and `profiling_report.md` (Phases 3, 5)
- `results/gpu_arch.json` and `env_info.json`

Cross-check the same logical artifacts via explicit phase paths when the workspace layout mirrors the run root:
- `{{RESULTS_DIR}}/benchmark_summary.json` (Phase 3 aggregated benchmark table)
- `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` (Phase 5 gap analysis)
- `{{OUTPUT_DIR}}/results/profile_analysis.json` (Phase 5 merged TraceLens + gap view)
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (Phase 6 target list)
- `{{PROBLEMS_DIR}}/kernel_type_classification.json` (Phase 6 typing / comm exclusion context)
- `{{PROBLEMS_DIR}}/fusion_opportunities.json` (Phase 6 fusion suggestions)
- `{{PROBLEMS_DIR}}/geak_results.json` (Phase 7 per-kernel speedups + metadata)
- `{{RESULTS_DIR}}/optimization_comparison.json` (Phase 8 measured E2E validation)
- `{{REPORT_DIR}}/benchmark_report.md` plus `{{REPORT_DIR}}/profiling_report.md` (narratives from Phases 3 and 5)
- `{{OUTPUT_DIR}}/results/gpu_arch.json` (roofline hardware context) and `{{ENV_INFO_FILE}}` (Phase 0 environment capture)

Optional but valuable when referenced in earlier phases: raw TraceLens CSV snapshots under `{{OUTPUT_DIR}}/results/tracelens_*_csvs/` (attach pointers, not necessarily the full CSV dump, unless the template calls for detailed tables).

**Gathering checklist:** (1) confirm Phase 8 validation bit is true, (2) confirm `geak_results.json` lists every kernel referenced in prose, (3) confirm `benchmark_summary.json` throughput matches the tables you cite, (4) attach GPU arch + env info for reproducibility.

### 2. Generate Report
Create `{{REPORT_DIR}}/optimization_report.md` using the canonical template at `{{TEMPLATES_DIR}}/optimization_report.md` (copy structure/headings verbatim, replace placeholders with measured values). Populate all sections with measured data from artifacts—do not invent numbers.

Before writing prose, verify high-signal JSON parses cleanly (`python3 -m json.tool <file>`). If `optimization_comparison.json` is missing or `validated` is `false`, the report must state that explicitly—never imply a passing E2E result.

### What the report must cover
- **Kernel optimization results:** Table or bullet summary of each targeted kernel, GEAK mode, baseline vs optimized timing, `speedup`, and whether patch recovery was required.
- **End-to-end comparison:** Throughput / latency deltas from `optimization_comparison.json`, noting concurrency, sequence lengths, and framework version parity between baseline and optimized runs.
- **Recommendations:** Prioritized next steps (remaining high-percent kernels, dispatch-level follow-ups, communication tuning, profiling re-runs) grounded in data from gap analysis, TraceLens, and manifest entries.

Call out explicit **risks and caveats** (e.g., kernels skipped due to roofline headroom, validation failures, partial plugin coverage) so readers can trust which portions of the pipeline finished.

When the Markdown template includes appendices for trace analysis, link back to the exact sections in `profiling_report.md` (e.g., collective imbalance chapter, roofline tables) instead of duplicating multi-page CSV dumps inline.

### 3. Generate Machine-Readable Summary
Run the summary generator **after** `optimization_report.md` is written so any last-minute manual edits to metrics are reflected in your review, but **before** declaring the workflow complete—the JSON is the machine-facing contract for downstream automation.

```bash
python3 "{{SCRIPTS_DIR}}/report/generate_optimization_summary.py" \
    --output "{{REPORT_DIR}}/optimization_summary.json" \
    --config-key "{{CONFIG_KEY}}" --framework "{{FRAMEWORK}}" \
    --env-info "{{ENV_INFO_FILE}}" \
    --results-dir "{{RESULTS_DIR}}" --problems-dir "{{PROBLEMS_DIR}}"
```

`generate_optimization_summary.py` flags (pass explicitly even when optional defaults exist):
- `--output` *(required)* — destination JSON path (`optimization_summary.json`)
- `--config-key` *(required)* — `{{CONFIG_KEY}}` for traceability
- `--framework` *(required)* — `{{FRAMEWORK}}` string recorded in the summary
- `--env-info` — path to `{{ENV_INFO_FILE}}`; when present, enriches GPU / GEAK availability fields
- `--results-dir` — directory containing `optimization_comparison.json` plus other Phase 8 JSON artifacts
- `--problems-dir` — directory containing `geak_results.json`, `optimization_manifest.json`, etc.

The emitted `optimization_summary.json` aggregates high-level fields consumed by dashboards: `config_key`, `framework`, optional `gpu_arch` / `geak_mode` (when `env-info` parses), throughput + `speedup` + `validated` from `optimization_comparison.json`, per-kernel stats (`kernels_attempted`, `kernels_improved`, `patches_recovered`, embedded `kernel_results`), and manifest sizing (`total_problem_files`). Extend the Python emitter only when new machine fields are required—keep the Markdown report as the human narrative.

### 4. Print Artifact Paths
```bash
echo "Report: {{REPORT_DIR}}/optimization_report.md"
echo "Summary: {{REPORT_DIR}}/optimization_summary.json"
```

Archive both files (and the intermediate JSON inputs) in the same artifact bundle your organization uses for audit—future regressions should diff against this exact tarball, not regenerated estimates.

If your team tracks git commits per optimization attempt, record the `CONFIG_KEY`, container image digest, and benchmark script SHA256 in `agent-results/phase-09-result.md` alongside the printed paths so the report maps to an exact reproducible stack.

### Completion
Write `agent-results/phase-09-result.md` with report path and summary path.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
