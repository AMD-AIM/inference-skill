# Phase 9: Final Optimization Report

## Instructions

You are a phase agent responsible for generating the final optimization report. You read exactly 2 files: this document and your handoff at `handoff/to-phase-09.md`.

**Tools**: Python, file I/O.
**Outputs**: Write `agent-results/phase-09-result.md`. Write `optimization_report.md` and `optimization_summary.json`.
**Sub-agents**: May spawn an analyzer subagent per `protocols/analyzer-manifest.schema.md`.

## Runbook

### 1. Gather Artifacts
Read from `{{OUTPUT_DIR}}`:
- `results/benchmark_summary.json` (Phase 3)
- `results/gap_analysis/gap_analysis.json` (Phase 5)
- `results/profile_analysis.json` (Phase 5)
- `results/trace_manifest.json` (Phase 5 — trace integrity and phase-split readiness)
- `problems/optimization_manifest.json` (Phase 6)
- `problems/kernel_type_classification.json` (Phase 6)
- `problems/fusion_opportunities.json` (Phase 6)
- `problems/geak_results.json` (Phase 7)
- `problems/manual_attempts.md` (Phase 7, if GEAK_MODE=manual — optional)
- `results/optimization_comparison.json` (Phase 8)
- `results/integration_manifest.json` (Phase 8 — per-target integration outcomes)
- `results/pipeline_blockers.json` (optional — terminal blockers from failed phases)
- `report/benchmark_report.md` and `profiling_report.md` (Phases 3, 5)
- `results/gpu_arch.json` and `env_info.json`

Cross-check the same logical artifacts via explicit phase paths when the workspace layout mirrors the run root:
- `{{RESULTS_DIR}}/benchmark_summary.json` (Phase 3 aggregated benchmark table)
- `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` (Phase 5 gap analysis)
- `{{OUTPUT_DIR}}/results/profile_analysis.json` (Phase 5 merged TraceLens + gap view)
- `{{OUTPUT_DIR}}/results/trace_manifest.json` (Phase 5 structured trace health)
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (Phase 6 target list)
- `{{PROBLEMS_DIR}}/kernel_type_classification.json` (Phase 6 typing / comm exclusion context)
- `{{PROBLEMS_DIR}}/fusion_opportunities.json` (Phase 6 fusion suggestions)
- `{{PROBLEMS_DIR}}/geak_results.json` (Phase 7 per-kernel speedups + metadata)
- `{{RESULTS_DIR}}/optimization_comparison.json` (Phase 8 measured E2E validation)
- `{{RESULTS_DIR}}/integration_manifest.json` (Phase 8 per-target integration outcomes)
- `{{RESULTS_DIR}}/pipeline_blockers.json` (optional — structured blocker entries)
- `{{REPORT_DIR}}/benchmark_report.md` plus `{{REPORT_DIR}}/profiling_report.md` (narratives from Phases 3 and 5)
- `{{OUTPUT_DIR}}/results/gpu_arch.json` (roofline hardware context) and `{{ENV_INFO_FILE}}` (Phase 0 environment capture)

Optional but valuable when referenced in earlier phases: raw TraceLens CSV snapshots under `{{OUTPUT_DIR}}/results/tracelens_*_csvs/` (attach pointers, not necessarily the full CSV dump, unless the template calls for detailed tables).

**Gathering checklist:** (1) confirm Phase 8 `performance_gate` is present and distinguish clean pass vs accepted warn-band result (if `validated` is false because the run landed in the warn band, say so explicitly), (2) read `integration_manifest.json` for per-target integration coverage — prefer this over inferring coverage from prose, (3) confirm `geak_results.json` lists every kernel referenced in prose, (4) confirm `benchmark_summary.json` throughput matches the tables you cite, (5) attach GPU arch + env info for reproducibility, (6) check for `pipeline_blockers.json` and handle accordingly.

### 2. Generate Report
Create `{{REPORT_DIR}}/optimization_report.md` using the canonical template at `{{TEMPLATES_DIR}}/optimization_report.md` (copy structure/headings verbatim, replace placeholders with measured values). Populate all sections with measured data from artifacts—do not invent numbers.

Before writing prose, verify high-signal JSON parses cleanly (`python3 -m json.tool <file>`). If `optimization_comparison.json` is missing, or if `validated` is `false` / `performance_gate` is not `pass`, the report must state that explicitly—never imply a passing E2E result.

### Pipeline blocker handling

When `results/pipeline_blockers.json` exists:
- Read the blocker entries and populate the `## Blockers` table in the report.
- Set `## Pipeline Status` completion to `pipeline incomplete` when any blocker comes from an early phase (`benchmark`, `profile-analyze`); otherwise use `completed with blockers`.
- For early-phase blockers (benchmark, profile-analyze): skip report sections that depend on those phases' outputs and note which sections are absent.
- For late-phase blockers (kernel-optimize, integration): include all available data but mark the affected sections as incomplete.

When `results/pipeline_blockers.json` is absent or empty:
- If Phase 8 landed in `performance_gate = warn`, set `## Pipeline Status` completion to `completed with warnings`.
- If Phase 8 landed in `performance_gate = fail`, set `## Pipeline Status` completion to `completed with blockers` even when no separate blocker entry was emitted.
- Otherwise set `## Pipeline Status` completion to `completed`.
- Populate the `## Deferred Work` section with data-driven recommendations when the pipeline has no terminal blockers.
- The `## Blockers` table should be omitted or empty.

### What the report must cover
- **Kernel optimization results:** Table or bullet summary of each targeted kernel, GEAK mode, baseline vs optimized timing, `speedup`, and whether patch recovery was required.
- **End-to-end comparison:** Throughput / latency deltas from `optimization_comparison.json`, noting concurrency, sequence lengths, and framework version parity between baseline and optimized runs.
- **Pipeline status:** Completion state, phases completed, and terminal blockers.
- **Deferred work** (only when pipeline completed without blockers): Prioritized next steps grounded in data from gap analysis, TraceLens, and manifest entries.

The report should NOT turn unresolved integration failures or incomplete profiling into generic Priority 1-4 recommendations. When blockers exist, the blocker table is the primary action item — do not suggest next steps that assume a clean pipeline.

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

The emitted `optimization_summary.json` aggregates high-level fields consumed by dashboards: `config_key`, `framework`, `pipeline_status` (`completed`, `completed with warnings`, `completed with blockers`, or `pipeline incomplete`), `all_phases_completed` (boolean — true when `pipeline_status` is `completed` or `completed with warnings`; this replaces the former `phases_completed` boolean to avoid collision with the array-based `progress.json.phases_completed`), optional `gpu_arch` / `geak_mode` (when `env-info` parses), throughput + `speedup` + `validated` from `optimization_comparison.json`, integration manifest fields (`integration_plugin_type`, `integration_total_targets`, `integration_integrated`, `integration_blocked`, `integration_coverage_pct`), per-kernel stats (`kernels_attempted`, `kernels_improved`, `patches_recovered`, embedded `kernel_results`), and manifest sizing (`total_problem_files`). Extend the Python emitter only when new machine fields are required—keep the Markdown report as the human narrative.

### 4. Print Artifact Paths
```bash
echo "Report: {{REPORT_DIR}}/optimization_report.md"
echo "Summary: {{REPORT_DIR}}/optimization_summary.json"
```

Archive both files (and the intermediate JSON inputs) in the same artifact bundle your organization uses for audit—future regressions should diff against this exact tarball, not regenerated estimates.

If your team tracks git commits per optimization attempt, record the `CONFIG_KEY`, container image digest, and benchmark script SHA256 in `agent-results/phase-09-result.md` alongside the printed paths so the report maps to an exact reproducible stack.

### Completion
Write `agent-results/phase-09-result.md` with report path and summary path.

Include these sticky fields in `## Key Findings`:
- `final_speedup`: float (end-to-end speedup from optimization_summary.json)
- `report_path`: string (path to the generated optimization_report.md)

---

## Mid-loop refresh (attempt > 1)

When the handoff frontmatter sets `attempt > 1` (or equivalently `report_freshness ∈ {post_loop_convergence, manual_refresh}`), this invocation is a mid-loop refresh, not the initial report. The orchestrator dispatches a refresh after either (a) the systemic RCA returned `terminal_action_systemic = accept_finding` and `budget_mode` flipped to `diagnostic`, or (b) the user explicitly requested an updated report.

Required behavior:

1. **Preserve prior content.** Read `prior_report_path` from the handoff frontmatter and load the existing `optimization_report.md`. Do NOT overwrite it. Prepend a new section with the heading `## Update — <YYYY-MM-DD> (<report_freshness>)` and write the merged document back to the same path.

2. **Required subsections in the update block:**
   - **Loop narrative** — one paragraph per attempt since the last refresh. Each paragraph names: attempt index, what changed in the dispatched hypothesis, what the per-phase RCA concluded, and the verdict.
   - **Structural diagnosis** — paste the systemic RCA's `summary` and `root_cause_class`. Quote `key_signal_names` verbatim. Include the fingerprint hash for cross-reference.
   - **Attribution honesty** — for every headline e2e number reported in the prior section, state explicitly whether `e2e_attributable` is `true`, `false`, or `null`. If a prior attempt's headline turned out to be a `cache_warmup_artifact`, mark it `[retracted]` here rather than editing the prior section.
   - **Headline numbers (current)** — the most recent attempt's e2e_speedup, ttft_regression_pct, and counter activation summary. If counters=0, lead with that.
   - **Recommended next-steps** — concrete actions outside this loop's authority (e.g. "rewire patch site to `FusedMoE._do_dispatch` per attempt-5 evidence"; "re-run Phase 7 with the production cudagraph captured baseline"). Do NOT propose another attempt of the same kind the loop already exhausted.

3. **Do NOT regenerate `optimization_summary.json` from prior attempts' artifacts blindly.** When `budget_mode == "diagnostic"`, the JSON's `pipeline_status` should be `"completed with structural diagnosis"`. When `budget_mode == "extended"` and the user lifted the cap, status reflects the latest attempt.

4. **Result frontmatter** — set `report_freshness` to match the handoff and emit `prior_report_path` under `## Artifacts` so the audit trail is preserved.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
