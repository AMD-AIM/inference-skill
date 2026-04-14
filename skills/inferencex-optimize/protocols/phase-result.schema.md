# Phase Result Schema

Every phase agent writes `agent-results/phase-{NN}-result.md` following this format.

## Format

```markdown
---
phase: {phase_key}
phase_index: {NN}
status: completed | failed | partial
timestamp: {ISO 8601}
---

## Summary
(1-3 sentences: what the phase accomplished)

## Artifacts
(bulleted list of files produced, with paths relative to OUTPUT_DIR)

## Key Findings
(phase-specific metrics or observations worth surfacing to the monitor)

## Data for Next Phase
(values or file paths that downstream phases will need)

## Errors
(only if status is failed or partial: description of what went wrong)
```

## Field Definitions

- **phase**: Canonical phase key from `phase-registry.json` (e.g., `env`, `config`, `benchmark`).
- **phase_index**: Zero-based index matching the registry.
- **status**: `completed` = all steps succeeded. `failed` = unrecoverable error. `partial` = some steps succeeded, some failed.
- **Artifacts**: Paths relative to `OUTPUT_DIR`. The monitor uses these to verify `file_exists` quality checks. Reference JSON manifests here when present (e.g., `results/trace_manifest.json`, `results/integration_manifest.json`).
- **Key Findings**: Used by the monitor for `metric_threshold` checks and detection rule evaluation. Include numeric values as flat `field_name: value` lines matching the registry's quality check `field` names.
- **Data for Next Phase**: The orchestrator uses this section to populate the next handoff's `## Prior Phase Outputs`.

## Required Scalar Fields by Phase

Phase agents MUST include these fields in `## Key Findings` so the monitor can evaluate them deterministically:

### Phase 02 (benchmark)
- `benchmark_result_status`: completed | failed | partial
- `benchmarks_succeeded`: integer count of successful benchmark runs
- `baseline_artifacts_ready`: true | false

### Phase 05 (profile-analyze)
- `trace_integrity_status`: valid | corrupt | missing
- `tracelens_status`: completed | skipped | failed
- `phase_split_status`: completed | skipped | unavailable
- `trace_count`: integer
- `world_size`: integer

### Phase 07 (kernel-optimize)
- `compiled_count`: integer count of compiled kernels
- `best_speedup`: float (best kernel-level speedup achieved)
- `winning_kernel_count`: integer count of kernels with speedup > 1.0
- `optimization_coverage_status`: complete | partial | none
- `expected_improvement_status`: improvable | parity_or_blocked (per hot target, summarize)

### Phase 08 (integration)

From `validate_optimization.py` → `optimization_comparison.json`:
- `artifacts_valid`: bool — both baseline and optimized JSONs loaded without error
- `performance_valid`: bool — `speedup >= 1.0`
- `validated`: bool — `artifacts_valid and performance_valid` (backward-compatible)
- `performance_gate`: pass | warn | fail
- `speedup`: float
- `e2e_speedup`: float (alias for speedup)
- `ttft_regression_pct`: float or null

These Phase 08 fields are read from `results/optimization_comparison.json` by detection rules (pre-extracted into `monitor/phase-08-context.json` by the orchestrator). Do not wire a `metric_threshold` check to them unless the Phase 08 result doc also mirrors the scalar into `## Key Findings`.

From `Phase 08` → `results/integration_manifest.json`:
- `schema_version`: string
- `plugin_type`: string (e.g. `sglang_plugin`, `vllm_plugin`)
- `comparison_file`: string (filename of the comparison JSON)
- `targets`: array of per-target integration outcomes
- `summary.coverage_pct`: float — fraction of targets integrated

Phase 09 reads this manifest directly for per-target coverage instead of inferring it from prose.

From the Phase 08 agent's result doc:
- `baseline_file`: filename of the baseline JSON used
- `optimized_file`: filename of the optimized JSON used
- `validation_status`: pass | warn | fail
- `coverage_pct`: float — fraction of Phase 07 winners integrated
- `blocked_target_count`: integer — targets with a structured blocker
- `critical_blocker_count`: integer — blocked targets where classification is not `true_kernel_parity`
