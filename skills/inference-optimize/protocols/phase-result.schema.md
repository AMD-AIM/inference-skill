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
- `phase_split_inputs_ready`: boolean (true when decode-only trace + phase-split script both exist)
- `roofline_coverage_pct`: float (0-100)

### Phase 06 (problem-generate / upstream-resolve)
- `upstream_resolved_count`: integer (kernels resolved to a known library)
- `unresolved_unknown_count`: integer (kernels with `library == unknown`)
- `unresolved_unknown_pct_of_top_time`: float (percent of top-N GPU time still unresolved)
- `forks_pinned_count`: integer (libraries successfully checked out at the pinned commit)
- `forks_required_count`: integer (libraries that needed forking)
- `bucket_a_count`, `bucket_b_count`, `bucket_c_count`: integer (axis-2 partition counts)
- `bucket_b_user_proceed_count`, `bucket_b_user_skip_count`,
  `bucket_b_user_redirect_count`: integer (must sum to `bucket_b_count`)
- `dispatch_redirect_planned_count`: integer (kernels routed via a `dispatch_redirect_*` strategy)
- `baseline_dispatch_trace_captured`: boolean (rocprofv3 baseline written)
- `ck_branch_merged_status`: boolean (probe of GEAK upstream `feature/ck-preprocess-main`)

### Phase 07 (kernel-optimize)
- `library_tests_passed_count`: integer (sum across Bucket A kernels; null-counted Bucket B kernels excluded)
- `library_tests_failed_count`: integer (sum across Bucket A kernels)
- `allocator_test_pass`: boolean (overall, Bucket A only)
- `dispatch_pre_flight_pass`: boolean (rocprofv3 confirmed expected symbols fire on the rebuilt env)
- `geak_speedup_lib_bench`: float (best per-kernel speedup reported by the library's inner-loop test)
- `redirect_commits_applied_count`: integer (dispatch-site commits placed on `geak/main`)
- `in_place_winners_count`: integer (Bucket A `in_place_optimize` GEAK winners committed)
- `no_harness_winners_count`: integer (Bucket B `in_place_optimize_no_harness` GEAK winners committed)
- `unverified_per_kernel_count`: integer (== `no_harness_winners_count`; counts kernels carrying `optimization_unverified_per_kernel = true` into Phase 9)

### Phase 08 (integration)

From `validate_optimization.py` → `optimization_comparison.json` (schema preserved for adjacent compatibility):
- `artifacts_valid`: bool — both baseline and optimized JSONs loaded without error
- `performance_valid`: bool — `speedup >= 1.0`
- `validated`: bool — `artifacts_valid and performance_valid`
- `performance_gate`: pass | warn | fail
- `speedup`: float
- `e2e_speedup`: float (alias for speedup)
- `ttft_regression_pct`: float or null

These Phase 08 fields are read from `results/optimization_comparison.json` by detection rules (pre-extracted into `monitor/phase-08-context.json` by the orchestrator). Do not wire a `metric_threshold` check to them unless the Phase 08 result doc also mirrors the scalar into `## Key Findings`.

From `results/dispatch_verification.json` (new, library-rebuild contract):
- `dispatch_verified`: boolean — expected GEAK-optimized symbols present and vendor symbols absent
- `expected_symbol_total_count`: integer — sum of dispatch counts for expected symbols
- `vendor_symbol_leaked_count`: integer — count of vendor baseline symbols that still fired
- `redirect_required_count`: integer — number of kernels with a `dispatch_redirect_*` strategy
- `redirect_honored_count`: integer — subset of the above where the redirect actually swapped the runtime symbol

From `results/integration_manifest.json` (schema_version 2.0, library-rebuild contract):
- `schema_version`: `"2.0"`
- `libraries_rebuilt`: array of `{lib, commit, install_log_path}` records
- `dispatch_verified`: boolean (mirrors dispatch_verification.json)
- `e2e_ran`: boolean
- `artifacts_valid`: boolean

Phase 09 reads `integration_manifest.json` directly. The legacy
`plugin_type` / `targets` / `summary.coverage_pct` fields are gone with
the plugin path.

From the Phase 08 agent's result doc (mirrored into `## Key Findings`):
- `dispatch_verified`: boolean
- `expected_symbol_total_count`: integer
- `vendor_symbol_leaked_count`: integer
- `redirect_honored_count`, `redirect_required_count`: integer
- `libraries_rebuilt_ok_count`: integer
- `libraries_rebuild_failed_count`: integer
- `e2e_speedup`: float (from `optimization_comparison.json`)
- `validation_status`: pass | warn | fail (mirrors `performance_gate`)
