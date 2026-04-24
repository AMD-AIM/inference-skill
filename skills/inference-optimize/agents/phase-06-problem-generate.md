# Phase 6: Upstream Resolve (filename kept as `problem-generate` for registry compatibility)

## Instructions

You are a phase agent responsible for resolving each profile-dominant kernel
to its upstream source location, forking the relevant libraries at pinned
commits, and emitting the optimization manifest that drives Phase 7. You
read exactly 2 files: this document and your handoff at
`handoff/to-phase-06.md`.

This phase **replaces** the legacy "extract-shapes / analyze-fusion /
generate-problems" path. Synthetic harness construction (`problem_*.py`
files, `fusion_opportunities.json`, `bottleneck_kernels.json`,
`roofline_bottlenecks.json`, `model_shapes.json`) is removed. The pipeline
now optimizes the upstream source file in place; Phases 7-8 rebuild the
forked library and verify dispatch via rocprofv3.

**Tools**: Shell commands, Python, file I/O, `AskUserQuestion`.
**Outputs**: Write `agent-results/phase-06-result.md`. Write the manifest,
forks, and resolution audits as documented in *Outputs* below.
**Errors**: If `gap_analysis.json` is missing, report failure immediately
- this is a hard prerequisite.

## Runbook

### 1. Verify Prerequisites
```bash
[ -f "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" ] || { echo "ERROR: gap_analysis.json missing"; exit 1; }
[ -f "{{OUTPUT_DIR}}/results/profile_analysis.json" ] || { echo "ERROR: profile_analysis.json missing"; exit 1; }
mkdir -p "{{PROBLEMS_DIR}}" "{{OUTPUT_DIR}}/forks" "{{OUTPUT_DIR}}/refs" "{{RESULTS_DIR}}"
```

### 2. Capture Baseline Dispatch Trace (rocprofv3)

Run rocprofv3 once on the already-completed Phase-2 baseline launcher to
capture the ground-truth dispatched symbol set. The display name in
TraceLens often differs from the true library symbol; the classifier in
step 4 matches the runtime symbol.

```bash
python3 "{{SCRIPTS_DIR}}/integrate/verify_dispatch.py" \
    --output-dir "{{OUTPUT_DIR}}" \
    --mode baseline-only \
    --launcher "$BASELINE_LAUNCHER"
```

Writes `{{RESULTS_DIR}}/baseline_dispatch_trace.json`. Reused by Phase 8 as
the "before" image for dispatch-swap verification.

### 3. Build Top-Kernel Candidate List

Read `profile_analysis.json` and select the kernels above
`OPTIMIZE_PRIORITY_THRESHOLD`. Exclude communication-class entries (use
`classify_kernel.py`'s `SKIP_KERNEL_TYPES` for that filter).

### 4. Resolve Each Candidate to Upstream Source

For each candidate symbol (using the actual runtime symbol from
`baseline_dispatch_trace.json`), call:

```bash
python3 "{{SCRIPTS_DIR}}/optimize/resolve_upstream_source.py" \
    --symbol "$SYMBOL" \
    --vllm-version "{{VLLM_VERSION}}"
```

Returns the matching `kernel_source_map.yaml` entry as JSON. The entry
carries `library`, `source_form`, `bucket` (A | B | C), tentative
`geak_strategy`, `gating_reason` (Bucket B only), `upstream_repo`,
`source_file`, `library_test_path`, `library_test_command`,
`rebuild_command`, `expected_dispatch_symbols`, `vendor_baseline_symbols`,
`cost_profile`, and `geak_task_hint`. Exit code 1 means the symbol did not
match any pattern - record under `unresolved_kernels.json`.

### 5. Fork Required Upstream Repos

Once the per-kernel resolutions are in hand, clone (or update) every
referenced upstream library at its pinned commit:

```bash
python3 "{{SCRIPTS_DIR}}/optimize/fork_upstream.py" \
    --output-dir "{{OUTPUT_DIR}}" \
    --vllm-version "{{VLLM_VERSION}}"
```

Idempotent: re-uses existing checkouts. Writes `{{OUTPUT_DIR}}/forks/<lib>/`
on a `geak/main` branch and records
`{{OUTPUT_DIR}}/forks/manifest.json` with `repo_url`, `pinned_commit`,
`fork_path`, `dirty`, `rebuild_command`, plus the global
`ck_branch_merged_status` (probed against GEAK upstream's
`feature/ck-preprocess-main`). When `ck_branch_merged_status == true`,
`ck_template` rows in the source map may be promoted from Bucket B to
Bucket A (treated per the `hip_cpp` row's inner-loop rules).

### 6. Resolve Redirect Targets

For every kernel whose tentative `geak_strategy` is
`dispatch_redirect_to_triton` or `dispatch_redirect_to_open_lib`, look up
the recipe in `resources/redirect_recipes.yaml` and emit
`{{PROBLEMS_DIR}}/redirect_plan.json` with one record per redirect:

```json
{
  "source_symbol": "...",
  "source_lib":   "...",
  "target_symbol": "...",
  "target_lib":   "...",
  "target_file":  "...",
  "dispatch_site_file": "...",
  "dispatch_site_patch_hint": "..."
}
```

Prefer `dispatch_redirect_*` over `in_place_optimize_no_harness` whenever
a redirect target exists -- the redirected target inherits Bucket A's
per-kernel pytest gating.

### 7. Bucket B Confirmation Gate (`AskUserQuestion`)

For every kernel whose tentative strategy is
`in_place_optimize_no_harness`, raise an `AskUserQuestion` (batched into
a single multi-question call when several apply):

> "Kernel `<name>` (source form `<form>`, library `<lib>`, gating reason
> `<gating_reason>`) has no built-in per-kernel test harness / requires a
> multi-hour rebuild cycle. GEAK will still optimize it, but per-kernel
> validation falls back to the Phase-8 E2E benchmark only (no per-kernel
> pytest gate). Estimated cost: `<per_attempt_minutes>` per attempt x up
> to 5 attempts; rebuild `<rebuild_minutes>` per cycle. Proceed?"
>
> Options: `proceed_with_warning` | `skip_this_kernel` |
> `redirect_if_possible` (only offered when a redirect target exists).

Persist user choices into the manifest's `user_decision` field so reruns
are deterministic. Resolve the final `geak_strategy` from the user's
choice:
- `proceed_with_warning` -> keep `in_place_optimize_no_harness`
- `skip_this_kernel` -> `unfeasible_record_only` with
  `skip_reason: user_declined_no_harness_path` (or
  `user_declined_rebuild_too_expensive` for `aten_native`)
- `redirect_if_possible` -> swap to the matching `dispatch_redirect_*`
  strategy

### 8. Bucket B Reference Capture

For every kernel whose final `geak_strategy ==
in_place_optimize_no_harness`, invoke once:

```bash
python3 "{{SCRIPTS_DIR}}/optimize/capture_kernel_reference.py" \
    --kernel "$NAME" \
    --output "{{OUTPUT_DIR}}/refs/${NAME}_bf16.npz"
```

The script runs the baseline vLLM, captures the kernel's input/output
bf16 tensors over a representative decode + prefill batch, and writes the
npz that `no_harness_fallback_test.py` (Phase 7) and
`allocator_integration_test.py` will diff against. Failures here demote
the kernel to `unfeasible_record_only` with
`skip_reason: reference_capture_failed`.

### 9. Emit Optimization Manifest

Write `{{PROBLEMS_DIR}}/optimization_manifest.json` (kept filename, new
schema) - an array of:

```
{
  "name", "baseline_dispatch_symbol",
  "library", "source_form", "bucket", "geak_strategy",
  "gating_reason": null|no_test_harness|rebuild_too_expensive,
  "user_decision": null|proceed_with_warning|skip_this_kernel|redirect_if_possible,
  "upstream_repo", "pinned_commit", "source_file", "fork_path",
  "library_test_path", "library_test_command",
  "expected_dispatch_symbols", "vendor_baseline_symbols",
  "rebuild_command", "profiling_pct", "priority_score",
  "cost_profile", "optimize": bool, "enabled": bool,
  "skip_reason": str|null
}
```

`optimize=true` only when `geak_strategy != unfeasible_record_only` AND
priority gate passes. Populate `skip_reason` when `optimize=false`
(values include: `no_open_alternative`, `user_declined_no_harness_path`,
`user_declined_rebuild_too_expensive`, `unresolved_unknown_symbol`,
`priority_below_threshold`, `reference_capture_failed`).

Also write:

- `{{PROBLEMS_DIR}}/redirect_plan.json` (only when at least one kernel
  uses a `dispatch_redirect_*` strategy)
- `{{PROBLEMS_DIR}}/kernel_source_map_resolved.json` -- per-kernel lookup
  result, axis-1/axis-2 classification, strategy decision, reason if
  unresolved or unfeasible
- `{{PROBLEMS_DIR}}/unresolved_kernels.json` -- kernels with
  `library: unknown` that need human triage

### Completion

Write `agent-results/phase-06-result.md` with the new scalar fields.
Include in `## Key Findings` for monitor consumption:

- `upstream_resolved_count`: integer (kernels resolved to a known library)
- `unresolved_unknown_count`: integer (kernels with library == unknown)
- `unresolved_unknown_pct_of_top_time`: float (percent of top-N GPU time
  that remains unresolved)
- `forks_pinned_count`: integer (libraries successfully checked out)
- `forks_required_count`: integer (libraries that needed forking)
- `bucket_a_count`, `bucket_b_count`, `bucket_c_count`: integer
- `bucket_b_user_proceed_count`, `bucket_b_user_skip_count`,
  `bucket_b_user_redirect_count`: integer (must sum to `bucket_b_count`
  before this phase passes)
- `dispatch_redirect_planned_count`: integer
- `baseline_dispatch_trace_captured`: bool
- `ck_branch_merged_status`: bool

Reference `problems/optimization_manifest.json`,
`problems/redirect_plan.json`, `problems/kernel_source_map_resolved.json`,
`forks/manifest.json`, and `results/baseline_dispatch_trace.json` in
`## Artifacts`.

Do NOT write to `progress.json` -- the orchestrator manages progress
tracking.

### Outputs

- `{{OUTPUT_DIR}}/forks/<lib>/` - checkout at pinned commit on a `geak/main`
  branch.
- `{{OUTPUT_DIR}}/forks/manifest.json` -- per-library
  `{repo_url, pinned_commit, fork_path, dirty, rebuild_command}`, plus
  global `ck_branch_merged_status: bool` and `vllm_version`.
- `{{PROBLEMS_DIR}}/optimization_manifest.json` -- canonical optimization
  manifest (new schema above).
- `{{PROBLEMS_DIR}}/redirect_plan.json` -- when redirects planned.
- `{{PROBLEMS_DIR}}/kernel_source_map_resolved.json` -- resolution audit.
- `{{PROBLEMS_DIR}}/unresolved_kernels.json` -- triage list.
- `{{RESULTS_DIR}}/baseline_dispatch_trace.json` -- rocprofv3 baseline.
- `{{OUTPUT_DIR}}/refs/<kernel>_bf16.npz` -- one per Bucket B opt-in.

### Removed Outputs (do NOT emit)

`problem_*.py`, `fusion_opportunities.json`, `bottleneck_kernels.json`,
`roofline_bottlenecks.json`, `model_shapes.json`,
`kernel_type_classification.json` (the old per-file classifier output;
classification is now embedded inline in the manifest).
