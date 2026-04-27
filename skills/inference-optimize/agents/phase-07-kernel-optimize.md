# Phase 7: Kernel Optimize (in-place GEAK on forked source)

## Instructions

You are a phase agent responsible for running GEAK against the upstream
source files in the forks produced by Phase 6, applying winning patches,
running the library's own test suite + the allocator-equivalent
integration test, and pre-flighting dispatch with rocprofv3 before Phase 8
burns wall-clock on the e2e benchmark. You read exactly 2 files: this
document and your handoff at `handoff/to-phase-07.md`.

This phase **replaces** the legacy synthetic-harness path
(`kernel_test_runner.py`, `kernel_finalize.py`,
`verify_winning_kernels.py`, `*_opt.py` files, `traj_*.json`,
`Model`/`ModelNew`/`get_inputs()` convention, `RESULT_JSON:` /
`GEAK_RESULT_LATENCY_MS=` parsers, the `simple` vs `kernel-url` mode
distinction, and the `mini` CLI alias).

**Tools**: Shell commands, Docker, Python, file I/O, `geak` CLI.
**Outputs**: Write `agent-results/phase-07-result.md`. Mutate forks under
`{{OUTPUT_DIR}}/forks/<lib>/` (new commits on `geak/main`). Write
`{{PROBLEMS_DIR}}/geak_results.json` and
`{{RESULTS_DIR}}/preflight_dispatch_trace.json`.

## Runbook

### Progress Reporting
This phase can run up to 90 minutes. Print one-line status updates:
- Before each kernel target: `[phase-07] Optimizing kernel N/M: <name> (<pct>% GPU time, strategy=<S>)...`
- After each GEAK invocation: `[phase-07] GEAK <name>: best_speedup=<X>x via <patch_file>`
- After commit-winners: `[phase-07] Applied N winners to fork <lib>`
- Before library-suite step: `[phase-07] Running library test suites...`
- Before allocator test: `[phase-07] Running allocator-integration test...`
- Before pre-flight: `[phase-07] Pre-flighting dispatch with rocprofv3...`

### Prerequisites

- `{{PROBLEMS_DIR}}/optimization_manifest.json` from Phase 6
- `{{PROBLEMS_DIR}}/redirect_plan.json` (when redirects planned)
- `{{OUTPUT_DIR}}/forks/<lib>/` checkouts on `geak/main`
- `{{OUTPUT_DIR}}/forks/manifest.json`
- `{{ENV_INFO_FILE}}` from Phase 0
- `{{OUTPUT_DIR}}/refs/<name>_bf16.npz` for every Bucket B kernel
- `{{KERNEL_SOURCE_MAP_PATH}}` and `{{LIBRARY_PINS_PATH}}` (skill-owned)

### 1. Load Manifest
```bash
python3 "{{SCRIPTS_DIR}}/optimize/load_optimization_manifest.py" \
    --manifest "{{PROBLEMS_DIR}}/optimization_manifest.json" \
    --optimize-scope "{{OPTIMIZE_SCOPE}}"
```

The loader prints the prioritized kernel queue (highest `priority_score`
first) grouped by `geak_strategy`. Follow that ordering. Skip every
kernel whose `optimize == false`.

### 2. Per-Kernel GEAK Invocation

All GEAK invocations share the canonical CLI shape:

```bash
geak \
    --repo "${FORK_PATH}" \
    --kernel-url "${FORK_PATH}/${SOURCE_FILE}" \
    --test-command "${TEST_COMMAND}" \
    --task "${GEAK_TASK_HINT}" \
    --gpu-ids "${GPU_IDS}" \
    --num-parallel 4 \
    --yolo --exit-immediately \
    --config "{{RESOURCES_DIR}}/geak_override.yaml" \
    -o "${OUTPUT_DIR}/${KERNEL_NAME}_opt/"
```

`--repo` accepts the fork directory directly (it has a valid `.git` HEAD
on `geak/main`). GEAK uses `git worktree add` to isolate per-agent edits
under `optimization_logs/<run>/results/round_1/worktrees/slot_N/`. **Do
NOT pre-step `git init`.**

`--test-command` and `--task` vary by `geak_strategy`:

#### 2a. `in_place_optimize` (Bucket A: `(*, triton)`, `(*, hip_cpp)`, and `(*, ck_template)` only when `ck_branch_merged_status == true`)

- **Triton** -- `--test-command "cd ${FORK_PATH} && pytest ${LIBRARY_TEST_PATH} -x -q"`.
  The library's pytest IS the inner loop; no synthetic harness, no
  incremental rebuild needed (Triton reloads via Python import).
- **HIP/C++** -- `--test-command "cd ${FORK_PATH} && python setup.py build_ext --inplace -q && pytest ${LIBRARY_TEST_PATH} -x -q"`.
  The incremental C++ ext build runs inside `--test-command` after each
  candidate patch (minutes per attempt). For AITER, the same pattern
  applies but `--test-command` may set `AITER_REBUILD=1` before the
  pytest invocation. Full `pip install -e .` is deferred to Phase 8.
- `--task` text comes from the kernel's `geak_task_hint` in
  `kernel_source_map.yaml` (per-kernel scoping guidance: "modify only
  function `<fn>`; do not touch dispatch / bindings"), with the kernel's
  roofline data + measurement-metric description appended.

#### 2b. `dispatch_redirect_to_triton` (Bucket B -> Bucket A)

For `(aiter_ck_template, ck_template)` with a Triton sibling:

1. Apply the dispatch-site patch from `redirect_plan.json`
   (`dispatch_site_patch_hint`) inside the host fork (typically
   `forks/aiter/aiter/__init__.py` or
   `forks/vllm/vllm/attention/...`). The patch forces selection of the
   Triton variant. Commit on `geak/main` as
   `redirect: <source_symbol> -> <target_symbol>`.
2. Then `in_place_optimize` the redirect target's Triton source file
   (`target_file` in `redirect_plan.json`).

#### 2c. `dispatch_redirect_to_open_lib` (Bucket B/C with open equivalent)

For `(hipblaslt_tensile, tensile_asm)`, `(inductor, inductor_codegen)`,
`(aten, aten_native)`, and any `(*, closed_vendor_binary)`:

1. Apply the dispatch-site patch in the host fork (typically vLLM's
   `model_executor/layers/{linear,fused_moe}/` or an FX-graph
   `escape_inductor` patch for inductor) to route the call through the
   open-lib alternative (`target_lib`/`target_symbol`). Commit on
   `geak/main`.
2. Then `in_place_optimize` the open-lib target's source file.

#### 2d. `in_place_optimize_no_harness` (Bucket B with user `proceed_with_warning`)

Covers `(*, ck_template)` on main with no sibling, `(*, tensile_asm)`,
`(*, handwritten_asm)`, `(*, aten_native)`:

- `--test-command` falls back to:
  ```
  python {{SCRIPTS_DIR}}/optimize/no_harness_fallback_test.py \
      --kernel <name> --fork ${FORK_PATH} \
      --reference {{OUTPUT_DIR}}/refs/<name>_bf16.npz
  ```
  The fallback boots a minimal vLLM with the rebuilt fork, runs ONE
  decode iter, prints `latency_ms=<N>` on stdout, exits non-zero on any
  numerical-divergence breach (`max_abs > 1e-2` for bf16).
- For `rebuild_too_expensive` kernels (`aten_native`),
  `--test-command` includes the long rebuild step
  (`pip install -e <pytorch-fork>`); cost-warning was already surfaced to
  the user during Phase 6.
- `--task` text appends an explicit warning: "No per-kernel library test
  exists for this source form; the test command runs a single live decode
  against a bf16 reference. Optimize for `latency_ms` (lower is better)
  without breaching the divergence threshold."
- Commit GEAK winner as in 2a, but commit message includes the
  `[no-harness]` tag.

#### 2e. `unfeasible_record_only`

Not attempted in Phase 7; passes straight through to Phase 9 reporting
with its `skip_reason` from the manifest.

### 3. Commit GEAK Winners

After all per-kernel GEAK runs complete, apply winners to the **main fork
tree** (not GEAK's per-agent worktrees), in order:

```bash
python3 - <<'PY'
import json, os, subprocess, sys
out = "{{OUTPUT_DIR}}/${KERNEL_NAME}_opt/best_results.json"
if not os.path.isfile(out):
    sys.exit(0)
br = json.load(open(out))
patch = br.get("best_patch_file")
patch_id = br.get("best_patch_id")
if not patch or patch_id is None:
    sys.exit(0)  # fall back to scanning patch_*_test.txt below
subprocess.check_call(["git", "-C", "${FORK_PATH}", "apply", patch])
msg = "geak: ${KERNEL_NAME} attempt%s speedup=%s" % (patch_id, br.get("best_patch_speedup"))
if "${STRATEGY}" == "in_place_optimize_no_harness":
    msg += " [no-harness]"
subprocess.check_call(["git", "-C", "${FORK_PATH}", "add", "-A"])
subprocess.check_call(["git", "-C", "${FORK_PATH}", "commit", "-m", msg])
PY
```

For redirects, the dispatch-site commit (`redirect: <src> -> <tgt>`) is
placed first, before the target-symbol GEAK commit.

If `best_results.json` is missing or `best_patch_id` is `null`, fall back
to scanning `optimization_logs/<run>/results/round_1/<kernel>-*/patch_*_test.txt`
files. Each candidate's emitted text is freeform (GEAK's SelectPatch
agent is LLM-driven); rank by lowest reported latency consistent with the
test harness, then map the winning text file back to its sibling
`patch_N.patch` and apply via `git -C ${FORK_PATH} apply`.

### 4. Library-Suite Validation (Bucket A only)

For every Bucket A `optimize=true` kernel, run the library's own test
suite against the now-patched fork:

```bash
python3 "{{SCRIPTS_DIR}}/optimize/library_test_driver.py" \
    --kernel "$NAME" --fork "$FORK_PATH"
```

This is broader than GEAK's per-kernel inner loop -- it catches
cross-kernel regressions and integration-test failures. The path and
command come from `kernel_source_map.yaml` (`library_test_path`,
`library_test_command`).

**Bucket B `in_place_optimize_no_harness` kernels skip this step entirely** -
there is no library test to run; the no-harness fallback `--test-command`
already exercised the candidate. Their `library_tests_*_count` fields are
written as `null` (not 0, to distinguish "skipped by design" from "all
failed").

### 5. Allocator-Equivalent Integration Test (Bucket A only)

Once per-kernel library tests pass, boot a minimal vLLM with the rebuilt
forks and exercise multi-step decode + multi-batch prefill against a
stored bf16 reference:

```bash
python3 "{{SCRIPTS_DIR}}/optimize/allocator_integration_test.py" \
    --kernel "$NAME" \
    --reference "{{OUTPUT_DIR}}/refs/${NAME}_bf16.npz" \
    --fork-root "{{OUTPUT_DIR}}/forks"
```

This is the structural fix for the RCA #5/#6 failure modes
(`true_kernel_parity` divergence and `load_fault` HSA fault under
fragmented allocator).

**Bucket B kernels skip this step too** -- the no-harness fallback
`--test-command` is itself an allocator-equivalent test (it boots the
same live vLLM stack); running it twice adds no signal.

### 6. Pre-Flight Dispatch Sanity Check (all buckets)

```bash
python3 "{{SCRIPTS_DIR}}/integrate/verify_dispatch.py" \
    --output-dir "{{OUTPUT_DIR}}" --mode pre-flight \
    --manifest "{{PROBLEMS_DIR}}/optimization_manifest.json"
```

Confirms the `expected_dispatch_symbols` actually fire on the rebuilt
env -- catches silent "fork installed but kernel never imported" before
Phase 8 burns wall-clock. Writes
`{{RESULTS_DIR}}/preflight_dispatch_trace.json`.

### 7. Emit `geak_results.json`

Write `{{PROBLEMS_DIR}}/geak_results.json` with one record per kernel:

```
{
  "name", "library", "geak_strategy", "gating_reason",
  "optimization_unverified_per_kernel": bool,
  "redirect_target", "upstream_repo", "fork_commit_after_winner",
  "library_test_pass_count": int|null,
  "library_test_fail_count": int|null,
  "library_test_log_path":   str|null,
  "allocator_test_pass":     bool|null,
  "allocator_test_log_path": str|null,
  "dispatch_pre_flight_pass": bool,
  "geak_speedup_lib_bench":   float,
  "geak_attempts_used":       int,
  "skip_reason":              str|null,
  "no_harness_warning":       str|null
}
```

`optimization_unverified_per_kernel` is true for every Bucket B winner;
`library_test_*` and `allocator_test_*` fields are `null` (not 0) for
Bucket B to distinguish "skipped by design" from "ran and all failed".

### Completion

Write `agent-results/phase-07-result.md`. Include in `## Key Findings`
the following flat `field: value` lines. Each scalar must be present;
omit a field only when the registry's note for it explicitly allows
it (e.g. `redirect_count_*` may be `null` when no redirect was
attempted). Booleans should be lower-case `true` / `false`. Integers
are non-negative.

#### Library / allocator / pre-flight scalars (existing contract)

- `library_tests_passed_count` (sum across Bucket A kernels)
- `library_tests_failed_count` (sum across Bucket A kernels)
- `allocator_test_pass`: bool (overall, Bucket A only)
- `dispatch_pre_flight_pass`: bool (all buckets)
- `geak_speedup_lib_bench`: float (best per-kernel speedup reported by
  the library inner-loop)
- `redirect_commits_applied_count`: integer (audit-only — does NOT
  count as a shipped winner)
- `in_place_winners_count`: integer (shipped in-place winners only)
- `redirect_winners_count`: integer (shipped redirect winners only —
  redirect rows that pass dispatch verification, count tolerance, and
  any required performance gate)
- `no_harness_winners_count`: integer (shipped no-harness winners only)
- `unverified_per_kernel_count`: integer (>= `no_harness_winners_count`)

#### Aggregated winner / artifact scalars (required by the registry)

- `winners_total_count`: integer (== `in_place_winners_count` +
  `redirect_winners_count` + `no_harness_winners_count`). Phase 7 must
  FAIL when this is `0`.
- `py_exports_shipped_count`: integer (count of files written under
  `optimized/` for shipped winners). Audit-only when no Python export
  is required.
- `optimized_artifact_count`: integer (count of integration-ready
  files under `optimized/` regardless of file extension). Phase 7
  must FAIL when this is `0` while a winner is claimed.
- `optimized_dir_empty`: bool (`true` when `optimized/` has zero
  files). Surface even when no winners were claimed so Phase 8 can
  refuse to start cleanly.
- `claimed_winner_artifacts_valid`: bool (`true` only when every row
  with `winner_strategy != not_a_winner_*` has its required
  integration artifact or manifest present and well-formed).

#### Redirect verification scalars

- `redirect_attempted`: bool (`true` if any row in `geak_results.json`
  has `geak_strategy` starting with `dispatch_redirect_`).
- `redirect_count_observed`: integer or `null` (count of redirect
  symbol dispatches observed in the post-patch preflight trace).
- `redirect_count_expected`: integer or `null` (sum of empirical
  per-bucket call counts the patch should accept).
- `redirect_count_ratio`: float or `null`
  (`redirect_count_observed / redirect_count_expected`).
- `redirect_count_within_tolerance`: bool or `null` — `true` only when
  observed is inside `[0.5x, 1.5x]` of expected. Phase 7 must FAIL
  when `false` and `redirect_attempted == true`.

#### Mini-A/B harness scalars

- `mini_ab_required`: bool (`true` whenever a redirect winner is
  claimed; future winner types may flip this to `false`).
- `mini_ab_harness_status`: enum: `not_applicable | passed | failed |
  unreliable_high_variance`. Phase 7 must FAIL when value is
  `unreliable_high_variance` and `mini_ab_required == true`.
- `rca_fingerprint`: string or `null` (stable identifier for the
  current failure mode; emitted only on FAIL so the orchestrator can
  detect repeated failures and trigger systemic-rca).

#### Winner accounting rules (no implicit promotion)

- A fork commit is NOT a winner.
- A generated patch source file is NOT a winner.
- A patch that lives only in `forks/` without a corresponding
  integration-ready artifact under `optimized/` (or an explicit
  no-export winner contract documented in the redirect plan) MUST
  NOT be counted in `winners_total_count`.
- A redirect row is a winner only when `dispatch_pre_flight_pass`,
  `redirect_count_within_tolerance`, and any required performance
  gate are all satisfied.
- If every candidate is degraded to `not_a_winner_*`,
  `winners_total_count` MUST be `0` and the phase MUST FAIL.

Reference `problems/geak_results.json` and
`results/preflight_dispatch_trace.json` in `## Artifacts`. Mutated forks
live under `{{OUTPUT_DIR}}/forks/<lib>/` on `geak/main`.

If the handoff contains a `## Root Cause Analysis` section from a prior
failed attempt, read the RCA artifact path and adjust per the new
blocker enum in `protocols/rerun-protocol.md` (e.g.,
`needs_library_fork`, `needs_rebuild_fix`, `dispatch_unverified`,
`redirect_not_honored`, `library_test_failure`, `allocator_test_failure`).

Do NOT write to `progress.json` -- the orchestrator manages progress
tracking.

### Removed Outputs (do NOT emit)

`{{OPTIMIZED_DIR}}/*_opt.py`, `kernel_test_runner.py` traces,
`kernel_finalize.py` outputs, `verify_winning_kernels.py` exit codes,
`traj_*.json`, `*_opt_best.json` files, `_runtime_counters_*.json`.
