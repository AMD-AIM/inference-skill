# Rerun Protocol

Defines failure handling, retry logic, RCA-first recovery, and escalation for the multi-agent pipeline.

## Failure Taxonomy

The monitor assigns one of three failure types in its review:

### infrastructure
Container crash, SSH timeout, GPU not available, disk full, Docker pull failure.

**Remediation**:
- Retry with backoff
- Check container health before respawning phase agent
- Verify GPU access
- Add `## Environment Check` section to handoff

### logic
Wrong analysis approach, incorrect script invocation, bad parameter choice, script bug.

**Remediation**:
- Include monitor's feedback in handoff so the fresh agent takes a different approach
- Highlight the specific mistake in `## Prior Attempt Feedback`

### data_quality
Missing expected output files, metrics outside plausible range, empty traces, truncated results.

**Remediation**:
- May require rerunning a dependency phase
- Use `fallback_target` from registry if available
- Verify input artifacts from dependency phases

## Retry Limits

- `max_per_phase`: `0` in the shipped registry. Positive values cap per-phase re-dispatches; `0` or negative values mean uncapped retries.
- `max_total`: `0` in the shipped registry. Positive values cap total re-dispatches across the run; `0` or negative values mean uncapped retries.

## RCA-First Recovery Flow

Every critical phase uses the same orchestrator-managed recovery loop. The RCA step is inserted **between** the monitor FAIL and the phase retry.

1. Monitor returns FAIL verdict with `failure_type` and rerun guidance.
2. **Spawn RCA agent** (does NOT increment counters):
   a. Orchestrator reads `phases[phase_key].rca_artifact` from the registry.
   b. Orchestrator constructs an `analyzer_manifest` from `rca_artifact.analysis_context`.
   c. Analysis agent writes `rca_artifact.output` (e.g. `results/integration_rca.json`).
3. **Increment retry counters** (`phase_reruns` and `total_reruns`).
4. **Check budget limits** (`max_per_phase`, `max_total`) only when those limits are positive. Because the counters were incremented for the rerun that is about to be dispatched, the budget is exhausted only when a counter becomes **greater than** its configured limit.
5. If budget remains and RCA does not recommend `stop_with_blocker`:
   a. Rewrite handoff with `## Prior Attempt Feedback` and `## Root Cause Analysis`.
   b. For infrastructure failures, also add `## Environment Check` section.
   c. Spawn a fresh phase agent (never reuse failed agents).
6. If budget exhausted:
   a. Check `fallback_target` in registry.
   b. If available and not yet attempted: rollback to the fallback phase, invalidate all outputs from the fallback phase forward, restart from there.
   c. If no fallback or fallback already attempted: write structured blocker entry to `results/pipeline_blockers.json`, then apply `terminal_policy` (`stop` or `allow_partial_report`).
7. If RCA recommends `stop_with_blocker` even when budget remains:
   a. Write structured blocker entry.
   b. Apply `terminal_policy`.

### RCA Budget Rules

- Spawning the RCA agent does **not** increment `phase_reruns` or `total_reruns`.
- Only the subsequent phase re-dispatch increments the retry counters.
- Equality with a configured positive limit still allows that re-dispatch. Exhaustion starts on the next failure, when `phase_reruns > max_per_phase` or `total_reruns > max_total`.
- If the RCA agent fails (timeout, crash): record an RCA failure note in the handoff, allow one plain retry if retries are uncapped or finite budget remains.
- If RCA repeatedly fails and finite retry budget is exhausted: emit a structured blocker, apply normal fallback/stop.

## Common RCA Schema

Every `*_rca.json` written by the analysis agent includes:

- `phase` — string, the phase key
- `failure_type` — string, monitor-assigned failure type
- `summary` — 1-3 sentence plain-text summary
- `evidence` — array of `{ "file": "<path>", "finding": "<observation>" }`
- `root_causes` — array of `{ "cause": "<description>", "confidence": "high | medium | low" }`
- `retry_recommendation` — one of: `retry_same`, `retry_with_changes`, `fallback`, `stop`
- `retry_guidance` — freeform prose guidance for the retrying phase agent
- `terminal_action` — one of: `stop_with_blocker`, `continue`, or `null` (per `protocols/rca.schema.json`, the authoritative contract). Use `stop_with_blocker` only when the failure is unrecoverable on retry; use `continue` (or omit / `null`) otherwise. Note: retry vs. fallback intent is expressed via `retry_recommendation` (`retry_same | retry_with_changes | fallback | stop`), not via `terminal_action`. The orchestrator branches on `terminal_action == "stop_with_blocker"` only.
- `suggested_fallback_target` — string or null
- `blocker_classifications` — array of `{ "target": "<name>", "classification": "<enum>" }`

### Phase-Specific RCA Fields

**Benchmark** (`results/benchmark_rca.json`):
- `missing_artifacts`, `harness_issue`, `environment_issue`

**Profile** (`results/profile_rca.json`):
- `trace_integrity_failures`, `missing_phase_split_inputs`, `tracelens_readiness`

**Problem-generate** (`results/problem_gen_rca.json`):
- `unresolved_symbols`, `unresolved_pct_of_top_time`, `forks_failed_to_pin`,
  `bucket_b_user_decision_gaps`, `baseline_dispatch_trace_status`,
  `ck_branch_merged_status`

**Kernel-optimize** (`results/kernel_opt_rca.json`):
- `failed_kernels`, `library_test_failures`, `allocator_test_status`,
  `dispatch_pre_flight_status`, `redirect_application_status`,
  `bucket_b_unverified_count`, `next_attempt_mode`

**Integration** (`results/integration_rca.json`):
- `dispatch_verification_status` (`verified | unverified | redirect_not_honored`),
  `vendor_symbol_leaks`, `libraries_failed_to_rebuild`,
  `expected_symbols_missing`, `e2e_attributable`, `headline_attribution_notes`

### Blocker Classification Enums

The single canonical blocker enum used by every per-phase RCA, the
systemic RCA, and the orchestrator's response policy:

| Enum value                                   | Phase that emits it    | Meaning |
|----------------------------------------------|------------------------|---------|
| `needs_library_fork`                         | problem-generate       | A required upstream library was not in `library_pins.yaml` or could not be cloned at the pinned commit. |
| `unresolved_unknown_symbol`                  | problem-generate       | A profile-dominant symbol did not match any pattern in `kernel_source_map.yaml`; needs human triage. |
| `kernel_source_map_stale_for_pinned_commit`  | problem-generate / kernel-optimize | The resolved `source_file` or `library_test_path` does not exist at the pinned commit; the source map needs a vLLM-version refresh. |
| `unfeasible_ck_template`                     | problem-generate       | A `ck_template` kernel hit `ck_branch_merged_status == false` and the user did not opt in via Bucket B. |
| `unfeasible_vendor_binary`                   | problem-generate       | The dominant kernel is a `closed_vendor_binary` with no source. Recorded for Phase 9; not retried. |
| `unfeasible_handwritten_asm`                 | problem-generate       | Hand-written GCN/SASS; recorded for Phase 9; not retried. |
| `unfeasible_aten_rebuild_too_expensive`      | problem-generate       | `aten_native` candidate where the user declined the multi-hour PyTorch rebuild path. |
| `library_test_failure`                       | kernel-optimize        | The library's own pytest regressed against the patched fork (Bucket A only). |
| `allocator_test_failure`                     | kernel-optimize        | The allocator-equivalent integration test diverged or HSA-faulted (Bucket A only). |
| `redirect_not_honored`                       | kernel-optimize / integration | The dispatch-site patch did not actually swap the runtime symbol; vendor symbol still fires. |
| `dispatch_unverified`                        | kernel-optimize / integration | rocprofv3 confirmed the expected GEAK-optimized symbol does not appear in the trace; the patched kernel never ran. |
| `needs_rebuild_fix`                          | integration            | One or more libraries in `forks/manifest.json` failed `pip install -e .` (or the editable install did not shadow the wheel copy). |
| `framework_limit`                            | integration            | The optimization is correct but vLLM/torch.compile/HIPGraph behavior makes it unreachable on the production decode path. |
| `baseline_drift`                             | integration            | Phase-2 baseline numbers drifted between the baseline run and the post-rebuild benchmark; comparison is unsafe. |

Per-phase `blocker_classifications` entries take the form
`{ "target": "<kernel-or-library-name>", "classification": "<enum>" }`.
Systemic RCA `blocker_classifications` entries may be bare strings.

## Terminal Blocker Behavior

When RCA-informed retry still fails and retry/fallback options are exhausted, the orchestrator writes a structured entry to `results/pipeline_blockers.json`.

### Stop versus Partial Report

Early prerequisite phases (`benchmark`, `profile-analyze`):
- `terminal_policy: stop` — halt the pipeline, do not generate a normal optimization report.

Later optimization phases (`kernel-optimize`, `integration`):
- `terminal_policy: allow_partial_report` — allow Phase 09 to run; report states `completed with blockers` or `pipeline incomplete`.

### Pipeline Status Derivation

`integration_outcome.pipeline_status()` is the single source of truth for the four status strings. All consumers (report template, summary JSON, E2E validator) must use the same function or its output:

| Condition | `pipeline_status` |
|-----------|-------------------|
| No blockers, integration gate = pass | `completed` |
| No blockers, integration gate = warn | `completed with warnings` |
| No blockers, integration gate = fail **or** late-phase blockers present | `completed with blockers` |
| Early-phase blockers (`benchmark`, `profile-analyze`) **or** integration expected but comparison missing | `pipeline incomplete` |

When `pipeline_blockers.json` is absent and `performance_gate = fail`, the fail gate alone is sufficient to set `completed with blockers` — a separate blocker file is not required. Conversely, when `performance_gate = warn` and no blocker file exists, the status is `completed with warnings`, not `completed with blockers`.

## Rollback Procedure

When falling back to a `fallback_target` phase:

1. Identify the fallback phase from the registry
2. Delete all `agent-results/phase-{NN}-result.md` for phases from fallback onward
3. Delete all `handoff/to-phase-{NN}.md` for phases from fallback onward
4. Delete all `monitor/phase-{NN}-review.md` for phases from fallback onward
5. Update `progress.json`: remove rolled-back phases from `phases_completed`
6. Resume dispatch from the fallback phase

## Human Escalation Protocol

When `HUMAN_LOOP=true` and `V2_MONITOR=true`, the orchestrator uses a signal-and-resume pattern: it pauses the pipeline, writes a structured escalation request, and waits for a human response before continuing or aborting.

### Run State Machine

| Current State | Event | Next State |
|---|---|---|
| running | phase PASS/FAIL | running |
| running | all phases complete | completed |
| running | budget exhausted (no fallback) | failed |
| running | RCA stop_with_blocker | failed |
| running | escalation triggered | escalation_pending |
| escalation_pending | response received (retry) | running |
| escalation_pending | response received (abort) | failed |
| escalation_pending | stale timeout (no response) | failed |

Re-escalation is valid: `escalation_pending -> running -> escalation_pending` can occur when a retried phase triggers another escalation.

`completed` and `failed` are terminal states. No transitions leave these states.

## Escalation Report

When stopping due to limits exceeded, provide the user:

- Which phase failed and how many attempts were made
- The failure type and monitor's analysis
- The RCA summary and blocker classifications (when available)
- The full sequence of monitor reviews for this phase
- Contents of `results/pipeline_blockers.json` (when present)
- Suggested next steps (manual fix, config change, environment check)
