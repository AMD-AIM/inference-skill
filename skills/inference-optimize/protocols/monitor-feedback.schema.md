# Monitor Feedback Schema

This is the canonical artifact contract for monitor reviews written to:

- `monitor/phase-{NN}-review.md`
- `monitor/phase-{NN}-review-attempt{N}.md` (retry attempts)

`orchestrator/monitor.md` defines monitor behavior and taxonomy. This file defines the review file shape.

## Verdict is binary

`verdict` MUST be exactly `PASS` or `FAIL`. No other values are
accepted in newly written monitor reviews. This includes (but is not
limited to) the following hybrid strings, which are explicitly
forbidden because they let monitor judgment silently override
structured gates:

- `PASS_with_caveats`
- `PASS_with_warnings`
- `FAIL_pushed_through`
- `WARN`
- `WARNING`
- `MIXED`

If the monitor wants to record caveats, retry guidance, or partial
findings, those belong in `## Summary`, `## Caveats`, `## Failure
Details`, or `## Rerun Guidance` — never in the verdict value.

A monitor that emits an invalid verdict string is treated as an
orchestration failure, not as a phase PASS. The phase-orchestrator's
self-checklist gate refuses to advance until the review file is
rewritten with a binary verdict.

### Legacy artifact normalization (resume only)

When the runner reads a pre-existing review file during resume (for
example after a long pause), it applies a one-shot normalization
shim:

| Legacy verdict       | Normalized verdict |
|----------------------|--------------------|
| `WARN`               | `FAIL`             |
| `PASS_with_caveats`  | `FAIL`             |
| `FAIL_pushed_through`| `FAIL`             |
| any other unknown    | `FAIL`             |

Normalization runs on read only. The runner never rewrites the legacy
file; it just refuses to treat the review as PASS for a critical
phase that already has a blocker recorded in `progress.json` or
`pipeline_blockers.json`.

## Base format (V1 and V2)

```markdown
---
phase: {phase_key}
phase_index: {NN}
verdict: PASS | FAIL
failure_type: infrastructure | logic | data_quality  # required when verdict=FAIL
---

## Summary
(1-2 sentence quality assessment; caveats live here, never in `verdict:`)

## Check Results
- {check_name}: {PASS|FAIL} -- {detail}
- {check_name}: {PASS|FAIL} -- {detail}

## Failure Details
(only present on FAIL)
What failed and why. Include the failure_type justification.

## Rerun Guidance
(only present on FAIL)
Specific corrective guidance for the next attempt.

## Caveats
(optional, present on PASS or FAIL)
Caveats about reported scalars, tolerated drift, or follow-ups that do
NOT block this phase. The verdict remains binary; this section
captures color commentary only.
```

## V2 extension fields

When `V2_MONITOR=true`, keep the base format and extend frontmatter with:

- `l1_verdict: PASS | FAIL`
- `escalation_required: false | "human" | "systemic_rca" | "manual_edit"`
- `problem_categories: []`

In V2 mode, `failure_type` may use the expanded taxonomy documented in `orchestrator/monitor.md`.

## Field semantics

- **verdict**: `PASS` when checks and detection guidance are satisfied; `FAIL` otherwise. No other strings are accepted on newly written reviews.
- **failure_type**: required for non-PASS verdicts.
- **Check Results**: one line per monitor check selected by orchestrator monitor level.
- **Rerun Guidance**: consumed by orchestration retry flow and copied into handoff feedback.
