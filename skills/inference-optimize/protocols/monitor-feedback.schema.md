# Monitor Feedback Schema

This is the canonical artifact contract for monitor reviews written to:

- `monitor/phase-{NN}-review.md`
- `monitor/phase-{NN}-review-attempt{N}.md` (retry attempts)

`orchestrator/monitor.md` defines monitor behavior and taxonomy. This file defines the review file shape.

## Base format (V1 and V2)

```markdown
---
phase: {phase_key}
phase_index: {NN}
verdict: PASS | FAIL
failure_type: infrastructure | logic | data_quality  # required when verdict=FAIL
---

## Summary
(1-2 sentence quality assessment)

## Check Results
- {check_name}: {PASS|FAIL} -- {detail}
- {check_name}: {PASS|FAIL} -- {detail}

## Failure Details
(only present on FAIL)
What failed and why. Include the failure_type justification.

## Rerun Guidance
(only present on FAIL)
Specific corrective guidance for the next attempt.
```

## V2 extension fields

When `V2_MONITOR=true`, keep the base format and extend frontmatter with:

- `l1_verdict: PASS | FAIL`
- `escalation_required: false | "human" | "systemic_rca" | "manual_edit"`
- `problem_categories: []`

In V2 mode, `failure_type` may use the expanded taxonomy documented in `orchestrator/monitor.md`.

## Field semantics

- **verdict**: `PASS` when checks and detection guidance are satisfied; `FAIL` otherwise.
- **failure_type**: required for non-PASS verdicts.
- **Check Results**: one line per monitor check selected by orchestrator monitor level.
- **Rerun Guidance**: consumed by orchestration retry flow and copied into handoff feedback.
