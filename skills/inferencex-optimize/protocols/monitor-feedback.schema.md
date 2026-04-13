# Monitor Feedback Schema

The monitor agent writes `monitor/phase-{NN}-review.md` following this format.

## Format

```markdown
---
phase: {phase_key}
phase_index: {NN}
verdict: PASS | WARN | FAIL
failure_type: infrastructure | logic | data_quality
---

## Summary
(1-2 sentence quality assessment)

## Check Results
- [ ] {check_name}: {PASS|FAIL} — {detail}
- [ ] {check_name}: {PASS|FAIL} — {detail}

## Failure Details
(only present on FAIL or WARN)
What went wrong and why. Include the failure_type justification.

## Rerun Guidance
(only present on FAIL)
Specific instructions for what the retry agent should do differently.
Informed by the failure taxonomy:
- infrastructure: retry with backoff, check container health, verify GPU access
- logic: different approach, corrected parameters, alternative script usage
- data_quality: rerun dependency phase, verify input artifacts
```

## Field Definitions

- **verdict**: PASS = all quality checks passed. WARN = minor issues, output usable. FAIL = checks failed, rerun needed.
- **failure_type**: Only present on non-PASS verdicts. One of:
  - `infrastructure` — container crash, SSH timeout, GPU unavailable, disk full
  - `logic` — wrong analysis approach, incorrect script invocation, bad parameter choice
  - `data_quality` — missing output files, metrics outside plausible range, empty traces
- **Check Results**: One entry per quality check from the orchestrator's prompt. For non-critical phases, a single generic `result_exists` check.
- **Rerun Guidance**: The orchestrator copies this into the `## Prior Attempt Feedback` section of the rewritten handoff.
