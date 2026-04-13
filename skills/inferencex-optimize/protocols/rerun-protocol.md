# Rerun Protocol

Defines failure handling, retry logic, and escalation for the multi-agent pipeline.

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

- `max_reruns_per_phase`: 2
- `max_total_reruns`: 5

## Rerun Flow

1. Monitor returns FAIL verdict with `failure_type` and rerun guidance.
2. Orchestrator increments retry counters.
3. If within limits:
   a. Orchestrator rewrites `handoff/to-phase-{NN}.md` with `## Prior Attempt Feedback`
   b. For infrastructure failures, add `## Environment Check` section
   c. Spawn a fresh phase agent (never reuse failed agents)
4. If per-phase limit exceeded:
   a. Check `fallback_target` in registry
   b. If available and not yet attempted: rollback to the fallback phase, invalidate all outputs from the fallback phase forward, restart from there
   c. If no fallback or fallback already attempted: escalate to user
5. If total limit exceeded:
   a. Stop pipeline
   b. Report failure with full monitor review history
   c. Suggest manual intervention points

## Rollback Procedure

When falling back to a `fallback_target` phase:

1. Identify the fallback phase from the registry
2. Delete all `agent-results/phase-{NN}-result.md` for phases from fallback onward
3. Delete all `handoff/to-phase-{NN}.md` for phases from fallback onward
4. Delete all `monitor/phase-{NN}-review.md` for phases from fallback onward
5. Update `progress.json`: remove rolled-back phases from `phases_completed`
6. Resume dispatch from the fallback phase

## Escalation Report

When stopping due to limits exceeded, provide the user:

- Which phase failed and how many attempts were made
- The failure type and monitor's analysis
- The full sequence of monitor reviews for this phase
- Suggested next steps (manual fix, config change, environment check)
