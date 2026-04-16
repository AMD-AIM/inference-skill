# Runner-to-LLM Boundary

This document defines the interface between the deterministic runner (control plane) and LLM agents (execution plane).

## Principle

The runner handles all mechanical work: sequencing, prerequisites, retry budgets, context assembly, progress tracking. LLM agents handle judgment: intake UX, phase execution, monitor explanation, RCA interpretation.

## Request Envelope

When the runner dispatches a phase agent, it sends a request with this structure:

```yaml
---
schema_version: "1.0"
phase: {phase_key}
phase_index: {NN}
attempt: {N}
mode: {mode}
handoff_path: handoff/to-phase-{NN}.md
output_path: agent-results/phase-{NN}-result.md
timeout_minutes: {from registry}
---
```

The handoff document at `handoff_path` contains the full context assembled by the runner.

## Allowed Writes

Phase agents may write to:

- `agent-results/phase-{NN}-result.md` (required)
- Files under `results/`, `profiles/`, `optimized/`, `problems/`, `report/` (phase-specific artifacts)

Phase agents must NOT write to:

- `progress.json` (runner-only)
- `monitor/` (monitor agent only)
- `handoff/` (runner-only)
- `config.json` (immutable after bootstrap)
- `results/parity/` (runner-only)

## Required Outputs

Every phase agent must produce `agent-results/phase-{NN}-result.md` following `protocols/phase-result.schema.md`. The result must include:

- YAML frontmatter with `phase`, `phase_index`, `status`, `timestamp`
- `## Key Findings` with required scalar fields for the phase (see phase-result.schema.md)
- `## Artifacts` listing all files produced

## Failure Taxonomy

When the runner detects a failure, it classifies it:

| Error Type | Description | Recovery |
|------------|-------------|----------|
| `schema_invalid` | Artifact fails schema validation | Re-dispatch with validation feedback |
| `missing_artifact` | Required artifact not produced | Re-dispatch with explicit artifact list |
| `monitor_error` | Monitor agent failure (crash, malformed output) | Fail-closed on critical phases (Commit 3) |
| `timeout` | Phase exceeded wall-clock limit | Treat as FAIL, route through RCA |
| `budget_exhausted` | Retry/total rerun limits exceeded | Fallback or stop |
| `manual_intervention_required` | Unrecoverable state needs human | Stop with structured blocker |

## Backward Compatibility

- The runner reads `progress.json` written by the legacy orchestrator (schema_version "1.0").
- The legacy orchestrator can resume runs started by the runner (downward compatible).
- `schema_version` mismatches cause a resume rejection, not a crash.

## V2: Two-Layer Verdict Authority

When `V2_MONITOR=true`, the verdict determination uses a two-layer model:

- **Layer 1 (runner, deterministic)**: Structured predicates from `detection_rules_structured_v2` in the registry, evaluated by `predicate_engine.evaluate_predicates_v2()`. Produces a floor verdict, problem categories, and per-rule details. Results written to `monitor/phase-{NN}-predicate.json`.
- **Layer 2 (LLM monitor)**: Judgment-based evaluation by the monitor agent. Receives L1 results plus expanded file inputs. Can **upgrade** the L1 verdict (PASS->WARN, WARN->FAIL) but **never downgrade** it.

Final verdict = `max(L1, L2)` by severity rank (PASS=0, WARN=1, FAIL=2).

If L2 fails (exception, timeout, malformed output), the runner falls back to the L1 verdict. This is safe because L1 is a floor — L2 can only upgrade.

### Verdict vs Response

PASS, WARN, and FAIL are **verdicts** — they describe what was observed. REDIRECT and ABORT are **response actions** — they describe what to do about it. The response policy engine (`response_policy.py`) maps verdicts to actions using a priority chain:

1. Safety stop (RCA `stop_with_blocker`) -> abort (non-overridable)
2. Human override (escalation response) -> follow human's choice
3. Budget constraint (exhausted) -> redirect to fallback or abort
4. RCA recommendation -> retry or fallback
5. Default -> retry if budget remains

### Escalation Boundary

`runner.py` cannot call `AskUserQuestion`. When escalation is needed:

1. Runner writes `monitor/escalation-request-phase-{NN}.json` and returns `RunnerState` with `status: "escalation_pending"`.
2. Claude reads the returned state, presents escalation to user, writes `monitor/escalation-response-phase-{NN}.json`, re-invokes `run()`.
3. Runner detects pending state on re-entry, reads response, resumes.

## Context-Budget Contract

The runner enforces bounded context before each dispatch:

1. Resolve `context_sources` from the registry to actual file contents or config values.
2. Large structured data stays in sidecar files; only file paths plus compact summaries enter the handoff.
3. After a phase completes, the runner retains only `phase_key + verdict + sticky_values` in accumulated state.
4. Before dispatch, enforce `max_context_lines` (global default: 500, per-phase overrides in registry). Truncate deterministically with a `[truncated: N lines omitted]` marker.
5. No LLM-based summarization for compression — that would introduce non-determinism.
