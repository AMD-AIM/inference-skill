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

## Context-Budget Contract

The runner enforces bounded context before each dispatch:

1. Resolve `context_sources` from the registry to actual file contents or config values.
2. Large structured data stays in sidecar files; only file paths plus compact summaries enter the handoff.
3. After a phase completes, the runner retains only `phase_key + verdict + sticky_values` in accumulated state.
4. Before dispatch, enforce `max_context_lines` (global default: 500, per-phase overrides in registry). Truncate deterministically with a `[truncated: N lines omitted]` marker.
5. No LLM-based summarization for compression — that would introduce non-determinism.
