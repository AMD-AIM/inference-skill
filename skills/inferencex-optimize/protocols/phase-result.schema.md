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
- **Artifacts**: Paths relative to `OUTPUT_DIR`. The monitor uses these to verify `file_exists` quality checks.
- **Key Findings**: Used by the monitor for `metric_threshold` checks. Include numeric values with field names matching the registry's quality check `field` names.
- **Data for Next Phase**: The orchestrator uses this section to populate the next handoff's `## Prior Phase Outputs`.
