# Handoff Document Format

The orchestrator writes `handoff/to-phase-{NN}.md` for each phase agent. This is the phase agent's primary input alongside its own `agents/phase-NN-*.md` doc.

## Format

```markdown
---
phase: {phase_key}
phase_index: {NN}
attempt: {N}
mode: "{MODE}"
---

## Context
(populated mechanically from phase-registry.json `required_context` for this phase,
resolved via `context_sources`; sorted alphabetically by variable name)

- **CONFIG_KEY**: ...
- **GPU_ARCH**: ...
- **OUTPUT_DIR**: ...
- **SCRIPTS_DIR**: ...

## Instructions
(phase-specific overrides, filter selections, user preferences that the orchestrator
passes through from the intake)

Execute phase {phase_key} (index {NN}).
Write results to agent-results/phase-{NN}-result.md.

## Prior Attempt Feedback
(only present on reruns — monitor's failure comments + failure_type + remediation guidance)

### Failure Type: {infrastructure | logic | data_quality}
{monitor's rerun guidance from the review}

### Environment Check
(only present for infrastructure failures — prompts the phase agent to verify
prerequisites before proceeding)

## Root Cause Analysis
(only present on reruns after RCA agent has run — orchestrator-constructed from RCA artifact)

- **RCA artifact**: {path to *_root_cause.json relative to OUTPUT_DIR}
- **Summary**: {1-2 sentence RCA summary from root_cause.json}
- **Retry recommendation**: {retry_recommendation field from RCA}
- **Blocker classifications**: {summary of target/classification pairs from RCA}
```

## Generation Rules

The orchestrator generates handoffs mechanically:

1. Read the phase's `required_context` array from `phase-registry.json`.
2. For each variable, look up `context_sources` to find the source type.
3. Resolve:
   - `"source": "config"` -> read from `config.json`
   - `"source": "artifact"` -> read from the specified file path relative to `OUTPUT_DIR`
   - `"source": "sticky"` -> read from `running-summary.md` YAML frontmatter
4. Populate the `## Context` section as sorted `- **KEY**: value` bullet entries.
5. Populate `## Instructions` from the intake selections.
6. On reruns, append `## Prior Attempt Feedback` from the monitor's review.
7. On reruns after RCA, append `## Root Cause Analysis` from the RCA artifact.

## Validation

`scripts/orchestrate/validate_handoff.py` validates every handoff before dispatch:
- Frontmatter must contain `phase` and `phase_index`
- Body must contain `## Context` and `## Instructions`
- Rerun handoffs must also contain `## Prior Attempt Feedback`

## Platform Delivery

- **Claude Code / OpenCode**: The orchestrator writes the handoff to disk. The phase agent reads the file path.
- **Cursor**: The orchestrator reads the handoff content and the phase agent's `agents/phase-NN-*.md` doc, concatenates them, and passes the combined text as the `Task` tool `prompt` parameter. See `protocols/platform-dispatch.md` for assembly details.
