# Handoff Document Format

The orchestrator writes `handoff/to-phase-{NN}.md` for each phase agent. This is the phase agent's primary input alongside its own `agents/phase-NN-*.md` doc.

## Format

```markdown
---
phase: {phase_key}
phase_index: {NN}
config_key: "{CONFIG_KEY}"
output_dir: "{OUTPUT_DIR}"
mode: "{MODE}"
---

## Resolved Variables
(populated mechanically from phase-registry.json `required_context` for this phase,
resolved via `context_sources`)

| Variable | Value | Source |
|----------|-------|--------|
| CONFIG_KEY | ... | config |
| OUTPUT_DIR | ... | config |
| GPU_ARCH | ... | artifact |
| ... | ... | ... |

## Prior Phase Outputs
(paths to artifacts from dependency phases, populated from prior phase results)

- phase-00 env_info.json: {path}
- phase-01 sweep_configs.json: {path}
- ...

## Instructions
(phase-specific overrides, filter selections, user preferences that the orchestrator
passes through from the intake)

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
   - `"source": "config"` → read from `config.json`
   - `"source": "artifact"` → read from the specified file path relative to `OUTPUT_DIR`
   - `"source": "sticky"` → read from `running-summary.md` YAML frontmatter
4. Populate the `## Resolved Variables` table.
5. Populate `## Prior Phase Outputs` from the completed phases' result docs.
6. Populate `## Instructions` from the intake selections.
7. On reruns, append `## Prior Attempt Feedback` from the monitor's review.
