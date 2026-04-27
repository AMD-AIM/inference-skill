# Platform Dispatch Protocol

The deterministic runner (`scripts/orchestrate/runner.py`) is the canonical control plane for all platforms. It accepts three optional callbacks that differ per platform:

| Callback | Signature | Responsibility |
|----------|-----------|---------------|
| `dispatch_fn` | `(phase_key, handoff_path) -> verdict_dict` | Spawn phase agent, return verdict |
| `monitor_fn` | `(phase_key, result_path, summary_path, checks) -> verdict_dict` | Spawn monitor agent, return verdict |
| `rca_fn` | `(phase_key, manifest_dict) -> rca_dict` | Spawn RCA/analysis agent, return analysis |

When a callback is `None`, the runner applies its default only in shadow/test scenarios. `dispatch_fn=None` is still valid in shadow mode (simulated PASS path).

`monitor_fn=None` is only valid for shadow/unit-test paths. Real runs must wire `monitor_fn` so verdicts come from a separate monitor agent. In non-shadow runs with `dispatch_fn` wired, the runner fails closed with `error_type: monitor_error` when `monitor_fn` is missing.

`rca_fn=None` is also shadow/test-only when the resolved phase list includes critical phases with non-null `rca_artifact`. In non-shadow runs with `dispatch_fn` wired, the runner fails closed at startup with `error_type: rca_error` when such RCA-required phases are present and `rca_fn` is missing.

The internal `rca_skipped[phase_key] = "rca_fn_not_wired"` marker remains as a defensive fallback for shadow/test paths and should not appear in properly wired real runs.

## Cursor

### Agent Type Mapping

| Agent Role | `subagent_type` | `model` | Prompt Shape |
|---|---|---|---|
| Phase-orchestrator | `generalPurpose` | inherit | Compact prompt with paths to `orchestrator/PHASE-ORCHESTRATOR.md` + `phase-registry.json` + prior `monitor/phase-{NN-1}-orchestration-summary.md` (or null) + scalars: `phase_key`, `phase_index`, `OUTPUT_DIR`, `SCRIPTS_DIR`. Returns the path to `monitor/phase-{NN}-orchestration-summary.md`. |
| Phase agent | `generalPurpose` | inherit | Compact prompt with file-path references (`agents/phase-NN-*.md` + `handoff/to-phase-NN.md`) |
| Monitor agent | `generalPurpose` | `fast` | Compact prompt with paths to `monitor.md`, result, summary, optional context JSON |
| Analysis agent | `generalPurpose` | inherit | Compact prompt with `analysis-agent.md` path + manifest path + bounded context paths |
| RCA agent | `generalPurpose` | inherit | Compact prompt with `rca-agent.md` path + manifest path + bounded context paths; **prepend `ultrathink`** keyword |
| Coding agent | `generalPurpose` | inherit | Task description from parent phase agent |
| Container operations | `shell` | `fast` | Shell commands only |

`inherit` = omit the `model` parameter so the subagent runs on the parent session's model.

### Prompt Assembly

**Phase agents**: Prefer path-based delivery. Pass a compact instruction prompt that references `agents/phase-NN-*.md` and `handoff/to-phase-NN.md` paths and asks the subagent to read them. Only inline snippets when absolutely necessary, and always cap the final prompt with `max_context_lines`.

**Monitor agents**: Pass file-path references (`orchestrator/monitor.md`, `monitor/running-summary.md`, `agent-results/phase-NN-result.md`, optional `monitor/phase-NN-context.json`) instead of concatenating full file contents in the parent orchestrator.

**Analysis agents** (routine in-phase data analysis): Pass `agents/analysis-agent.md` and analyzer manifest paths. If the manifest enumerates many context files, pass only a bounded subset directly and let the subagent pull additional files as needed.

**RCA agents** (orchestrator-level root cause on monitor FAIL): Pass `agents/rca-agent.md` and analyzer manifest paths; avoid pre-inlining entire `analysis_context` trees. **Prepend the literal keyword `ultrathink` as the first token of the assembled prompt** — Cursor uses this keyword to enable the maximum reasoning-effort budget.

### Question Tool

Use `AskQuestion` tool for guided setup forms (SKILL.md intake flow).

## Claude Code

### Agent Type Mapping

| Agent Role | Tool | Prompt Shape |
|---|---|---|
| Phase-orchestrator | `Agent` | Path to `orchestrator/PHASE-ORCHESTRATOR.md` + `phase-registry.json` + prior orchestration-summary path (or null) + scalars: `phase_key`, `phase_index`, `OUTPUT_DIR`, `SCRIPTS_DIR`. The subagent runs the inner dispatch loop in a fresh context and returns the path to its summary file. |
| Phase agent | `Agent` | Path to `handoff/to-phase-NN.md` (agent reads from disk) |
| Monitor agent | `Agent` | Paths to monitor docs + result + summary |
| Analysis agent | `Agent` | Path to `analysis-agent.md` + analyzer manifest file |
| RCA agent | `Agent` | Path to `rca-agent.md` + analyzer manifest file. Reasoning effort is controlled by the agent frontmatter without a fixed token cap. |
| Coding agent | `Agent` | Spawned by parent phase agent, not orchestrator |
| Container operations | `bash` | Direct shell commands |

The runner-side context compaction changes do **not** alter the Claude Code dispatch contract: phase/monitor/RCA agents still receive file paths, read their own docs from disk, and benefit from smaller handoff/manifests without any Cursor-specific prompt assembly.

### Question Tool

Use `question` tool for guided setup forms.

## OpenCode

Same as Claude Code. OpenCode discovers skills from `~/.claude/skills/` and uses the `Agent` tool with path-based handoffs. The new context-budget changes therefore apply through compact handoff/manifests, not through a different prompt shape.

### Question Tool

Use `question` tool. For non-interactive runs (`opencode run`), set permission `"question": "allow"` in `.opencode/opencode.jsonc`.

## Codex

Codex discovers skills from `$CODEX_HOME/skills/<skill-name>` (default `~/.codex/skills/<skill-name>`). The installer also supports an explicit project-local target at `<project>/.codex/skills/<skill-name>` when `--project` is supplied.

Use the same file-based handoff contract as Claude Code / OpenCode: pass paths to `orchestrator/PHASE-ORCHESTRATOR.md`, `phase-registry.json`, handoff files, monitor inputs, and RCA manifests instead of inlining large documents. When a Codex runtime provides a delegation tool, use it with these path-only prompts. If no dedicated question tool is available, ask concise numbered choices in chat.

## Context Budget

All platforms share the same `max_context_lines` control from `phase-registry.json`. The runner's `truncate_context()` applies deterministic truncation only when this value is greater than zero. The current default is `8000` lines. `phase-registry.json` also defines `context_value_char_limit`, `context_keys_preview`, `context_items_preview`, and `cursor_agent_doc_max_lines` for deterministic context compaction. On Cursor, these caps apply to any prompt content assembled by the parent orchestrator. On Claude Code / OpenCode / Codex, the same compaction applies to handoff rendering and any optional inline snippets, while agent docs and manifests continue to be read from disk by path.
