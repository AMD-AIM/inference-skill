# Platform Dispatch Protocol

The deterministic runner (`scripts/orchestrate/runner.py`) is the canonical control plane for all platforms. It accepts three optional callbacks that differ per platform:

| Callback | Signature | Responsibility |
|----------|-----------|---------------|
| `dispatch_fn` | `(phase_key, handoff_path) -> verdict_dict` | Spawn phase agent, return verdict |
| `monitor_fn` | `(phase_key, result_path, summary_path, checks) -> verdict_dict` | Spawn monitor agent, return verdict |
| `rca_fn` | `(phase_key, manifest_dict) -> rca_dict` | Spawn RCA/analysis agent, return analysis |

When a callback is `None`, the runner applies its default: `dispatch_fn` returns PASS (shadow mode), `rca_fn` is skipped.

`monitor_fn=None` is only valid for shadow/unit-test paths. Real runs must wire `monitor_fn` so verdicts come from a separate monitor agent. In non-shadow runs with `dispatch_fn` wired, the runner fails closed with `error_type: monitor_error` when `monitor_fn` is missing.

When a phase has a non-null `rca_artifact` in `phase-registry.json` and the host platform supplies `rca_fn=None`, the runner emits a `WARNING`-level log line and records `rca_skipped: true` in the phase's progress entry. RCA is not silently absent: integrators are expected to wire `rca_fn` for any platform that runs the optimize/profile modes end-to-end. Shadow-mode and unit-test runs may legitimately leave `rca_fn=None`; the warning makes the absence auditable.

## Cursor

### Agent Type Mapping

| Agent Role | `subagent_type` | `model` | Prompt Shape |
|---|---|---|---|
| Phase agent | `generalPurpose` | inherit | Agent doc + handoff content inlined |
| Monitor agent | `generalPurpose` | `fast` | `monitor.md` + result + running summary inlined |
| Analysis agent | `generalPurpose` | inherit | `analysis-agent.md` + analyzer manifest + context file contents inlined (routine in-phase data analysis) |
| RCA agent | `generalPurpose` | inherit | `rca-agent.md` + analyzer manifest + context file contents inlined; **prepend the literal keyword `ultrathink`** to the assembled prompt — Cursor does not read agent frontmatter, so this is the only way to enable the high-reasoning-effort budget the RCA agent expects |
| Coding agent | `generalPurpose` | inherit | Task description from parent phase agent |
| Container operations | `shell` | `fast` | Shell commands only |

`inherit` = omit the `model` parameter so the subagent runs on the parent session's model.

### Prompt Assembly

**Phase agents**: Read `agents/phase-NN-*.md` and `handoff/to-phase-NN.md`, concatenate with `---` separator, then apply `max_context_lines` from the registry (default `20000` lines). Pass as `Task` tool `prompt`.

**Monitor agents**: Read `orchestrator/monitor.md`, `monitor/running-summary.md`, `agent-results/phase-NN-result.md`, and optionally `monitor/phase-NN-context.json`. Concatenate. Pass as `Task` tool `prompt` with `model: "fast"`.

**Analysis agents** (routine in-phase data analysis): Read `agents/analysis-agent.md` and the manifest's `analysis_context` file contents. Concatenate. Pass as `Task` tool `prompt`.

**RCA agents** (orchestrator-level root cause on monitor WARN/FAIL): Read `agents/rca-agent.md` and the manifest's `analysis_context` file contents. Concatenate. **Prepend the literal keyword `ultrathink` as the first token of the assembled prompt** — Cursor uses this keyword to enable the maximum reasoning-effort budget. Pass as `Task` tool `prompt`.

### Question Tool

Use `AskQuestion` tool for guided setup forms (SKILL.md intake flow).

## Claude Code

### Agent Type Mapping

| Agent Role | Tool | Prompt Shape |
|---|---|---|
| Phase agent | `Agent` | Path to `handoff/to-phase-NN.md` (agent reads from disk) |
| Monitor agent | `Agent` | Paths to monitor docs + result + summary |
| Analysis agent | `Agent` | Path to `analysis-agent.md` + analyzer manifest file |
| RCA agent | `Agent` | Path to `rca-agent.md` + analyzer manifest file. Reasoning effort is controlled by the agent frontmatter without a fixed token cap. |
| Coding agent | `Agent` | Spawned by parent phase agent, not orchestrator |
| Container operations | `bash` | Direct shell commands |

### Question Tool

Use `question` tool for guided setup forms.

## OpenCode

Same as Claude Code. OpenCode discovers skills from `~/.claude/skills/` and uses the `Agent` tool with path-based handoffs.

### Question Tool

Use `question` tool. For non-interactive runs (`opencode run`), set permission `"question": "allow"` in `.opencode/opencode.jsonc`.

## Context Budget

All platforms share the same `max_context_lines` control from `phase-registry.json`. The runner's `truncate_context()` applies deterministic truncation only when this value is greater than zero. The current default is `20000` lines (high cap for practical near-unlimited context without hard-window spikes). On Cursor, the setting applies to the assembled prompt (agent doc + handoff combined). On Claude Code / OpenCode, it applies to the handoff document (agents read their own docs separately from disk).
