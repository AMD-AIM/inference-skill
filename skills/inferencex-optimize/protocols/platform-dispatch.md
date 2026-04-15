# Platform Dispatch Protocol

The deterministic runner (`scripts/orchestrate/runner.py`) is the canonical control plane for all platforms. It accepts three optional callbacks that differ per platform:

| Callback | Signature | Responsibility |
|----------|-----------|---------------|
| `dispatch_fn` | `(phase_key, handoff_path) -> verdict_dict` | Spawn phase agent, return verdict |
| `monitor_fn` | `(phase_key, result_path, summary_path, checks) -> verdict_dict` | Spawn monitor agent, return verdict |
| `rca_fn` | `(phase_key, manifest_dict) -> rca_dict` | Spawn RCA/analysis agent, return analysis |

When a callback is `None`, the runner applies its default: `dispatch_fn` returns PASS (shadow mode), `monitor_fn` passes through the dispatch verdict, `rca_fn` is skipped.

## Cursor

### Agent Type Mapping

| Agent Role | `subagent_type` | `model` | Prompt Shape |
|---|---|---|---|
| Phase agent | `generalPurpose` | inherit | Agent doc + handoff content inlined |
| Monitor agent | `generalPurpose` | `fast` | `monitor.md` + result + running summary inlined |
| Analysis/RCA agent | `generalPurpose` | inherit | Analyzer manifest + context file contents inlined |
| Coding agent | `generalPurpose` | inherit | Task description from parent phase agent |
| Container operations | `shell` | `fast` | Shell commands only |

`inherit` = omit the `model` parameter so the subagent runs on the parent session's model.

### Prompt Assembly

**Phase agents**: Read `agents/phase-NN-*.md` and `handoff/to-phase-NN.md`, concatenate with `---` separator, truncate to `max_context_lines` (500). Pass as `Task` tool `prompt`.

**Monitor agents**: Read `orchestrator/monitor.md`, `monitor/running-summary.md`, `agent-results/phase-NN-result.md`, and optionally `monitor/phase-NN-context.json`. Concatenate. Pass as `Task` tool `prompt` with `model: "fast"`.

**RCA agents**: Read `agents/analysis-agent.md` and the manifest's `analysis_context` file contents. Concatenate. Pass as `Task` tool `prompt`.

### Question Tool

Use `AskQuestion` tool for guided setup forms (SKILL.md intake flow).

## Claude Code

### Agent Type Mapping

| Agent Role | Tool | Prompt Shape |
|---|---|---|
| Phase agent | `Agent` | Path to `handoff/to-phase-NN.md` (agent reads from disk) |
| Monitor agent | `Agent` | Paths to monitor docs + result + summary |
| Analysis/RCA agent | `Agent` | Path to analyzer manifest file |
| Coding agent | `Agent` | Spawned by parent phase agent, not orchestrator |
| Container operations | `bash` | Direct shell commands |

### Question Tool

Use `question` tool for guided setup forms.

## OpenCode

Same as Claude Code. OpenCode discovers skills from `~/.claude/skills/` and uses the `Agent` tool with path-based handoffs.

### Question Tool

Use `question` tool. For non-interactive runs (`opencode run`), set permission `"question": "allow"` in `.opencode/opencode.jsonc`.

## Context Budget

All platforms share the same budget: `max_context_lines` (default 500) from `phase-registry.json`. The runner's `truncate_context()` enforces this deterministically. On Cursor, the budget applies to the assembled prompt (agent doc + handoff combined). On Claude Code / OpenCode, the budget applies to the handoff document (agents read their own docs separately from disk).
