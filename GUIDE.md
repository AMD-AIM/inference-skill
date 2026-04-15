# Guide

This guide covers the verified ways to use `inferencex-optimize` from `OpenCode` and `Cursor` after installing this repo.

## Prerequisites

- `opencode` is installed and on `PATH`
- the skill is installed with `./install.sh`
- your OpenCode model/provider credentials are already configured

Examples below use `amd-anthropic/claude-opus-4-6`, but you can replace that with your normal OpenCode model.

## 1. inferencex-optimize - Default user prompt

The intended user-facing prompt is simple:

```text
Use inferencex-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

From there, the agent should:

- treat the model/config name as the starting point
- first ask exactly three high-level question groups: `Run plan`, `Output`, and `GPUs`
- ask those three groups in one batched form
- then do lightweight discovery before asking TP / sequence length / concurrency
- offer the smoke fast path: `Use recommended smoke defaults`, `Review each filter`, or `Use full discovered sweep`
- emit short status updates before discovery, after discovery, before confirmation, and at phase boundaries
- confirm `tp`, `seq-len`, `conc`, mode, and output behavior
- summarize the plan
- begin execution

## 2. Verify OpenCode can discover the skill

Run:

```bash
opencode debug skill
```

Expected output includes:

```json
[
  {
    "name": "inferencex-optimize",
    "description": "Run the InferenceX benchmark and profiling workflow...",
    "location": "/home/you/.claude/skills/inferencex-optimize/SKILL.md"
  }
]
```

## 3. Minimal one-shot OpenCode test

```bash
mkdir -p /tmp/inference-skill-opencode-project/.opencode
cat > /tmp/inference-skill-opencode-project/.opencode/opencode.jsonc <<'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "skill": "allow",
    "question": "allow",
    "read": "allow"
  }
}
EOF

cd /tmp/inference-skill-opencode-project
opencode run -m amd-anthropic/claude-opus-4-6 \
  "Use the skill tool to load the inferencex-optimize skill. Then reply with only the six phase names as a comma-separated list. Do not use any other tools."
```

Verified expected response:

```text
env, config, benchmark, benchmark-analyze, profile, profile-analyze
```

## 4. Guided interactive OpenCode usage

Inside any project where the skill is discoverable:

```bash
opencode
```

Then use a prompt like:

```text
Use inferencex-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

The agent should then ask a small number of choice-based setup questions before starting work.

## 5. Non-interactive shell / CI note

In some non-TTY shells, `opencode run` may not emit visible formatted output unless it has a pseudo-terminal.

If you are validating from CI or another non-interactive shell, wrap the command with `script`:

```bash
script -q -c 'cd /tmp/inference-skill-opencode-project && opencode run -m amd-anthropic/claude-opus-4-6 "Use the skill tool to load the inferencex-optimize skill. Then reply with only the six phase names as a comma-separated list."' /tmp/inference-skill-opencode-run.log
```

This repo was verified with that form.

## 6. Cursor usage

After `./install.sh`, the Cursor rule is installed at:

```bash
~/.cursor/rules/inferencex-optimize.mdc
```

### Verify the rule is installed

```bash
ls ~/.cursor/rules/inferencex-optimize.mdc
```

### Usage in Cursor

The rule is **Agent Requested** — Cursor's AI loads it automatically when you ask about the workflow in Composer (agent mode).

Example prompt in Cursor Composer:

```text
Use inferencex-optimize for qwen3.5-bf16-mi355x-sglang.
```

Cursor's agent will recognize the request, pull in the rule, and follow the guided setup flow from `SKILL.md`.

## 7. Multi-agent architecture

The `inferencex-optimize` skill now supports a multi-agent orchestration model:

- **Orchestrator** reads `SKILL.md`, `INTAKE.md`, then dispatches phases via `ORCHESTRATOR.md` and `phase-registry.json`
- **Phase agents** are self-contained: each reads its own `agents/phase-NN-*.md` + a handoff document
- **Monitor agent** verifies phase output quality and maintains a rolling summary
- **Coder/Analyzer subagents** handle specialized tasks within phases

The file-based communication protocol (handoffs, results, reviews) is platform-agnostic. Only the agent spawning mechanism differs:
- Claude Code / OpenCode: `Agent` tool with path-based handoffs
- Cursor: `Task` tool with `subagent_type` per the dispatch table (phase/monitor/RCA = `generalPurpose`, containers = `shell`)

See `protocols/platform-dispatch.md` for the full adapter contract and `orchestrator/ORCHESTRATOR.md` for the dispatch loop protocol.

## 8. Duplicate install note

If you install the skill both globally and project-locally, OpenCode may resolve the global copy when listing skills.

To avoid confusion while debugging:

- keep only one active install, or
- use `HOME=$(mktemp -d)` for an isolated verification environment

Example isolated verification:

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" ./install.sh
HOME="$TMP_HOME" opencode debug skill
```