# Guide

This guide covers the verified ways to use `inferencex-optimize` from `OpenCode` after installing this repo.

## Prerequisites

- `opencode` is installed and on `PATH`
- the skill is installed with `./install.sh`
- your OpenCode model/provider credentials are already configured

Examples below use `amd-anthropic/claude-opus-4-6`, but you can replace that with your normal OpenCode model.

## 1. Default user prompt

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
    "description": "Run the InferenceX benchmark and profiling workflow for a config key, including Docker setup, sweep filtering, benchmark execution, torch profiler trace collection, TraceLens analysis, and report generation. Use when benchmarking or profiling a model with InferenceX, rerunning a specific phase, or regenerating benchmark or profiling reports.",
    "location": "/home/you/.claude/skills/inferencex-optimize/SKILL.md"
  }
]
```

## 3. Minimal one-shot OpenCode test

For scripted or headless `opencode run`, give the project an explicit local permission config so the `skill` tool does not stop for approval.

Create a scratch project:

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
```

Then run:

```bash
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

Or:

```text
Use inferencex-optimize skill for qwen3.5-bf16-mi355x-sglang and help me quickly smoke test it.
```

The agent should then ask a small number of choice-based setup questions before starting work.

## 5. Non-interactive shell / CI note

In some non-TTY shells, `opencode run` may not emit visible formatted output unless it has a pseudo-terminal.

If you are validating from CI or another non-interactive shell, wrap the command with `script`:

```bash
script -q -c 'cd /tmp/inference-skill-opencode-project && opencode run -m amd-anthropic/claude-opus-4-6 "Use the skill tool to load the inferencex-optimize skill. Then reply with only the six phase names as a comma-separated list. Do not use any other tools."' /tmp/inference-skill-opencode-run.log
```

This repo was verified with that form, and the captured output included:

```text
Loaded skill: inferencex-optimize
env, config, benchmark, benchmark-analyze, profile, profile-analyze
```

## 6. Duplicate install note

If you install `inferencex-optimize` both globally and project-locally with the same skill name, OpenCode may resolve the global copy when listing skills.

To avoid confusion while debugging:

- keep only one active install of `inferencex-optimize`, or
- use `HOME=$(mktemp -d)` for an isolated verification environment

Example isolated verification:

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" ./install.sh
HOME="$TMP_HOME" opencode debug skill
```

## 7. Cursor usage

After `./install.sh`, a Cursor rule is installed at:

```bash
~/.cursor/rules/inferencex-optimize.mdc
```

### Verify the rule is installed

```bash
ls ~/.cursor/rules/inferencex-optimize.mdc
head -6 ~/.cursor/rules/inferencex-optimize.mdc
```

Expected frontmatter:

```yaml
---
description: >-
  InferenceX benchmark and profiling workflow skill. Use this rule when...
alwaysApply: false
---
```

### Verify path rewrites

All file references in the rule should be absolute paths, not relative:

```bash
grep -E "\]\(.*\.md\)" ~/.cursor/rules/inferencex-optimize.mdc | head -5
```

Every match should contain the full installed skill path (e.g., `~/.claude/skills/inferencex-optimize/INTAKE.md`), not bare filenames.

### Usage in Cursor

The rule is **Agent Requested** — Cursor's AI loads it automatically when you ask about InferenceX benchmarking, profiling, or optimization in Composer (agent mode).

Example prompt in Cursor Composer:

```text
Use inferencex-optimize for qwen3.5-bf16-mi355x-sglang.
```

Cursor's agent will recognize the request, pull in the rule, and follow the guided setup flow from `SKILL.md`.

The installed rule embeds absolute paths to `INTAKE.md`, `RUNTIME.md`, and the phase docs at `~/.claude/skills/inferencex-optimize/`. The agent reads them on demand as the workflow progresses.

### Note on the pre-existing `inferencex-optimizer.mdc`

If you previously had a manually written rule at `~/.cursor/rules/inferencex-optimizer.mdc` (note the `-r` suffix), the installer does not touch it. You may want to disable or delete that older rule to avoid duplicate context. The new `inferencex-optimize.mdc` covers the full workflow including the optimize phases (6-9) that the older rule omitted.
