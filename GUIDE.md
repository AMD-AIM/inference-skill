# Guide

This guide covers verified usage of `inference-optimize` from OpenCode, Cursor, and Codex after running `./install.sh`.

## Prerequisites

- `opencode` is installed and available on `PATH`
- skill installed with `./install.sh`
- provider/model credentials are configured in OpenCode

## 1) Default prompt

```text
Use inference-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

The detailed guided setup contract (question groups, discovery order, confirmation flow) is defined in:

- `skills/inference-optimize/SKILL.md`
- `skills/inference-optimize/INTAKE.md`

## 2) Verify OpenCode discovery

```bash
opencode debug skill
```

Expected entry includes:

```json
[
  {
    "name": "inference-optimize",
    "location": "/home/you/.claude/skills/inference-optimize/SKILL.md"
  }
]
```

## 3) Minimal one-shot OpenCode check

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
  "Use the skill tool to load the inference-optimize skill. Then reply with exactly: loaded-inference-optimize."
```

Expected response:

```text
loaded-inference-optimize
```

## 4) Guided interactive OpenCode usage

```bash
opencode
```

Then prompt:

```text
Use inference-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

## 5) Non-interactive shell / CI note

In non-TTY environments, OpenCode output may be suppressed. Use a pseudo-terminal:

```bash
script -q -c 'cd /tmp/inference-skill-opencode-project && opencode run -m amd-anthropic/claude-opus-4-6 "Use the skill tool to load the inference-optimize skill. Then reply with exactly: loaded-inference-optimize."' /tmp/inference-skill-opencode-run.log
```

## 6) Cursor usage

After `./install.sh`, the generated rule path is:

```bash
~/.cursor/rules/inference-optimize.mdc
```

Verify:

```bash
ls ~/.cursor/rules/inference-optimize.mdc
```

Prompt example in Cursor Composer:

```text
Use inference-optimize for qwen3.5-bf16-mi355x-sglang.
```

## 7) Codex usage

After `./install.sh`, the Codex skill path is:

```bash
${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize
```

Verify:

```bash
ls "${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize/SKILL.md"
ls "${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize/agents/openai.yaml"
```

Prompt example in Codex:

```text
Use $inference-optimize for qwen3.5-bf16-mi355x-sglang.
```

Restart Codex after installing so new skills are picked up.

## 8) Architecture references

For multi-agent dispatch and protocol details:

- `skills/inference-optimize/orchestrator/ORCHESTRATOR.md`
- `skills/inference-optimize/orchestrator/PHASE-ORCHESTRATOR.md`
- `skills/inference-optimize/protocols/platform-dispatch.md`

## 9) Duplicate install note

If both global and project-local installs exist, OpenCode may resolve the global one first. For isolated verification:

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" ./install.sh
HOME="$TMP_HOME" opencode debug skill
```
