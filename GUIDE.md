# Guide

This guide covers the verified ways to use `inferencex-optimize` and `vllm-optimize` from `OpenCode` after installing this repo.

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

## 2. vllm-optimize - Default user prompt

For standalone vLLM benchmarking:

```text
Use vllm-optimize skill for Qwen/Qwen3.5-35B-A3B
```

From there, the agent should:

- treat the model name as the starting point
- first ask exactly three high-level question groups: `Run plan`, `Output`, and `GPUs`
- ask those three groups in one batched form
- then do lightweight discovery before asking sequence length and concurrency
- offer the smoke fast path: `Use recommended smoke defaults`, `Review each filter`, or `Use full discovered sweep`
- emit short status updates between each stage
- summarize the plan
- begin execution

## 3. Verify OpenCode can discover the skills

Run:

```bash
opencode debug skill
```

Expected output includes both skills:

```json
[
  {
    "name": "inferencex-optimize",
    "description": "Run the InferenceX benchmark and profiling workflow...",
    "location": "/home/you/.claude/skills/inferencex-optimize/SKILL.md"
  },
  {
    "name": "vllm-optimize",
    "description": "Run vLLM benchmark and profiling workflow in containerized environments...",
    "location": "/home/you/.claude/skills/vllm-optimize/SKILL.md"
  }
]
```

## 4. Minimal one-shot OpenCode test

### For inferencex-optimize

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

### For vllm-optimize

```bash
cd /tmp/inference-skill-opencode-project
opencode run -m amd-anthropic/claude-opus-4-6 \
  "Use the skill tool to load the vllm-optimize skill. Then reply with only the four phase names as a comma-separated list. Do not use any other tools."
```

Expected response:

```text
vllm-setup, benchmark, profiling, analysis
```

## 5. Guided interactive OpenCode usage

Inside any project where the skill is discoverable:

```bash
opencode
```

Then use a prompt like:

```text
# For InferenceX
Use inferencex-optimize skill for qwen3.5-bf16-mi355x-sglang.

# For vLLM
Use vllm-optimize skill for Qwen/Qwen3.5-35B-A3B
```

The agent should then ask a small number of choice-based setup questions before starting work.

## 6. Non-interactive shell / CI note

In some non-TTY shells, `opencode run` may not emit visible formatted output unless it has a pseudo-terminal.

If you are validating from CI or another non-interactive shell, wrap the command with `script`:

```bash
script -q -c 'cd /tmp/inference-skill-opencode-project && opencode run -m amd-anthropic/claude-opus-4-6 "Use the skill tool to load the inferencex-optimize skill. Then reply with only the six phase names as a comma-separated list."' /tmp/inference-skill-opencode-run.log
```

This repo was verified with that form.

## 7. Cursor usage

After `./install.sh`, Cursor rules are installed at:

```bash
~/.cursor/rules/inferencex-optimize.mdc
~/.cursor/rules/vllm-optimize.mdc
```

### Verify the rules are installed

```bash
ls ~/.cursor/rules/inferencex-optimize.mdc
ls ~/.cursor/rules/vllm-optimize.mdc
```

### Usage in Cursor

Each rule is **Agent Requested** — Cursor's AI loads it automatically when you ask about the respective workflow in Composer (agent mode).

Example prompts in Cursor Composer:

```text
# For InferenceX
Use inferencex-optimize for qwen3.5-bf16-mi355x-sglang.

# For vLLM
Use vllm-optimize for Qwen/Qwen3.5-35B-A3B
```

Cursor's agent will recognize each request, pull in the rule, and follow the guided setup flow from `SKILL.md`.

## 8. Duplicate install note

If you install either skill both globally and project-locally with the same skill name, OpenCode may resolve the global copy when listing skills.

To avoid confusion while debugging:

- keep only one active install of each skill, or
- use `HOME=$(mktemp -d)` for an isolated verification environment

Example isolated verification:

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" ./install.sh
HOME="$TMP_HOME" opencode debug skill
```