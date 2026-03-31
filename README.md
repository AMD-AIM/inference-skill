# inference-skill

Standalone distribution repo for `inferencex-optimize` and `vllm-optimize` skills.

This repo packages GPU inference benchmarking and profiling workflows as reusable skills that can be installed once and used from:

- `Claude Code`
- `OpenCode`
- `Cursor`

Claude Code and OpenCode discover skills from Claude-compatible install locations. Cursor uses a generated `.mdc` rule. One `./install.sh` run sets up all three.

## Two Skills

### inferencex-optimize

Full InferenceX benchmark and profiling workflow including:
- Docker container setup
- Sweep filtering and configuration
- Benchmark execution
- Torch profiler trace collection
- TraceLens analysis
- Report generation

### vllm-optimize

Standalone vLLM benchmark and profiling workflow:
- Automated vLLM server startup
- Concurrency sweep benchmark
- Torch profiler trace capture
- GPU kernel analysis with proper filtering
- Works in containerized environments
- Supports AMD MI355X/MI300X and NVIDIA GPUs

## Guide

For verified OpenCode and Cursor usage, see [GUIDE.md](GUIDE.md).

## Intended UX

### inferencex-optimize

```text
Use inferencex-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

### vllm-optimize

```text
Use vllm-optimize skill for Qwen/Qwen3.5-35B-A3B
```

After either prompt, the agent should drive a short guided setup:
- first ask exactly three high-level question groups: `Run plan`, `Output`, and `GPUs`
- ask those questions together as one grouped form, not one-by-one
- then do lightweight discovery before asking `tp`, `seq-len`, and `conc`
- offer a smoke fast path with recommended defaults or per-filter review
- emit visible status updates between each stage so the user knows what is happening
- summarize the plan
- start work

## Repo layout

```text
inference-skill/
  install.sh
  LICENSE
  skills/
    inferencex-optimize/
      SKILL.md
      INTAKE.md
      RUNTIME.md
      EXAMPLES.md
      INSTALL.md
      LICENSE
      phases/
      templates/
      scripts/
      resources/
    vllm-optimize/
      SKILL.md
      INTAKE.md
      RUNTIME.md
      README.md
      phases/
      scripts/
```

## Install

Clone the repo and install globally:

```bash
git clone https://github.com/AMD-AIM/inference-skill.git
cd inference-skill
./install.sh
```

Install into a specific project:

```bash
./install.sh --project /path/to/project
```

Create a linked install for local development:

```bash
./install.sh --project /path/to/project --link
```

## Install targets

Global install writes to:

```text
~/.claude/skills/inferencex-optimize       # skill files (Claude Code + OpenCode)
~/.claude/skills/vllm-optimize             # skill files (Claude Code + OpenCode)
~/.cursor/skills/inferencex-optimize       # symlink (Cursor native skill)
~/.cursor/skills/vllm-optimize             # symlink (Cursor native skill)
~/.cursor/rules/inferencex-optimize.mdc    # Cursor agent-requested rule
~/.cursor/rules/vllm-optimize.mdc          # Cursor agent-requested rule
```

Project install writes to the same three locations under the project directory.

## Source of truth

The standalone skills live under `skills/inferencex-optimize/` and `skills/vllm-optimize/`.

Each directory is the source of truth for its respective skill:
- `SKILL.md` - skill definition and metadata
- guided intake flow
- runtime defaults and bootstrap rules
- interaction examples
- phase instructions
- helper scripts

## Development workflow

1. Edit files under `skills/inferencex-optimize/` or `skills/vllm-optimize/`.
2. Reinstall with `./install.sh` or use `--link` during development.
3. Validate the installed result from the destination skill directory.