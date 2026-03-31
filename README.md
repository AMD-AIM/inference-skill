# inference-skill

Standalone distribution repo for the `inferencex-optimize` skill.

This repo packages the InferenceX benchmark and profiling workflow as a reusable skill that can be installed once and used from:

- `Claude Code`
- `OpenCode`
- `Cursor`

Claude Code and OpenCode discover skills from Claude-compatible install locations. Cursor uses a generated `.mdc` rule. One `./install.sh` run sets up all three.

## Guide

For verified OpenCode and Cursor usage, see [GUIDE.md](GUIDE.md).

## Intended UX

The intended entry point is simple:

```text
Use inferencex-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

After that, the agent should drive a short guided setup:

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
~/.cursor/skills/inferencex-optimize       # symlink (Cursor native skill)
~/.cursor/rules/inferencex-optimize.mdc    # Cursor agent-requested rule
```

Project install writes to the same three locations under the project directory.

## Source of truth

The standalone skill lives under `skills/inferencex-optimize/`.

That directory is the source of truth for:

- `SKILL.md`
- guided intake flow
- runtime defaults and bootstrap rules
- interaction examples
- phase instructions
- helper scripts
- bundled TraceLens resource

## Development workflow

1. Edit files under `skills/inferencex-optimize/`.
2. Reinstall with `./install.sh` or use `--link` during development.
3. Validate the installed result from the destination skill directory.
