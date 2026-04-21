# inference-skill

Standalone distribution repo for the `inference-optimize` skill.

This repo packages GPU inference benchmarking, profiling, and kernel optimization workflows as a reusable skill that can be installed once and used from:

- `Claude Code`
- `OpenCode`
- `Cursor`

Claude Code and OpenCode discover skills from Claude-compatible install locations. Cursor uses a generated `.mdc` rule. One `./install.sh` run sets up all three.

## What this repo ships

`inference-skill` packages one canonical skill payload:

- `skills/inference-optimize/` (skill source of truth)
- `install.sh` (installer for Claude Code, OpenCode, and Cursor)
- control-plane tests and E2E validator assets

The implementation itself remains benchmark-framework agnostic while targeting the external [InferenceX repository](https://github.com/SemiAnalysisAI/InferenceX) during runtime execution.

## Install

Global install:

```bash
git clone https://github.com/AMD-AIM/inference-skill.git
cd inference-skill
./install.sh
```

Project-local install:

```bash
./install.sh --project /path/to/project
```

Linked install for local development:

```bash
./install.sh --project /path/to/project --link
```

## Install targets

Global install writes to:

```text
~/.claude/skills/inference-optimize       # skill files (Claude Code + OpenCode)
~/.cursor/skills/inference-optimize       # symlink (Cursor native skill)
~/.cursor/rules/inference-optimize.mdc    # Cursor agent-requested rule
```

Project install writes to the same three locations under the project directory.

## Quick verification

```bash
./install.sh --verify
opencode debug skill
ls ~/.cursor/rules/inference-optimize.mdc
```

Expected discovery name: `inference-optimize`.

## Usage prompt

```text
Use inference-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

For verified OpenCode and Cursor flows, see [GUIDE.md](GUIDE.md).

## Source-of-truth map

All operational contracts live under `skills/inference-optimize/`:

- `SKILL.md`: entrypoint behavior, file read order, mode/config references
- `INTAKE.md`: guided setup question flow
- `RUNTIME.md`: bootstrap, required inputs, runtime guardrails
- `orchestrator/ORCHESTRATOR.md`: outer dispatcher contract
- `orchestrator/PHASE-ORCHESTRATOR.md`: per-phase inner loop contract
- `orchestrator/monitor.md`: monitor-agent behavior
- `orchestrator/phase-registry.json`: phase metadata, dependencies, quality checks
- `protocols/*.md|*.json`: schema and protocol contracts

Repo-root docs (`README.md`, `GUIDE.md`) are install and operator guides, not runtime contract sources.

## Repository layout

```text
inference-skill/
  install.sh
  pyproject.toml
  .github/workflows/control-plane.yml
  skills/
    inference-optimize/
      SKILL.md
      INTAKE.md
      RUNTIME.md
      orchestrator/
      agents/
      protocols/
      scripts/
      templates/
      tests/
      resources/
```

## Testing

Run control-plane tests from repo root:

```bash
pytest -v
```

Run script-only tests:

```bash
pytest skills/inference-optimize/scripts/tests/ -v
```

Run E2E validator against an output directory:

```bash
python3 skills/inference-optimize/tests/e2e_optimize_test.py --output-dir <path>
```

## Development workflow

1. Edit files under `skills/inference-optimize/`.
2. Reinstall with `./install.sh` (or use `--link`).
3. Re-run tests and re-verify installed targets.