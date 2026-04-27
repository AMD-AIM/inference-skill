# Install Inference Optimize Skill

This skill works in **Claude Code**, **OpenCode**, **Cursor**, and **Codex**.

## One-command install

```bash
git clone https://github.com/AMD-AIM/inference-skill.git
cd inference-skill
bash install.sh
```

This installs the skill to four locations:

| Target | Path | Purpose |
|--------|------|---------|
| Claude Code / OpenCode | `~/.claude/skills/inference-optimize/` | Skill files (copy or symlink) |
| Codex | `${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize/` | Codex skill files (copy or symlink) |
| Cursor skill | `~/.cursor/skills/inference-optimize/` | Symlink to the above |
| Cursor rule | `~/.cursor/rules/inference-optimize.mdc` | Agent-requested rule for discovery |

## Project-local install

```bash
bash install.sh --project /path/to/project
```

Writes Codex discovery files to `<project>/.codex/skills/` and Cursor discovery files to `<project>/.cursor/skills/` and `<project>/.cursor/rules/`. Claude Code / OpenCode continue to use the user-scoped `~/.claude/skills/` install.

## Linked dev install

```bash
bash install.sh --link
```

Symlinks instead of copying, so edits in the repo checkout are immediately reflected.

## What gets installed

- `SKILL.md`, `INTAKE.md`, `RUNTIME.md`, `EXAMPLES.md`, `INSTALL.md`, `LICENSE`
- `orchestrator/` -- `ORCHESTRATOR.md`, `phase-registry.json`, `monitor.md`
- `agents/` -- 10 phase agent docs + `coding-agent.md` + `analysis-agent.md`
- `protocols/` -- communication schemas (handoff format, result schema, etc.)
- `phases/README.md` -- optional archive pointer to the canonical `agents/phase-NN-*.md` docs
- `templates/` -- report templates and schemas
- `scripts/{env,container,profiling,optimize,plugin,report}/` -- ~30 helper scripts organized by category
- `tests/E2E_TEST.md` + `tests/e2e_optimize_test.py` -- E2E runbook and validator
- `resources/TraceLens-internal.tar.gz` -- required bundled TraceLens fallback asset

## Reinstall / upgrade

If an older install exists, the installer moves it to a timestamped backup under `.skill-backups/` before replacing it. Repeated installs are idempotent.

## Verify

```bash
ls ~/.claude/skills/inference-optimize/SKILL.md
ls "${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize/SKILL.md"
ls "${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize/agents/openai.yaml"
ls ~/.cursor/skills/inference-optimize/SKILL.md
ls ~/.cursor/rules/inference-optimize.mdc
ls ~/.claude/skills/inference-optimize/resources/TraceLens-internal.tar.gz
ls "${CODEX_HOME:-$HOME/.codex}/skills/inference-optimize/resources/TraceLens-internal.tar.gz"
ls ~/.cursor/skills/inference-optimize/resources/TraceLens-internal.tar.gz
python3 ~/.claude/skills/inference-optimize/tests/e2e_optimize_test.py --help
```

If you installed before E2E packaging was added, rerun `bash install.sh` to upgrade the installed skill payload.

## Discovery

- **Claude Code**: discovers skills from `~/.claude/skills/` and project `.claude/skills/`.
- **OpenCode**: discovers the same Claude-compatible skill paths.
- **Codex**: discovers skills from `$CODEX_HOME/skills/` (default `~/.codex/skills/`). `--project` also writes `<project>/.codex/skills/` for explicit project-local use.
- **Cursor**: discovers skills from `~/.cursor/skills/` (native skill) AND from `.cursor/rules/*.mdc` (agent-requested rule). Both are installed by `install.sh`.
