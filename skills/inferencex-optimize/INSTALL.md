# Install InferenceX Optimize Skill

This skill is designed to work in both:

- `Claude Code`
- `OpenCode`

The shared install target is `~/.claude/skills/inferencex-optimize/`, which both tools can discover.

## One-command install

From the `inference-skill` repo root:

```bash
./install.sh
```

That command installs a self-contained copy of this skill into:

```bash
~/.claude/skills/inferencex-optimize
```

## Project-local install

To install the skill into a specific project instead of your home directory:

```bash
./install.sh --project /path/to/project
```

That writes to:

```bash
/path/to/project/.claude/skills/inferencex-optimize
```

## Linked dev install

If you want the target to track this repo checkout during development:

```bash
./install.sh --project /path/to/project --link
```

## What gets installed

The installer creates a standalone skill package with:

- `SKILL.md`
- `INTAKE.md`
- `RUNTIME.md`
- `EXAMPLES.md`
- `INSTALL.md`
- `LICENSE`
- `phases/*.md`
- `templates/agent-config.md`
- `scripts/*.py`
- `resources/TraceLens-internal.tar.gz` when present

## Reinstall / upgrade behavior

- If an older `inferencex-optimize` skill exists at the destination, the installer moves it to a timestamped backup under `.skill-backups/` before replacing it.
- The installer is idempotent for repeated upgrades from the same repo checkout.

## Discovery notes

- Claude Code discovers skills from `~/.claude/skills/` and project `.claude/skills/`.
- OpenCode also discovers Claude-compatible skills from those same locations.

## Verify install

After install, confirm the target exists:

```bash
ls ~/.claude/skills/inferencex-optimize
```

For OpenCode, the skill will appear through the native `skill` tool.
For Claude Code, the skill is available as a normal installed skill and can be loaded when relevant.

For verified OpenCode usage examples, see [`GUIDE.md`](../../GUIDE.md).
