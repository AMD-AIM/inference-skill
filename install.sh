#!/usr/bin/env bash
set -euo pipefail

SKILL_NAMES=("inferencex-optimize" "vllm-optimize")
MODE="copy"

usage() {
  cat <<'EOF'
Install the InferenceX and vLLM Optimize skills for Claude Code, OpenCode, and Cursor.

Usage:
  ./install.sh
  ./install.sh --project /path/to/project
  ./install.sh --dest /custom/skill/dir
  ./install.sh --project /path/to/project --link

Options:
  --project PATH   Install into PATH/.claude/skills/SKILL_NAME
  --dest PATH      Install into an explicit skill directory path
  --link           Symlink instead of copying files
  --copy           Copy files explicitly (default)
  -h, --help       Show this help text
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      [[ $# -ge 2 ]] || { echo "Missing value for --project" >&2; exit 1; }
      CURSOR_PROJECT="$2"
      shift 2
      ;;
    --dest)
      [[ $# -ge 2 ]] || { echo "Missing value for --dest" >&2; exit 1; }
      shift 2
      echo "--dest option is deprecated. Use --project instead." >&2
      exit 1
      ;;
    --link)
      MODE="link"
      shift
      ;;
    --copy)
      MODE="copy"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "Required file not found: $path" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || {
    echo "Required directory not found: $path" >&2
    exit 1
  }
}

install_skill() {
  local SKILL_NAME="$1"
  local SOURCE_DIR="$REPO_ROOT/skills/$SKILL_NAME"
  local DEST_DIR="${HOME}/.claude/skills/${SKILL_NAME}"

  require_dir "$SOURCE_DIR"
  require_file "$SOURCE_DIR/SKILL.md"

  # vllm-optimize uses README.md, inferencex-optimize uses INSTALL.md
  if [[ "$SKILL_NAME" == "inferencex-optimize" ]]; then
    require_file "$SOURCE_DIR/INSTALL.md"
    require_file "$SOURCE_DIR/LICENSE"
    require_dir "$SOURCE_DIR/phases"
    require_dir "$SOURCE_DIR/templates"
    require_dir "$SOURCE_DIR/scripts"
    require_dir "$SOURCE_DIR/tests"
    require_file "$SOURCE_DIR/tests/E2E_TEST.md"
    require_file "$SOURCE_DIR/tests/e2e_optimize_test.py"
  else
    require_file "$SOURCE_DIR/README.md"
    require_file "$SOURCE_DIR/RUNTIME.md"
    require_dir "$SOURCE_DIR/phases"
  fi

  mkdir -p "$(dirname "$DEST_DIR")"

  BACKUP_ROOT="$(dirname "$DEST_DIR")/.skill-backups/$SKILL_NAME"

  if [[ -e "$DEST_DIR" || -L "$DEST_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_ROOT"
    BACKUP_PATH="${BACKUP_ROOT}/${TIMESTAMP}"
    mv "$DEST_DIR" "$BACKUP_PATH"
    echo "Backed up existing install to: $BACKUP_PATH"
  fi

  if [[ "$MODE" == "link" ]]; then
    ln -s "$SOURCE_DIR" "$DEST_DIR"
    echo "  Linked: $SKILL_NAME -> $SOURCE_DIR"
  else
    mkdir -p "$DEST_DIR"
    cp -R "$SOURCE_DIR"/. "$DEST_DIR"/
    echo "  Copied: $SKILL_NAME -> $DEST_DIR"
  fi

  # Cursor skill symlink
  CURSOR_BASE="${CURSOR_PROJECT:-$HOME}"
  CURSOR_SKILL_DIR="${CURSOR_BASE}/.cursor/skills/${SKILL_NAME}"
  mkdir -p "$(dirname "$CURSOR_SKILL_DIR")"
  ln -sfn "$DEST_DIR" "$CURSOR_SKILL_DIR"

  # Cursor rule generation
  CURSOR_RULE_DEST="${CURSOR_BASE}/.cursor/rules/${SKILL_NAME}.mdc"
  
  # Extract SKILL.md body (everything after the closing --- of frontmatter)
  SKILL_BODY="$(awk '/^---/{if(++c==2){found=1;next}} found' "${DEST_DIR}/SKILL.md")"

  # Rewrite relative markdown links to absolute paths into the installed skill dir
  SKILL_BODY_ABS="$(printf '%s\n' "$SKILL_BODY" \
    | sed "s|](INTAKE.md)|](${DEST_DIR}/INTAKE.md)|g" \
    | sed "s|](RUNTIME.md)|](${DEST_DIR}/RUNTIME.md)|g" \
    | sed "s|](README.md)|](${DEST_DIR}/README.md)|g" \
    | sed "s|](EXAMPLES.md)|](${DEST_DIR}/EXAMPLES.md)|g" \
    | sed "s|](tests/E2E_TEST.md)|](${DEST_DIR}/tests/E2E_TEST.md)|g" \
    | sed "s|](tests/e2e_optimize_test.py)|](${DEST_DIR}/tests/e2e_optimize_test.py)|g" \
    | sed "s|](phases/|](${DEST_DIR}/phases/|g")"

  # Generate description based on skill name
  local DESC=""
  if [[ "$SKILL_NAME" == "inferencex-optimize" ]]; then
    DESC="InferenceX benchmark and profiling workflow skill. Use this rule when the user asks to benchmark, profile, or optimize a model with InferenceX, names a config key, or asks to run any phase of the InferenceX pipeline."
  else
    DESC="vLLM benchmark and profiling workflow skill. Use this rule when the user asks to benchmark or profile a model with vLLM, run vLLM inference optimization, or analyze GPU kernel performance."
  fi

  MDC_CONTENT="---
description: >-
  ${DESC}
alwaysApply: false
---
${SKILL_BODY_ABS}"

  mkdir -p "$(dirname "$CURSOR_RULE_DEST")"
  printf '%s\n' "$MDC_CONTENT" > "$CURSOR_RULE_DEST"
  echo "  Rule:   $CURSOR_RULE_DEST"
}

# Install each skill
echo "Installing skills..."
for SKILL_NAME in "${SKILL_NAMES[@]}"; do
  echo ""
  echo "=== $SKILL_NAME ==="
  install_skill "$SKILL_NAME"
done

echo ""
echo "============================================"
echo "  Installation Complete"
echo "============================================"
echo "Installed skills: ${SKILL_NAMES[*]}"
echo ""
echo "Compatible with: Claude Code, OpenCode, Cursor"