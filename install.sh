#!/usr/bin/env bash
set -euo pipefail

SKILL_NAME="inferencex-optimize"
MODE="copy"

usage() {
  cat <<'EOF'
Install the InferenceX Optimize skill for Claude Code and OpenCode.

Usage:
  ./install.sh
  ./install.sh --project /path/to/project
  ./install.sh --dest /custom/skill/dir
  ./install.sh --project /path/to/project --link

Options:
  --project PATH   Install into PATH/.claude/skills/inferencex-optimize
  --dest PATH      Install into an explicit skill directory path
  --link           Symlink instead of copying files
  --copy           Copy files explicitly (default)
  -h, --help       Show this help text
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$REPO_ROOT/skills/$SKILL_NAME"
DEST_DIR="${HOME}/.claude/skills/${SKILL_NAME}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      [[ $# -ge 2 ]] || { echo "Missing value for --project" >&2; exit 1; }
      DEST_DIR="$2/.claude/skills/${SKILL_NAME}"
      shift 2
      ;;
    --dest)
      [[ $# -ge 2 ]] || { echo "Missing value for --dest" >&2; exit 1; }
      DEST_DIR="$2"
      shift 2
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

require_dir "$SOURCE_DIR"
require_file "$SOURCE_DIR/SKILL.md"
require_file "$SOURCE_DIR/INSTALL.md"
require_file "$SOURCE_DIR/LICENSE"
require_dir "$SOURCE_DIR/phases"
require_dir "$SOURCE_DIR/templates"
require_dir "$SOURCE_DIR/scripts"

mkdir -p "$(dirname "$DEST_DIR")"

BACKUP_ROOT="$(dirname "$(dirname "$DEST_DIR")")/.skill-backups/$SKILL_NAME"

if [[ -e "$DEST_DIR" || -L "$DEST_DIR" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BACKUP_ROOT"
  BACKUP_PATH="${BACKUP_ROOT}/${TIMESTAMP}"
  mv "$DEST_DIR" "$BACKUP_PATH"
  echo "Backed up existing install to: $BACKUP_PATH"
fi

if [[ "$MODE" == "link" ]]; then
  ln -s "$SOURCE_DIR" "$DEST_DIR"
  INSTALL_MODE_MESSAGE="symlinked"
else
  mkdir -p "$DEST_DIR"
  cp -R "$SOURCE_DIR"/. "$DEST_DIR"/
  INSTALL_MODE_MESSAGE="copied"
fi

cat <<EOF
Installed skill: $SKILL_NAME
Destination: $DEST_DIR
Mode: $INSTALL_MODE_MESSAGE

This destination is compatible with:
- Claude Code
- OpenCode
EOF
