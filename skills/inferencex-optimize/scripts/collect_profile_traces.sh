#!/usr/bin/env bash
# Collect profile traces and benchmark results from the repo into the output directory.
#
# Usage: bash collect_profile_traces.sh \
#   --repo-dir <dir> --profile-dir <dir> --output-dir <dir> --result-filename <name>
set -euo pipefail

REPO_DIR=""
PROFILE_DIR=""
OUTPUT_DIR=""
RESULT_FILENAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir)         REPO_DIR="$2"; shift 2 ;;
        --profile-dir)      PROFILE_DIR="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --result-filename)  RESULT_FILENAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Collect torch traces, skip async_llm frontend-only traces
for f in "$REPO_DIR"/profiles/*.json*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        *async_llm*) rm -f "$f" ;;
        *)           cp "$f" "$PROFILE_DIR/" && rm -f "$f" ;;
    esac
done

# Collect profiler summary
if [ -f "$REPO_DIR/profiles/profiler_out_0.txt" ]; then
    cp "$REPO_DIR/profiles/profiler_out_0.txt" "$PROFILE_DIR/profiler_out_0.txt"
    rm -f "$REPO_DIR/profiles/profiler_out_0.txt"
    echo "Collected profiler_out_0.txt"
fi

# Collect benchmark result JSONs
mkdir -p "$OUTPUT_DIR/results"
cp "$REPO_DIR"/${RESULT_FILENAME}*.json "$OUTPUT_DIR/results/" 2>/dev/null || true
rm -f "$REPO_DIR"/${RESULT_FILENAME}*.json 2>/dev/null || true
cp "$REPO_DIR"/results/${RESULT_FILENAME}*.json "$OUTPUT_DIR/results/" 2>/dev/null || true
rm -f "$REPO_DIR"/results/${RESULT_FILENAME}*.json 2>/dev/null || true

echo "Collected trace files:"
ls -lh "$PROFILE_DIR/"
echo "Collected benchmark results:"
ls -lh "$OUTPUT_DIR/results/" 2>/dev/null || echo "(none)"
