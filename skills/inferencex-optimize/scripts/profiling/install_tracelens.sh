#!/usr/bin/env bash
# Install TraceLens if not already available.
# Usage: bash install_tracelens.sh [TARBALL_DIR]
#   TARBALL_DIR: directory containing TraceLens-internal.tar.gz (optional)
#
# Outputs TRACELENS_INSTALL_FAILED=true on failure.
set -euo pipefail

TARBALL_DIR="${1:-}"

export PATH="$HOME/.local/bin:$PATH"

tracelens_cli_ready() {
    command -v TraceLens_generate_perf_report_pytorch &>/dev/null
}

if tracelens_cli_ready; then
    echo "TraceLens CLI already available"
    exit 0
fi

if [ ! -d "$HOME/TraceLens-internal" ]; then
    echo "Cloning TraceLens-internal..."
    if ! git clone git@github.com:AMD-AGI/TraceLens-internal.git "$HOME/TraceLens-internal"; then
        echo "Git clone failed, extracting from bundled tarball..."
        rm -rf "$HOME/TraceLens-internal"
        TARBALL="${TARBALL_DIR:+$TARBALL_DIR/}TraceLens-internal.tar.gz"
        if [ -f "$TARBALL" ]; then
            if tar xzf "$TARBALL" -C "$HOME"; then
                echo "Extracted TraceLens-internal from tarball"
            else
                echo "ERROR: tar extraction failed (exit code $?)"
                echo "TRACELENS_INSTALL_FAILED=true"
                exit 1
            fi
        else
            echo "ERROR: TraceLens tarball not found at $TARBALL"
            echo "TRACELENS_INSTALL_FAILED=true"
            exit 1
        fi
    fi
fi

if [ -d "$HOME/TraceLens-internal" ]; then
    echo "Installing TraceLens (this may take a few minutes)..."
    pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
    hash -r
    if tracelens_cli_ready; then
        echo "TraceLens CLI installed successfully"
    else
        echo "First install attempt failed — retrying..."
        pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
        hash -r
        if tracelens_cli_ready; then
            echo "TraceLens CLI installed successfully on retry"
        else
            echo "TRACELENS_INSTALL_FAILED=true"
            echo "ERROR: TraceLens installation failed after retry"
            exit 1
        fi
    fi
fi
