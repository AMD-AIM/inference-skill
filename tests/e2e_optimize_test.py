#!/usr/bin/env python3
"""Backward-compatible wrapper for the canonical skill E2E validator."""

from pathlib import Path
import runpy
import sys


CANONICAL = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "inferencex-optimize"
    / "tests"
    / "e2e_optimize_test.py"
)

if not CANONICAL.is_file():
    print(f"Error: canonical E2E validator not found: {CANONICAL}")
    sys.exit(1)

runpy.run_path(str(CANONICAL), run_name="__main__")
