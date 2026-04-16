#!/usr/bin/env python3
"""DEPRECATED: Use generate_problems.py instead.

This file is kept for backward compatibility. It re-exports everything
from generate_problems.py. All new code should import from generate_problems
directly.
"""
import sys
import warnings

warnings.warn(
    "generate_problems_inferencex.py is deprecated; use generate_problems.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical module
from generate_problems import *  # noqa: F401,F403
from generate_problems import main

if __name__ == "__main__":
    sys.exit(main())
