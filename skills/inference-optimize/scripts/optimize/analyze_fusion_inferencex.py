#!/usr/bin/env python3
"""DEPRECATED: Use analyze_fusion.py instead.

This file is kept for backward compatibility. It re-exports everything
from analyze_fusion.py. All new code should import from analyze_fusion
directly.
"""
import sys
import os
import warnings

warnings.warn(
    "analyze_fusion_inferencex.py is deprecated; use analyze_fusion.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical module
from analyze_fusion import *  # noqa: F401,F403
from analyze_fusion import main

if __name__ == "__main__":
    sys.exit(main())
