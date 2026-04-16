"""Per-phase validators for the control-plane suite.

Each validator module exports:
    validate(output_dir: str) -> list[CheckResult]

The unified runner imports all validators and produces test_report.json.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CheckResult:
    phase: str
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"


def collect_validators():
    """Return a dict of phase_key -> validate function."""
    from . import (
        phase_02_benchmark,
        phase_05_profile_analyze,
        phase_07_kernel_optimize,
        phase_08_integration,
    )
    return {
        "benchmark": phase_02_benchmark.validate,
        "profile-analyze": phase_05_profile_analyze.validate,
        "kernel-optimize": phase_07_kernel_optimize.validate,
        "integration": phase_08_integration.validate,
    }
