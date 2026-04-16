"""Roofline-informed effort budget calculator.

Determines per-kernel optimization attempt limits based on roofline
analysis data. Kernels already near their theoretical ceiling get fewer
attempts, while kernels with large gaps get more.

Infrastructure failures (OOM, container crash) use a separate budget
from optimization attempts.
"""
from typing import NamedTuple


class EffortBudget(NamedTuple):
    """Per-kernel effort budget.

    max_optimization_attempts: How many optimization attempts (GEAK/manual).
                              Derived from roofline efficiency.
    max_infra_retries: How many infrastructure retries (OOM, container crash).
                      Always >= 2, separate from optimization budget.
    """
    max_optimization_attempts: int
    max_infra_retries: int


def compute_effort_budget(roofline_efficiency=None, bound=None, default_opt=5):
    """Compute effort budget from roofline analysis data.

    Args:
        roofline_efficiency: Percentage of theoretical peak (0-100).
                           None if no roofline data available.
        bound: Bottleneck type from profiling: "memory", "compute", or "unknown".
              Memory-bound kernels get max attempts regardless of efficiency.
        default_opt: Default optimization attempts when no roofline data.

    Returns:
        EffortBudget namedtuple.
    """
    if roofline_efficiency is None:
        return EffortBudget(max_optimization_attempts=default_opt, max_infra_retries=2)

    # Memory-bound override -- checked BEFORE efficiency branches
    if bound == "memory":
        return EffortBudget(max_optimization_attempts=5, max_infra_retries=2)

    if roofline_efficiency >= 90.0:
        return EffortBudget(max_optimization_attempts=1, max_infra_retries=2)

    if roofline_efficiency >= 50.0:
        return EffortBudget(max_optimization_attempts=3, max_infra_retries=2)

    return EffortBudget(max_optimization_attempts=5, max_infra_retries=2)
