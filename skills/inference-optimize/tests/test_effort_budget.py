"""Tests for roofline-informed effort budget calculator."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.optimize.effort_budget import compute_effort_budget, EffortBudget


class TestEffortBudget:
    def test_high_efficiency_low_attempts(self):
        """roofline_efficiency >= 90% -> max_optimization_attempts=1"""
        budget = compute_effort_budget(roofline_efficiency=95.0, bound="compute")
        assert budget.max_optimization_attempts == 1
        assert budget.max_infra_retries == 2

    def test_medium_efficiency(self):
        """roofline_efficiency 50-90% -> max_optimization_attempts=3"""
        budget = compute_effort_budget(roofline_efficiency=75.0, bound="compute")
        assert budget.max_optimization_attempts == 3
        assert budget.max_infra_retries == 2

    def test_low_efficiency_max_attempts(self):
        """roofline_efficiency < 50% -> max_optimization_attempts=5"""
        budget = compute_effort_budget(roofline_efficiency=30.0, bound="compute")
        assert budget.max_optimization_attempts == 5
        assert budget.max_infra_retries == 2

    def test_memory_bound_override(self):
        """Memory-bound kernels get max attempts regardless of efficiency."""
        budget = compute_effort_budget(roofline_efficiency=95.0, bound="memory")
        assert budget.max_optimization_attempts == 5
        assert budget.max_infra_retries == 2

    def test_no_roofline_data_default(self):
        """No roofline data -> default budget."""
        budget = compute_effort_budget(roofline_efficiency=None)
        assert budget.max_optimization_attempts == 5
        assert budget.max_infra_retries == 2

    def test_boundary_90(self):
        """Exactly 90% -> 1 attempt."""
        budget = compute_effort_budget(roofline_efficiency=90.0, bound="compute")
        assert budget.max_optimization_attempts == 1

    def test_boundary_50(self):
        """Exactly 50% -> 3 attempts."""
        budget = compute_effort_budget(roofline_efficiency=50.0, bound="compute")
        assert budget.max_optimization_attempts == 3

    def test_just_below_50(self):
        """49.9% -> 5 attempts."""
        budget = compute_effort_budget(roofline_efficiency=49.9, bound="compute")
        assert budget.max_optimization_attempts == 5

    def test_namedtuple_fields(self):
        """EffortBudget is a proper NamedTuple."""
        budget = compute_effort_budget(roofline_efficiency=75.0)
        assert isinstance(budget, EffortBudget)
        assert isinstance(budget, tuple)
        assert budget._fields == ("max_optimization_attempts", "max_infra_retries")

    def test_memory_bound_checked_before_efficiency(self):
        """Memory bound is checked BEFORE efficiency branches."""
        # High efficiency + memory bound = max attempts
        budget = compute_effort_budget(roofline_efficiency=99.0, bound="memory")
        assert budget.max_optimization_attempts == 5

    def test_infra_retries_always_2(self):
        """Infrastructure retries are always 2 regardless of other params."""
        for eff in [None, 10.0, 50.0, 75.0, 95.0]:
            budget = compute_effort_budget(roofline_efficiency=eff)
            assert budget.max_infra_retries == 2
