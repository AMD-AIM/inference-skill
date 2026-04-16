"""Tests for cross-kernel interaction threshold derivation."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.orchestrate.predicate_engine import evaluate_predicates_v2

CROSS_KERNEL_FALLBACK_PCT = 15.0


def derive_cross_kernel_threshold(config, benchmark_noise=None):
    """Derive cross-kernel interaction threshold.

    Priority:
    1. Explicit config value
    2. 3 * benchmark noise stddev
    3. Fallback percentage (default 15%)
    """
    explicit = config.get("CROSS_KERNEL_INTERACTION_THRESHOLD")
    if explicit is not None:
        return float(explicit)

    if benchmark_noise and "stddev_pct" in benchmark_noise:
        return 3.0 * benchmark_noise["stddev_pct"]

    fallback = config.get("CROSS_KERNEL_FALLBACK_PCT", CROSS_KERNEL_FALLBACK_PCT)
    return float(fallback)


class TestCrossKernelThreshold:
    def test_explicit_config(self):
        threshold = derive_cross_kernel_threshold(
            {"CROSS_KERNEL_INTERACTION_THRESHOLD": 20.0}
        )
        assert threshold == 20.0

    def test_noise_derived(self):
        threshold = derive_cross_kernel_threshold(
            {}, benchmark_noise={"stddev_pct": 5.0}
        )
        assert threshold == 15.0  # 3 * 5.0

    def test_fallback_default(self):
        threshold = derive_cross_kernel_threshold({})
        assert threshold == 15.0

    def test_fallback_custom(self):
        threshold = derive_cross_kernel_threshold(
            {"CROSS_KERNEL_FALLBACK_PCT": 10.0}
        )
        assert threshold == 10.0

    def test_explicit_overrides_noise(self):
        """Explicit config takes priority over noise-derived."""
        threshold = derive_cross_kernel_threshold(
            {"CROSS_KERNEL_INTERACTION_THRESHOLD": 5.0},
            benchmark_noise={"stddev_pct": 10.0},
        )
        assert threshold == 5.0

    def test_threshold_used_in_predicates(self):
        """Cross-kernel threshold works with $ref in predicate rules."""
        rules = [
            {"field": "cross_kernel_delta_pct", "op": "gt",
             "value": {"$ref": "$CROSS_KERNEL_THRESHOLD"},
             "verdict": "WARN", "category": "cross_kernel_interference"},
        ]
        threshold = derive_cross_kernel_threshold(
            {}, benchmark_noise={"stddev_pct": 5.0}
        )
        context = {"cross_kernel_delta_pct": 18.0}
        thresholds = {"CROSS_KERNEL_THRESHOLD": threshold}
        verdict, _, categories = evaluate_predicates_v2(rules, context, thresholds)
        assert verdict == "WARN"
        assert "cross_kernel_interference" in categories

    def test_below_threshold_passes(self):
        rules = [
            {"field": "cross_kernel_delta_pct", "op": "gt",
             "value": {"$ref": "$CROSS_KERNEL_THRESHOLD"},
             "verdict": "WARN", "category": "cross_kernel_interference"},
        ]
        context = {"cross_kernel_delta_pct": 10.0}
        thresholds = {"CROSS_KERNEL_THRESHOLD": 15.0}
        verdict, _, categories = evaluate_predicates_v2(rules, context, thresholds)
        assert verdict == "PASS"
        assert categories == []
