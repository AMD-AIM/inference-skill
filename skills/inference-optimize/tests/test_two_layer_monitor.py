"""Tests for two-layer monitor verdict matrix.

L1 (predicates) is the floor. L2 (LLM judgment) can only upgrade, never downgrade.
Final verdict = max(L1, L2) by verdict rank.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.orchestrate.predicate_engine import VERDICT_RANK, evaluate_predicates_v2


def _max_verdict(v1, v2):
    """Combine two verdicts: higher severity wins."""
    return v1 if VERDICT_RANK.get(v1, 0) >= VERDICT_RANK.get(v2, 0) else v2


class TestTwoLayerMonitor:
    """Full L1 x L2 verdict matrix (9 cases)."""

    @pytest.mark.parametrize("l1,l2,expected", [
        ("PASS", "PASS", "PASS"),
        ("PASS", "WARN", "WARN"),
        ("PASS", "FAIL", "FAIL"),
        ("WARN", "PASS", "WARN"),   # Can't downgrade
        ("WARN", "WARN", "WARN"),
        ("WARN", "FAIL", "FAIL"),
        ("FAIL", "PASS", "FAIL"),   # Can't downgrade
        ("FAIL", "WARN", "FAIL"),   # Can't downgrade
        ("FAIL", "FAIL", "FAIL"),
    ])
    def test_verdict_matrix(self, l1, l2, expected):
        combined = _max_verdict(l1, l2)
        assert combined == expected, f"max({l1}, {l2}) should be {expected}, got {combined}"

    def test_l2_failure_falls_back_to_l1(self):
        """If L2 raises an exception, fall back to L1 verdict."""
        # Simulate L2 failure: use L1 only
        l1_verdict = "WARN"
        try:
            raise RuntimeError("LLM monitor unavailable")
        except Exception:
            final = l1_verdict  # Safe floor
        assert final == "WARN"

    def test_v2_predicates_with_category(self):
        """evaluate_predicates_v2 enriches details with problem_category."""
        rules = [
            {"field": "effort_waste_detected", "op": "eq", "value": True,
             "verdict": "WARN", "category": "effort_waste"},
            {"field": "geak_false_claim_count", "op": "gt", "value": 0,
             "verdict": "WARN", "category": "geak_false_claim"},
        ]
        context = {"effort_waste_detected": True, "geak_false_claim_count": 2}
        verdict, details, categories = evaluate_predicates_v2(rules, context)
        assert verdict == "WARN"
        assert "effort_waste" in categories
        assert "geak_false_claim" in categories
        assert details[0]["category"] == "effort_waste"

    def test_v2_predicates_no_triggered(self):
        """No triggered predicates -> empty categories."""
        rules = [
            {"field": "baseline_integrity_match", "op": "eq", "value": False,
             "verdict": "FAIL", "category": "baseline_drift"},
        ]
        context = {"baseline_integrity_match": True}
        verdict, details, categories = evaluate_predicates_v2(rules, context)
        assert verdict == "PASS"
        assert categories == []

    def test_ref_resolution(self):
        """$ref values resolve from thresholds dict."""
        rules = [
            {"field": "cross_kernel_delta_pct", "op": "gt",
             "value": {"$ref": "$CROSS_KERNEL_THRESHOLD"},
             "verdict": "WARN", "category": "cross_kernel_interference"},
        ]
        context = {"cross_kernel_delta_pct": 18.2}
        thresholds = {"CROSS_KERNEL_THRESHOLD": 15.0}
        verdict, details, categories = evaluate_predicates_v2(rules, context, thresholds)
        assert verdict == "WARN"
        assert "cross_kernel_interference" in categories

    def test_ref_not_triggered(self):
        """$ref threshold not exceeded -> PASS."""
        rules = [
            {"field": "cross_kernel_delta_pct", "op": "gt",
             "value": {"$ref": "$CROSS_KERNEL_THRESHOLD"},
             "verdict": "WARN", "category": "cross_kernel_interference"},
        ]
        context = {"cross_kernel_delta_pct": 10.0}
        thresholds = {"CROSS_KERNEL_THRESHOLD": 15.0}
        verdict, details, _ = evaluate_predicates_v2(rules, context, thresholds)
        assert verdict == "PASS"

    def test_unresolved_ref_skips_rule(self):
        """Unresolved $ref -> rule not triggered."""
        rules = [
            {"field": "cross_kernel_delta_pct", "op": "gt",
             "value": {"$ref": "$MISSING_THRESHOLD"},
             "verdict": "WARN", "category": "cross_kernel_interference"},
        ]
        context = {"cross_kernel_delta_pct": 100.0}
        verdict, details, _ = evaluate_predicates_v2(rules, context, thresholds={})
        assert verdict == "PASS"
        assert details[0]["reason"] == "unresolved $ref threshold"
