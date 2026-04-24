#!/usr/bin/env python3
"""Tests for the structured predicate engine.

Run:  python3 -m pytest tests/test_predicate_engine.py -v
"""

import json
import pathlib
import sys

import pytest

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER_DIR = SKILL_ROOT / "scripts" / "orchestrate"
REGISTRY_PATH = SKILL_ROOT / "orchestrator" / "phase-registry.json"

sys.path.insert(0, str(RUNNER_DIR))

from predicate_engine import evaluate_predicates


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


class TestBasicOperators:
    def test_eq_match(self):
        rules = [{"field": "gate", "op": "eq", "value": "fail", "verdict": "FAIL"}]
        v, d = evaluate_predicates(rules, {"gate": "fail"})
        assert v == "FAIL"

    def test_eq_no_match(self):
        rules = [{"field": "gate", "op": "eq", "value": "fail", "verdict": "FAIL"}]
        v, _ = evaluate_predicates(rules, {"gate": "pass"})
        assert v == "PASS"

    def test_lt_numeric(self):
        rules = [{"field": "speedup", "op": "lt", "value": 0.97, "verdict": "FAIL"}]
        v, _ = evaluate_predicates(rules, {"speedup": 0.90})
        assert v == "FAIL"

    def test_lt_not_triggered(self):
        rules = [{"field": "speedup", "op": "lt", "value": 0.97, "verdict": "FAIL"}]
        v, _ = evaluate_predicates(rules, {"speedup": 1.05})
        assert v == "PASS"

    def test_gt_numeric(self):
        rules = [{"field": "pct", "op": "gt", "value": 20.0, "verdict": "FAIL"}]
        v, _ = evaluate_predicates(rules, {"pct": 25.0})
        assert v == "FAIL"

    def test_missing_field_skipped(self):
        rules = [{"field": "missing", "op": "eq", "value": "x", "verdict": "FAIL"}]
        v, d = evaluate_predicates(rules, {})
        assert v == "PASS"
        assert d[0]["triggered"] is False


class TestConditionalPredicates:
    def test_condition_met(self):
        rules = [{"field": "ttft_pct", "op": "gt", "value": 20.0,
                   "verdict": "FAIL", "condition": "gate == warn"}]
        v, _ = evaluate_predicates(rules, {"ttft_pct": 25.0, "gate": "warn"})
        assert v == "FAIL"

    def test_condition_not_met(self):
        rules = [{"field": "ttft_pct", "op": "gt", "value": 20.0,
                   "verdict": "FAIL", "condition": "gate == warn"}]
        v, d = evaluate_predicates(rules, {"ttft_pct": 25.0, "gate": "pass"})
        assert v == "PASS"
        assert d[0]["triggered"] is False


class TestVerdictPriority:
    def test_fail_overrides_legacy_warn(self):
        rules = [
            {"field": "speedup", "op": "lt", "value": 1.0, "verdict": "WARN"},
            {"field": "speedup", "op": "lt", "value": 0.97, "verdict": "FAIL"},
        ]
        v, _ = evaluate_predicates(rules, {"speedup": 0.90})
        assert v == "FAIL"

    def test_legacy_warn_normalizes_to_fail(self):
        rules = [
            {"field": "speedup", "op": "lt", "value": 1.0, "verdict": "WARN"},
            {"field": "speedup", "op": "lt", "value": 0.97, "verdict": "FAIL"},
        ]
        v, _ = evaluate_predicates(rules, {"speedup": 0.98})
        assert v == "FAIL"

    def test_pass_when_all_clear(self):
        rules = [
            {"field": "speedup", "op": "lt", "value": 1.0, "verdict": "WARN"},
            {"field": "speedup", "op": "lt", "value": 0.97, "verdict": "FAIL"},
        ]
        v, _ = evaluate_predicates(rules, {"speedup": 1.05})
        assert v == "PASS"


class TestIntegrationPredicates:
    """Test the actual integration phase predicates from the registry."""

    def test_integration_pass(self):
        reg = _load_registry()
        rules = reg["phases"]["integration"]["quality"]["detection_rules_structured"]
        context = {"performance_gate": "pass", "e2e_speedup": 1.2,
                    "ttft_regression_pct": -5.0}
        v, _ = evaluate_predicates(rules, context)
        assert v == "PASS"

    def test_integration_fail_regression(self):
        reg = _load_registry()
        rules = reg["phases"]["integration"]["quality"]["detection_rules_structured"]
        context = {"performance_gate": "fail", "e2e_speedup": 0.90,
                    "ttft_regression_pct": 5.0}
        v, _ = evaluate_predicates(rules, context)
        assert v == "FAIL"

    def test_integration_warn_band_is_hard_fail(self):
        reg = _load_registry()
        rules = reg["phases"]["integration"]["quality"]["detection_rules_structured"]
        context = {"performance_gate": "warn", "e2e_speedup": 0.98,
                    "ttft_regression_pct": 2.0}
        v, _ = evaluate_predicates(rules, context)
        assert v == "FAIL"

    def test_integration_ttft_upgrade(self):
        reg = _load_registry()
        rules = reg["phases"]["integration"]["quality"]["detection_rules_structured"]
        context = {"performance_gate": "warn", "e2e_speedup": 0.98,
                    "ttft_regression_pct": 25.0}
        v, _ = evaluate_predicates(rules, context)
        assert v == "FAIL"


class TestBenchmarkPredicates:
    def test_benchmark_pass(self):
        reg = _load_registry()
        rules = reg["phases"]["benchmark"]["quality"]["detection_rules_structured"]
        context = {"benchmarks_succeeded": 4, "benchmark_result_status": "completed",
                    "baseline_artifacts_ready": True}
        v, _ = evaluate_predicates(rules, context)
        assert v == "PASS"

    def test_benchmark_fail_no_successes(self):
        reg = _load_registry()
        rules = reg["phases"]["benchmark"]["quality"]["detection_rules_structured"]
        context = {"benchmarks_succeeded": 0, "benchmark_result_status": "failed",
                    "baseline_artifacts_ready": False}
        v, _ = evaluate_predicates(rules, context)
        assert v == "FAIL"


class TestKernelOptimizePredicates:
    def test_pre_flight_failure_is_hard_fail(self):
        """rocprofv3 pre-flight: when patched symbol does not actually fire on
        the rebuilt env, the phase must FAIL before Phase 8 burns wall-clock."""
        reg = _load_registry()
        rules = reg["phases"]["kernel-optimize"]["quality"]["detection_rules_structured"]
        context = {
            "dispatch_pre_flight_pass": False,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
        }
        v, _ = evaluate_predicates(rules, context)
        assert v == "FAIL"

    def test_library_test_regression_fails(self):
        """A Bucket A library suite regression against the patched fork must
        FAIL the phase even when the per-kernel inner loop reported success."""
        reg = _load_registry()
        rules = reg["phases"]["kernel-optimize"]["quality"]["detection_rules_structured"]
        context = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 1,
            "allocator_test_pass": True,
        }
        v, _ = evaluate_predicates(rules, context)
        assert v == "FAIL"

    def test_clean_kernel_optimize_passes(self):
        reg = _load_registry()
        rules = reg["phases"]["kernel-optimize"]["quality"]["detection_rules_structured"]
        context = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
        }
        v, _ = evaluate_predicates(rules, context)
        assert v == "PASS"


class TestAllRegistryPredicatesValid:
    """Every detection_rules_structured in the registry must be well-formed."""

    def test_all_rules_have_required_fields(self):
        reg = _load_registry()
        for key, phase in reg["phases"].items():
            rules = phase.get("quality", {}).get("detection_rules_structured", [])
            for i, rule in enumerate(rules):
                assert "field" in rule, f"Phase {key} rule {i}: missing 'field'"
                assert "op" in rule, f"Phase {key} rule {i}: missing 'op'"
                assert "verdict" in rule, f"Phase {key} rule {i}: missing 'verdict'"
                assert rule["verdict"] in ("PASS", "FAIL"), (
                    f"Phase {key} rule {i}: invalid verdict {rule['verdict']}"
                )
