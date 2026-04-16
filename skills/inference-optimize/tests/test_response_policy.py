"""Tests for response policy engine."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.orchestrate.response_policy import determine_response


RERUN_LIMITS = {"max_per_phase": 2, "max_total": 5}


class MockRunnerState:
    def __init__(self, retry_counts=None, total_reruns=0, human_extensions=None):
        self.retry_counts = retry_counts or {}
        self.total_reruns = total_reruns
        self.human_extensions = human_extensions or {}
        self.max_human_extensions = 3


class TestResponsePolicy:
    def test_pass_continues(self):
        state = MockRunnerState()
        resp = determine_response("PASS", None, "benchmark", {}, state)
        assert resp["action"] == "continue"

    def test_warn_continues(self):
        state = MockRunnerState()
        resp = determine_response("WARN", None, "benchmark", {}, state)
        assert resp["action"] == "continue"

    def test_safety_stop_overrides_everything(self):
        state = MockRunnerState()
        rca = {"terminal_action": "stop_with_blocker", "analysis": "critical failure"}
        escalation = {"action": "retry", "notes": "try again"}
        resp = determine_response("FAIL", "logic", "benchmark", {}, state,
                                   rca_result=rca, escalation_result=escalation)
        assert resp["action"] == "abort"
        assert "safety stop" in resp["reason"].lower()

    def test_human_override_retry(self):
        state = MockRunnerState()
        escalation = {"action": "retry", "notes": "I'll handle it"}
        resp = determine_response("FAIL", "logic", "kernel-optimize", {}, state,
                                   escalation_result=escalation)
        assert resp["action"] == "retry"

    def test_human_extension_cap(self):
        state = MockRunnerState(
            retry_counts={"kernel-optimize": 2},
            human_extensions={"kernel-optimize": 3},
        )
        escalation = {"action": "retry", "notes": "one more try"}
        resp = determine_response("FAIL", "logic", "kernel-optimize", {}, state,
                                   escalation_result=escalation,
                                   rerun_limits=RERUN_LIMITS, phase_reruns=3)
        assert resp["action"] == "abort"
        assert "cap" in resp["reason"].lower()

    def test_human_extension_within_cap(self):
        state = MockRunnerState(
            retry_counts={"kernel-optimize": 2},
            human_extensions={"kernel-optimize": 1},
        )
        escalation = {"action": "retry", "notes": "try"}
        resp = determine_response("FAIL", "logic", "kernel-optimize", {}, state,
                                   escalation_result=escalation,
                                   rerun_limits=RERUN_LIMITS, phase_reruns=3)
        assert resp["action"] == "retry"
        assert "extended" in resp["reason"].lower()

    def test_budget_exhausted_with_fallback(self):
        state = MockRunnerState(retry_counts={"kernel-optimize": 2}, total_reruns=3)
        meta = {"fallback_target": "problem-generate"}
        resp = determine_response("FAIL", "logic", "kernel-optimize", meta, state,
                                   rerun_limits=RERUN_LIMITS, phase_reruns=3)
        assert resp["action"] == "redirect"
        assert resp["target"] == "problem-generate"

    def test_budget_exhausted_allow_partial(self):
        state = MockRunnerState(retry_counts={"kernel-optimize": 2}, total_reruns=3)
        meta = {"terminal_policy": "allow_partial_report"}
        resp = determine_response("FAIL", "logic", "kernel-optimize", meta, state,
                                   rerun_limits=RERUN_LIMITS, phase_reruns=3)
        assert resp["action"] == "continue"

    def test_budget_exhausted_no_fallback_abort(self):
        state = MockRunnerState(retry_counts={"benchmark": 2}, total_reruns=3)
        meta = {"terminal_policy": "stop"}
        resp = determine_response("FAIL", "logic", "benchmark", meta, state,
                                   rerun_limits=RERUN_LIMITS, phase_reruns=3)
        assert resp["action"] == "abort"

    def test_rca_recommends_retry(self):
        state = MockRunnerState()
        rca = {"terminal_action": "retry", "analysis": "transient error"}
        resp = determine_response("FAIL", "infrastructure", "benchmark", {}, state,
                                   rca_result=rca)
        assert resp["action"] == "retry"

    def test_rca_recommends_fallback(self):
        state = MockRunnerState()
        meta = {"fallback_target": "profile"}
        rca = {"terminal_action": "fallback"}
        resp = determine_response("FAIL", "logic", "profile-analyze", meta, state,
                                   rca_result=rca)
        assert resp["action"] == "redirect"
        assert resp["target"] == "profile"

    def test_default_retry(self):
        state = MockRunnerState()
        resp = determine_response("FAIL", "logic", "kernel-optimize", {}, state)
        assert resp["action"] == "retry"

    def test_total_budget_exhausted(self):
        state = MockRunnerState(total_reruns=6)
        resp = determine_response("FAIL", "logic", "kernel-optimize",
                                   {"fallback_target": "problem-generate"}, state,
                                   rerun_limits=RERUN_LIMITS, phase_reruns=1)
        assert resp["action"] == "redirect"
