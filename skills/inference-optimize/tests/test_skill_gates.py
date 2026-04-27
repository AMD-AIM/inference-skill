"""Regression tests for the skill-gates fix (registry winners contract,
empty optimized/ gate, awaiting_user_instruction flow, retry budget,
binary monitor verdicts).

These tests pin the behaviors changed in
``.cursor/plans/inference_skill_gates_3390fed8.plan.md``.
"""
import json
import os
import pathlib
import sys
import tempfile

import pytest

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = pathlib.Path(__file__).resolve().parent
RUNNER_DIR = SKILL_ROOT / "scripts" / "orchestrate"
REGISTRY_PATH = SKILL_ROOT / "orchestrator" / "phase-registry.json"

sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(TESTS_DIR))

from runner import DeterministicRunner  # noqa: E402
from predicate_engine import evaluate_predicates  # noqa: E402

from validators import phase_07_kernel_optimize as phase07  # noqa: E402


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _kernel_optimize_rules():
    reg = _load_registry()
    return reg["phases"]["kernel-optimize"]["quality"]["detection_rules_structured"]


# ---------------------------------------------------------------------------
# Phase 7 structured gates: shipped-winner semantics
# ---------------------------------------------------------------------------


class TestPhase07StructuredGates:
    def test_zero_winners_fails(self):
        """winners_total_count == 0 must trigger FAIL."""
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 0,
            "claimed_winner_artifacts_valid": True,
            "optimized_artifact_count": 0,
            "redirect_attempted": False,
            "mini_ab_required": False,
        }
        verdict, details = evaluate_predicates(rules, ctx)
        assert verdict == "FAIL"
        triggered = [d for d in details if d.get("triggered")]
        fields = {d["field"] for d in triggered}
        assert "winners_total_count" in fields

    def test_redirect_count_out_of_tolerance_fails(self):
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 1,
            "claimed_winner_artifacts_valid": True,
            "optimized_artifact_count": 1,
            "redirect_attempted": True,
            "redirect_count_within_tolerance": False,
            "mini_ab_required": False,
        }
        verdict, details = evaluate_predicates(rules, ctx)
        assert verdict == "FAIL"
        fields = {d["field"] for d in details if d.get("triggered")}
        assert "redirect_count_within_tolerance" in fields

    def test_redirect_count_skipped_when_no_redirect_attempted(self):
        """The shape gate must NOT fire on phases that did not attempt
        a redirect (e.g. in_place winners only)."""
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 1,
            "claimed_winner_artifacts_valid": True,
            "optimized_artifact_count": 1,
            "redirect_attempted": False,
            "redirect_count_within_tolerance": False,  # value doesn't matter
            "mini_ab_required": False,
        }
        verdict, _ = evaluate_predicates(rules, ctx)
        assert verdict == "PASS"

    def test_unreliable_mini_ab_fails(self):
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 1,
            "claimed_winner_artifacts_valid": True,
            "optimized_artifact_count": 1,
            "redirect_attempted": True,
            "redirect_count_within_tolerance": True,
            "mini_ab_required": True,
            "mini_ab_harness_status": "unreliable_high_variance",
        }
        verdict, details = evaluate_predicates(rules, ctx)
        assert verdict == "FAIL"
        fields = {d["field"] for d in details if d.get("triggered")}
        assert "mini_ab_harness_status" in fields

    def test_empty_optimized_dir_with_claimed_winner_fails(self):
        """The new conditional rule must FAIL when winners are claimed
        but optimized_artifact_count is 0 — this matches the observed
        run where optimized/ was empty even though redirect commits
        were applied."""
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 1,
            "claimed_winner_artifacts_valid": True,
            "optimized_artifact_count": 0,
            "redirect_attempted": False,
            "mini_ab_required": False,
        }
        verdict, details = evaluate_predicates(rules, ctx)
        assert verdict == "FAIL"
        fields = {d["field"] for d in details if d.get("triggered")}
        assert "optimized_artifact_count" in fields

    def test_empty_optimized_dir_no_winners_passes(self):
        """An empty optimized/ is not by itself a failure when there
        are also no winners — winners_total_count == 0 fires the
        zero-winner rule, and the conditional optimized_artifact rule
        does not fire."""
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 0,
            "claimed_winner_artifacts_valid": True,
            "optimized_artifact_count": 0,
            "redirect_attempted": False,
            "mini_ab_required": False,
        }
        verdict, details = evaluate_predicates(rules, ctx)
        # Still FAIL because winners_total_count < 1 fires; but make
        # sure the optimized_artifact rule did NOT fire.
        assert verdict == "FAIL"
        triggered = {d["field"] for d in details if d.get("triggered")}
        assert "optimized_artifact_count" not in triggered

    def test_observed_failure_run_signature(self):
        """End-to-end signature of the
        /home/ziwei/inference_qwen3.5-9b-vllm-mi355x_20260420_071230 run:
        zero exports, redirect over-routing, empty optimized/, mini-A/B
        unreliable. The new structured rules must FAIL on this exact
        scalar profile."""
        rules = _kernel_optimize_rules()
        ctx = {
            "dispatch_pre_flight_pass": True,
            "library_tests_failed_count": 0,
            "allocator_test_pass": True,
            "winners_total_count": 0,
            "py_exports_shipped_count": 0,
            "claimed_winner_artifacts_valid": False,
            "optimized_artifact_count": 0,
            "optimized_dir_empty": True,
            "redirect_attempted": True,
            "redirect_count_observed": 774,
            "redirect_count_expected": 212,
            "redirect_count_within_tolerance": False,
            "mini_ab_required": True,
            "mini_ab_harness_status": "unreliable_high_variance",
        }
        verdict, details = evaluate_predicates(rules, ctx)
        assert verdict == "FAIL"
        triggered = {d["field"] for d in details if d.get("triggered")}
        # All four shipped-winner gates fire on this scalar profile.
        for required in (
            "winners_total_count",
            "claimed_winner_artifacts_valid",
            "redirect_count_within_tolerance",
            "mini_ab_harness_status",
        ):
            assert required in triggered, f"{required} should fire"


# ---------------------------------------------------------------------------
# Phase 7 validator alignment with registry semantics
# ---------------------------------------------------------------------------


class TestPhase07ValidatorAlignment:
    def _write_geak_results(self, tmpdir, kernels):
        os.makedirs(os.path.join(tmpdir, "problems"), exist_ok=True)
        with open(os.path.join(tmpdir, "problems", "geak_results.json"), "w") as f:
            json.dump({"kernels": kernels}, f)

    def _write_preflight(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, "results"), exist_ok=True)
        with open(os.path.join(tmpdir, "results",
                               "preflight_dispatch_trace.json"), "w") as f:
            json.dump({"dispatch_pre_flight_pass": True}, f)

    def test_zero_shipped_winners_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "forks"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "optimized"), exist_ok=True)
            self._write_geak_results(tmpdir, [
                {
                    "name": "k1",
                    "geak_strategy": "dispatch_redirect_to_open_lib",
                    "winner_strategy": "not_a_winner_count_out_of_tolerance",
                },
            ])
            self._write_preflight(tmpdir)
            results = phase07.validate(tmpdir)
            named = {r.name: r for r in results}
            assert named["winners_present"].passed is False
            assert "winners_total_count=0" in named["winners_present"].detail

    def test_redirect_commits_alone_dont_count_as_winner(self):
        """Audit-only `redirect_commits_applied` rows that did not
        ship a winner must not pass the validator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "forks"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "optimized"), exist_ok=True)
            self._write_geak_results(tmpdir, [
                {
                    "name": "k1",
                    "geak_strategy": "dispatch_redirect_to_open_lib",
                    "winner_strategy": "not_a_winner_count_out_of_tolerance",
                    "redirect_commits_applied_count": 1,
                },
            ])
            self._write_preflight(tmpdir)
            results = phase07.validate(tmpdir)
            named = {r.name: r for r in results}
            assert named["winners_present"].passed is False

    def test_optimized_dir_empty_with_claimed_winner_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "forks"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "optimized"), exist_ok=True)
            self._write_geak_results(tmpdir, [
                {
                    "name": "k1",
                    "geak_strategy": "in_place_optimize",
                    "winner_strategy": "shipped",
                    "geak_speedup_lib_bench": 1.5,
                },
            ])
            self._write_preflight(tmpdir)
            results = phase07.validate(tmpdir)
            named = {r.name: r for r in results}
            assert named["winners_present"].passed is True
            assert named["optimized_artifact_count"].passed is False

    def test_redirect_within_tolerance_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "forks"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "optimized"), exist_ok=True)
            with open(os.path.join(tmpdir, "optimized", "k1.py"), "w") as f:
                f.write("# winner\n")
            self._write_geak_results(tmpdir, [
                {
                    "name": "k1",
                    "geak_strategy": "dispatch_redirect_to_open_lib",
                    "winner_strategy": "shipped",
                    "redirect_count_within_tolerance": True,
                    "claimed_winner_artifact_valid": True,
                },
            ])
            self._write_preflight(tmpdir)
            results = phase07.validate(tmpdir)
            named = {r.name: r for r in results}
            assert named["winners_present"].passed is True
            assert named["redirect_count_within_tolerance"].passed is True
            assert named["claimed_winner_artifacts_valid"].passed is True


# ---------------------------------------------------------------------------
# Monitor verdict normalization (binary contract + legacy shim)
# ---------------------------------------------------------------------------


class TestMonitorVerdictBinary:
    def test_pass_with_caveats_normalizes_to_fail(self):
        assert (
            DeterministicRunner.normalize_monitor_verdict("PASS_with_caveats")
            == "FAIL"
        )

    def test_fail_pushed_through_normalizes_to_fail(self):
        assert (
            DeterministicRunner.normalize_monitor_verdict("FAIL_pushed_through")
            == "FAIL"
        )

    def test_warn_normalizes_to_fail(self):
        assert DeterministicRunner.normalize_monitor_verdict("WARN") == "FAIL"

    def test_random_string_normalizes_to_fail(self):
        assert DeterministicRunner.normalize_monitor_verdict("MIXED") == "FAIL"

    def test_pass_passes_through(self):
        assert DeterministicRunner.normalize_monitor_verdict("PASS") == "PASS"

    def test_fail_passes_through(self):
        assert DeterministicRunner.normalize_monitor_verdict("FAIL") == "FAIL"

    def test_invalid_verdict_detection(self):
        assert DeterministicRunner.is_invalid_verdict("PASS_with_caveats")
        assert DeterministicRunner.is_invalid_verdict("WARN")
        assert not DeterministicRunner.is_invalid_verdict("PASS")
        assert not DeterministicRunner.is_invalid_verdict("FAIL")


# ---------------------------------------------------------------------------
# Runner control flow: awaiting_user_instruction replaces partial-report skip
# ---------------------------------------------------------------------------


def _make_optimize_workspace(tmpdir):
    """Create the minimum artifact tree so optimize-mode prerequisite
    checks pass for shadow/mocked runs."""
    for d in (
        "handoff", "agent-results", "monitor", "results",
        "results/parity", "results/gap_analysis",
        "problems", "optimized", "profiles", "scripts",
        "templates", "reports", "resources", "forks",
    ):
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
    with open(os.path.join(tmpdir, "env_info.json"), "w") as f:
        json.dump({"gpu_arch": "mi300x", "gpu_count": 8}, f)
    with open(os.path.join(tmpdir, "results", "sweep_configs.json"), "w") as f:
        json.dump({"framework": "sglang", "precision": "bf16"}, f)
    with open(os.path.join(tmpdir, "results", "gap_analysis",
                           "gap_analysis.json"), "w") as f:
        json.dump({"top_kernels": ["k1"]}, f)
    with open(os.path.join(tmpdir, "results", "profile_analysis.json"), "w") as f:
        json.dump({"status": "complete"}, f)
    with open(os.path.join(tmpdir, "problems",
                           "optimization_manifest.json"), "w") as f:
        json.dump({"optimizations": []}, f)
    with open(os.path.join(tmpdir, "problems", "geak_results.json"), "w") as f:
        json.dump({"kernels": []}, f)
    with open(os.path.join(tmpdir, "forks", "manifest.json"), "w") as f:
        json.dump({"libraries": [], "ck_branch_merged_status": False,
                   "vllm_version": "test"}, f)
    with open(os.path.join(tmpdir, "optimized", "mock_winner.py"), "w") as f:
        f.write("# mock integration input\n")


def _build_optimize_config(tmpdir, **overrides):
    config = {
        "CONFIG_KEY": "test-config",
        "OUTPUT_DIR": tmpdir,
        "MODE": "optimize",
        "REPO_DIR": os.path.join(tmpdir, "benchmark_repo"),
        "REPO_URL": "https://github.com/SemiAnalysisAI/InferenceX.git",
        "HF_CACHE": "/tmp/hf",
        "SCRIPTS_DIR": os.path.join(tmpdir, "scripts"),
        "PROFILE_DIR": os.path.join(tmpdir, "profiles"),
        "RESULTS_DIR": os.path.join(tmpdir, "results"),
        "PROBLEMS_DIR": os.path.join(tmpdir, "problems"),
        "OPTIMIZED_DIR": os.path.join(tmpdir, "optimized"),
        "GEAK_DIR": "/tmp/geak",
        "GEAK_OE_DIR": "/tmp/geak_oe",
        "GEAK_MODE": "auto",
        "OPTIMIZE_SCOPE": "all",
        "OPTIMIZE_PRIORITY_THRESHOLD": "0.1",
        "ENV_INFO_FILE": os.path.join(tmpdir, "env_info.json"),
        "RESOURCES_DIR": os.path.join(tmpdir, "resources"),
        "TEMPLATES_DIR": os.path.join(tmpdir, "templates"),
        "REPORT_DIR": os.path.join(tmpdir, "reports"),
        "GPUS": "0,1",
        "DRY_RUN_NOTE": "",
        "ENFORCE_EAGER_FLAG": "",
        "FILTER_EP": "",
        "FILTER_TP": "",
        "FILTER_CONC_START": "",
        "FILTER_CONC_END": "",
        "FILTER_SEQ": "",
    }
    config.update(overrides)
    return config


class TestRunnerUserDecisionFlow:
    def test_v1_runner_enforces_phase7_result_scalars_even_if_monitor_passes(self):
        """Regression for active-gates: V1 must not trust a PASS monitor
        when the phase result says Phase 7 shipped zero winners."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=False)

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "kernel-optimize":
                    result_path = os.path.join(
                        tmpdir, "agent-results", "phase-07-result.md")
                    with open(result_path, "w") as f:
                        f.write(
                            "# Phase 7\n\n"
                            "winners_total_count: 0\n"
                            "claimed_winner_artifacts_valid: false\n"
                            "optimized_artifact_count: 0\n"
                            "redirect_attempted: true\n"
                            "redirect_count_within_tolerance: false\n"
                            "mini_ab_required: true\n"
                            "mini_ab_harness_status: unreliable_high_variance\n"
                            "dispatch_pre_flight_pass: true\n"
                            "library_tests_failed_count: 0\n"
                            "allocator_test_pass: true\n"
                        )
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": "stop_with_blocker",
                    "retry_recommendation": "stop",
                    "root_cause_class": "zero_winners",
                    "key_signal_names": ["winners_total_count"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert state.status == "awaiting_user_instruction"
            assert state.awaiting_user_instruction_phase == "kernel-optimize"
            assert "kernel-optimize" not in state.phases_completed

    def test_v2_runner_enforces_phase7_result_scalars_even_if_monitor_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=True)

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "kernel-optimize":
                    result_path = os.path.join(
                        tmpdir, "agent-results", "phase-07-result.md")
                    with open(result_path, "w") as f:
                        f.write(
                            "winners_total_count: 0\n"
                            "claimed_winner_artifacts_valid: false\n"
                            "optimized_artifact_count: 0\n"
                            "redirect_attempted: true\n"
                            "redirect_count_within_tolerance: false\n"
                            "mini_ab_required: true\n"
                            "mini_ab_harness_status: unreliable_high_variance\n"
                            "dispatch_pre_flight_pass: true\n"
                            "library_tests_failed_count: 0\n"
                            "allocator_test_pass: true\n"
                        )
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": "stop_with_blocker",
                    "retry_recommendation": "stop",
                    "root_cause_class": "zero_winners",
                    "key_signal_names": ["winners_total_count"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert state.status == "awaiting_user_instruction"
            predicate_path = os.path.join(
                tmpdir, "monitor", "phase-07-predicate.json")
            with open(predicate_path) as f:
                predicate = json.load(f)
            assert predicate["verdict"] == "FAIL"
            triggered = {
                d["field"] for d in predicate["details"] if d.get("triggered")
            }
            assert "winners_total_count" in triggered

    def test_rca_stop_with_blocker_pauses_for_user(self):
        """RCA stop_with_blocker on a critical phase pauses the run for
        explicit user instruction, never auto-skips to report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=False)

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": "stop_with_blocker",
                    "analysis": "test stop",
                    "root_cause_class": "test_stop",
                    "key_signal_names": ["s1"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )

            assert state.status == "awaiting_user_instruction"
            assert state.awaiting_user_instruction_phase == "kernel-optimize"
            assert state.terminal_state["outcome"] == "awaiting_user_instruction"
            assert state.terminal_state["rca_stop_recommended"] is True
            assert "report-generate" not in state.phases_completed

            request_path = os.path.join(
                tmpdir, "monitor", "user_decision_request.json"
            )
            assert os.path.isfile(request_path)
            with open(request_path) as f:
                request = json.load(f)
            assert request["phase"] == "kernel-optimize"
            assert request["reason"] == "rca_stop_with_blocker"
            options = {opt["id"] for opt in request["options"]}
            assert {"retry", "stop", "generate_report_anyway"}.issubset(options)

    def test_default_budget_keeps_retrying(self):
        """With the default registry budget (max_per_phase=1000, max_total=10000),
        a phase that fails 5 times in a row continues to retry without
        falling back or failing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=False, MODE="benchmark")
            attempts = {"benchmark": 0}

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "benchmark":
                    attempts[phase_key] += 1
                    if attempts[phase_key] <= 5:
                        return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "benchmark" and attempts[phase_key] <= 5:
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            rca_calls = {"n": 0}

            def rca_fn(phase_key, manifest):
                rca_calls["n"] += 1
                return {
                    "terminal_action": "retry",
                    "root_cause_class": f"signal_{rca_calls['n']}",
                    "key_signal_names": [f"sig_{rca_calls['n']}"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )

            assert state.status == "completed"
            assert state.retry_counts.get("benchmark") == 5
            assert state.fallbacks_used == []

    def test_fallback_only_triggered_with_finite_budget(self):
        """A custom registry with a tiny budget falls back; the default
        budget does not."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            registry["rerun"] = {"max_per_phase": 1, "max_total": 1}
            config = _build_optimize_config(tmpdir, V2_MONITOR=True)
            attempts = {}

            def dispatch_fn(phase_key, handoff_path):
                attempts[phase_key] = attempts.get(phase_key, 0) + 1
                if phase_key == "kernel-optimize" and attempts[phase_key] <= 3:
                    return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "kernel-optimize" and attempts.get(phase_key, 0) <= 3:
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            rca_count = {"n": 0}

            def rca_fn(phase_key, manifest):
                rca_count["n"] += 1
                return {
                    "terminal_action": "retry",
                    "root_cause_class": f"sig_{rca_count['n']}",
                    "key_signal_names": [f"k_{rca_count['n']}"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert any(
                fb.get("phase_key") == "kernel-optimize"
                for fb in state.fallbacks_used
            )

    def test_canonical_rca_retry_recommendation_fallback_is_honored(self):
        """The response policy must honor schema-compliant
        retry_recommendation=fallback (not only legacy
        terminal_action=fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            registry["rerun"] = {"max_per_phase": 1000, "max_total": 10000}
            config = _build_optimize_config(tmpdir, V2_MONITOR=True)

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": None,
                    "retry_recommendation": "fallback",
                    "root_cause_class": "bad_phase6",
                    "key_signal_names": ["source_map_stale"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert any(
                fb.get("phase_key") == "kernel-optimize"
                and fb.get("fallback_target") == "problem-generate"
                for fb in state.fallbacks_used
            )

    def test_canonical_rca_retry_recommendation_stop_pauses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=True)

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": "continue",
                    "retry_recommendation": "stop",
                    "root_cause_class": "unrecoverable",
                    "key_signal_names": ["no_valid_target"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert state.status == "awaiting_user_instruction"
            assert state.awaiting_user_instruction_phase == "kernel-optimize"

    def test_retry_handoff_contains_rca_guidance(self):
        """After a FAIL+RCA retry, the next attempt handoff must include
        RCA guidance, not only a generic attempt count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=False, MODE="benchmark")
            attempts = {"benchmark": 0}

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "benchmark":
                    attempts[phase_key] += 1
                    if attempts[phase_key] == 2:
                        with open(handoff_path) as f:
                            handoff = f.read()
                        assert "## Root Cause Analysis" in handoff
                        assert "increase warmup" in handoff
                    if attempts[phase_key] == 1:
                        return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "benchmark" and attempts[phase_key] == 1:
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": None,
                    "retry_recommendation": "retry_with_changes",
                    "summary": "Harness needs more warmup",
                    "retry_guidance": "increase warmup and stabilize GPU clocks",
                    "root_cause_class": "harness_noise",
                    "key_signal_names": ["variance"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert state.status == "completed"
            assert attempts["benchmark"] == 2

    def test_repeated_fingerprint_without_systemic_callback_pauses(self):
        """Repeated fingerprints must not keep retrying blindly when
        systemic_rca_fn is not wired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=False, MODE="benchmark")

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "benchmark":
                    return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "benchmark":
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": None,
                    "retry_recommendation": "retry_with_changes",
                    "root_cause_class": "same_bug",
                    "key_signal_names": ["same_signal"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
                systemic_rca_fn=None,
            )
            assert state.status == "awaiting_user_instruction"
            assert state.awaiting_user_instruction_phase == "benchmark"
            assert state.total_reruns == 1

    def test_integration_prereq_blocks_empty_optimized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            os.remove(os.path.join(tmpdir, "optimized", "mock_winner.py"))
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=False)

            def dispatch_fn(phase_key, handoff_path):
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {"terminal_action": None, "retry_recommendation": "retry_same"}

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )
            assert state.status == "failed"
            assert "integration" not in state.phases_completed
            failure_path = os.path.join(tmpdir, "runner_failure.json")
            with open(failure_path) as f:
                failure = json.load(f)
            assert failure["phase"] == "integration"
            assert "optimized/ is empty" in failure["message"]

    def test_phase_07_pause_not_skip_to_report(self):
        """Phase 7 FAIL with default budget but stop_with_blocker RCA
        does not promote any phase to phases_completed past
        kernel-optimize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_optimize_workspace(tmpdir)
            registry = _load_registry()
            config = _build_optimize_config(tmpdir, V2_MONITOR=True)

            def dispatch_fn(phase_key, handoff_path):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL"}
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "kernel-optimize":
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest):
                return {
                    "terminal_action": "stop_with_blocker",
                    "root_cause_class": "test",
                    "key_signal_names": ["s"],
                }

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=dispatch_fn,
                monitor_fn=monitor_fn,
                rca_fn=rca_fn,
            )

            assert state.status == "awaiting_user_instruction"
            assert "report-generate" not in state.phases_completed
            assert "integration" not in state.phases_completed


# ---------------------------------------------------------------------------
# Registry hygiene: terminal_policy must not auto-skip to report
# ---------------------------------------------------------------------------


class TestRegistryHygiene:
    def test_no_auto_partial_report_in_critical_phases(self):
        """kernel-optimize and integration must not declare
        terminal_policy=allow_partial_report on the shipped registry."""
        reg = _load_registry()
        for phase_key in ("kernel-optimize", "integration"):
            tp = reg["phases"][phase_key].get("terminal_policy")
            assert tp != "allow_partial_report", (
                f"{phase_key}: allow_partial_report removed; got {tp}"
            )

    def test_kernel_optimize_has_winner_gates(self):
        reg = _load_registry()
        rules = reg["phases"]["kernel-optimize"]["quality"][
            "detection_rules_structured"
        ]
        fields = {r.get("field") for r in rules}
        for required in (
            "winners_total_count",
            "claimed_winner_artifacts_valid",
            "redirect_count_within_tolerance",
            "mini_ab_harness_status",
            "optimized_artifact_count",
        ):
            assert required in fields, (
                f"detection_rules_structured for kernel-optimize must include {required}"
            )

    def test_kernel_optimize_quality_check_uses_winners_total(self):
        reg = _load_registry()
        checks = reg["phases"]["kernel-optimize"]["quality"]["checks"]
        winners_check = next(
            (c for c in checks if c["name"] == "winners_present"), None
        )
        assert winners_check is not None
        # New contract: winners_total_count >= 1, not in_place_winners_count >= 0.
        assert winners_check["field"] == "winners_total_count"
        assert winners_check["min"] == 1
