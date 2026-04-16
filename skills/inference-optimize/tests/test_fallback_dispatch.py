"""Tests for V2 fallback re-dispatch (index-driven loop).

The V1 fallback bug: after budget exhaustion triggers a fallback, the
outer `for` loop advances past the fallback phase — it's never re-dispatched.

The V2 index-driven `while` loop sets `phase_idx` to the fallback target
and continues without incrementing, so the fallback phase IS re-dispatched.
"""
import json
import os
import sys
import pathlib
import tempfile
import pytest

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER_DIR = SKILL_ROOT / "scripts" / "orchestrate"
REGISTRY_PATH = SKILL_ROOT / "orchestrator" / "phase-registry.json"

sys.path.insert(0, str(RUNNER_DIR))

from runner import DeterministicRunner, RunnerState


def _build_config(tmpdir, v2=True):
    return {
        "CONFIG_KEY": "test-config",
        "OUTPUT_DIR": tmpdir,
        "MODE": "optimize",
        "V2_MONITOR": str(v2),
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


def _create_artifacts(tmpdir):
    for d in ["handoff", "agent-results", "monitor", "results", "results/parity",
              "problems", "optimized", "profiles", "scripts", "templates",
              "reports", "resources", "results/gap_analysis"]:
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
    with open(os.path.join(tmpdir, "env_info.json"), "w") as f:
        json.dump({"gpu_arch": "mi300x", "gpu_count": 8}, f)
    with open(os.path.join(tmpdir, "results/sweep_configs.json"), "w") as f:
        json.dump({"framework": "sglang", "precision": "bf16"}, f)
    with open(os.path.join(tmpdir, "results/gap_analysis/gap_analysis.json"), "w") as f:
        json.dump({"top_kernels": ["kernel_a", "kernel_b"]}, f)
    with open(os.path.join(tmpdir, "results/profile_analysis.json"), "w") as f:
        json.dump({"status": "complete"}, f)


class TestV2FallbackDispatch:
    """V2 path re-dispatches from fallback target (unlike V1 which skips it)."""

    def test_fallback_redispatches_target(self):
        """After budget exhaustion + fallback, the fallback phase is dispatched."""
        dispatched_phases = []

        def dispatch_fn(phase_key, handoff_path):
            dispatched_phases.append(phase_key)
            if phase_key == "kernel-optimize" and dispatched_phases.count("kernel-optimize") <= 3:
                return {"verdict": "FAIL", "failure_type": "logic"}
            return {"verdict": "PASS"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, v2=True)
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn)

            # kernel-optimize has fallback_target = problem-generate
            # After 3 FAILs, budget exhausted -> redirect to problem-generate
            # V2 should re-dispatch problem-generate
            assert "problem-generate" in dispatched_phases, (
                f"problem-generate should be re-dispatched after fallback, got: {dispatched_phases}"
            )
            # Count: problem-generate should appear at least twice
            # (once initially, once after fallback)
            pg_count = dispatched_phases.count("problem-generate")
            assert pg_count >= 2, (
                f"Expected problem-generate dispatched >=2 times, got {pg_count}"
            )

    def test_fallback_then_success_completes(self):
        """Fallback re-dispatch -> subsequent success -> pipeline completes."""
        dispatch_count = {}

        def dispatch_fn(phase_key, handoff_path):
            dispatch_count[phase_key] = dispatch_count.get(phase_key, 0) + 1
            # kernel-optimize fails first 3 times, then passes after fallback
            if phase_key == "kernel-optimize":
                if dispatch_count[phase_key] <= 3:
                    return {"verdict": "FAIL", "failure_type": "logic"}
                return {"verdict": "PASS"}
            return {"verdict": "PASS"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, v2=True)
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn)

            assert state.status == "completed", f"Expected completed, got {state.status}"
            assert len(state.fallbacks_used) == 1
            assert state.fallbacks_used[0]["phase_key"] == "kernel-optimize"
            assert state.fallbacks_used[0]["fallback_target"] == "problem-generate"

    def test_fallback_exhausted_aborts(self):
        """If fallback also fails and budget exhausted again, pipeline aborts."""
        def dispatch_fn(phase_key, handoff_path):
            if phase_key in ("kernel-optimize", "problem-generate"):
                return {"verdict": "FAIL", "failure_type": "logic"}
            return {"verdict": "PASS"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, v2=True)
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn)

            # Should eventually fail (budget exhausted after fallback exhausted)
            assert state.status == "failed"

    def test_v1_path_still_has_fallback_bug(self):
        """V1 path (V2_MONITOR=false) still has the known fallback bug.

        This documents the existing behavior. The fallback phase is NOT
        re-dispatched in V1 because the outer for loop advances past it.
        """
        dispatched_phases = []

        def dispatch_fn(phase_key, handoff_path):
            dispatched_phases.append(phase_key)
            if phase_key == "kernel-optimize":
                return {"verdict": "FAIL", "failure_type": "logic"}
            return {"verdict": "PASS"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, v2=False)
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn)

            # In V1, problem-generate appears only once (initial dispatch).
            # The fallback break doesn't re-dispatch it.
            pg_count = dispatched_phases.count("problem-generate")
            assert pg_count == 1, (
                f"V1 should dispatch problem-generate exactly once (bug), got {pg_count}"
            )
