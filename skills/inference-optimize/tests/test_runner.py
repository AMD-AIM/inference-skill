#!/usr/bin/env python3
"""Tests for the deterministic runner.

Run:  python3 -m pytest tests/test_runner.py -v
"""

import json
import os
import pathlib
import sys
import tempfile

import pytest

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER_DIR = SKILL_ROOT / "scripts" / "orchestrate"
REGISTRY_PATH = SKILL_ROOT / "orchestrator" / "phase-registry.json"

sys.path.insert(0, str(RUNNER_DIR))

from runner import (
    DeterministicRunner,
    RunnerState,
    atomic_write_json,
    compute_parity_hash,
    truncate_context,
)


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _minimal_config(mode="benchmark"):
    return {
        "MODE": mode,
        "CONFIG_KEY": "test",
        "OUTPUT_DIR": "/tmp/test",
        "SKIP_INTEGRATION": "false",
    }


def _make_retry_rca_fn():
    """Return RCA stub with changing fingerprints to avoid systemic triggers."""
    call_count = {"value": 0}

    def _rca_fn(phase_key, manifest_dict):
        call_count["value"] += 1
        n = call_count["value"]
        return {
            "terminal_action": "retry",
            "retry_recommendation": "retry",
            "root_cause_class": f"test_rca_{phase_key}",
            "key_signal_names": [f"signal_{n}"],
        }

    return _rca_fn


class TestAtomicWrite:
    def test_atomic_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            atomic_write_json(path, {"key": "value"})
            with open(path) as f:
                data = json.load(f)
            assert data["key"] == "value"

    def test_atomic_write_overwrites(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            atomic_write_json(path, {"v": 1})
            atomic_write_json(path, {"v": 2})
            with open(path) as f:
                data = json.load(f)
            assert data["v"] == 2

    def test_atomic_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "test.json")
            atomic_write_json(path, {"ok": True})
            assert os.path.isfile(path)


class TestTruncateContext:
    def test_no_truncation_needed(self):
        lines = ["a", "b", "c"]
        assert truncate_context(lines, 10) == lines

    def test_truncation_adds_marker(self):
        lines = [str(i) for i in range(100)]
        result = truncate_context(lines, 10)
        assert len(result) == 10
        assert "[truncated:" in result[-1]
        assert "91 lines omitted" in result[-1]

    def test_exact_limit_no_truncation(self):
        lines = ["a", "b", "c"]
        assert truncate_context(lines, 3) == lines

    def test_non_positive_limit_disables_truncation(self):
        lines = [str(i) for i in range(100)]
        assert truncate_context(lines, 0) == lines
        assert truncate_context(lines, -1) == lines


class TestParityHash:
    def test_deterministic(self):
        snapshot = {"a": 1, "b": [2, 3]}
        h1 = compute_parity_hash(snapshot)
        h2 = compute_parity_hash(snapshot)
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_parity_hash({"a": 1})
        h2 = compute_parity_hash({"a": 2})
        assert h1 != h2

    def test_key_order_independent(self):
        h1 = compute_parity_hash({"b": 2, "a": 1})
        h2 = compute_parity_hash({"a": 1, "b": 2})
        assert h1 == h2


class TestRunnerState:
    def test_to_progress_has_required_fields(self):
        state = RunnerState("/tmp/test", "optimize", ["env", "config"])
        progress = state.to_progress()
        required = {"schema_version", "phases_completed", "current_phase",
                     "status", "retry_counts", "total_reruns", "fallbacks_used"}
        assert required.issubset(set(progress.keys()))

    def test_write_and_read_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = RunnerState(tmpdir, "benchmark", ["env", "config"])
            state.phases_completed = ["env"]
            state.current_phase = "config"
            state.write_progress()

            path = os.path.join(tmpdir, "progress.json")
            with open(path) as f:
                data = json.load(f)
            assert data["phases_completed"] == ["env"]
            assert data["current_phase"] == "config"
            assert data["schema_version"] == "1.0"

    def test_parity_manifest_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = RunnerState(tmpdir, "benchmark", ["env", "config"])
            state.phases_completed = ["env", "config"]
            state.status = "completed"
            state.write_parity_manifest()

            manifest_path = os.path.join(tmpdir, "results", "parity", "parity-manifest.json")
            assert os.path.isfile(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert "parity_hash" in manifest
            assert manifest["schema_version"] == "1.0"

    def test_from_progress_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = RunnerState(tmpdir, "optimize", ["env", "config", "benchmark"])
            state.phases_completed = ["env"]
            state.current_phase = "config"
            state.retry_counts = {"env": 0}
            state.total_reruns = 0

            progress = state.to_progress()
            restored = RunnerState.from_progress(tmpdir, progress)
            assert restored.phases_completed == state.phases_completed
            assert restored.current_phase == state.current_phase
            assert restored.mode == state.mode


class TestDeterministicRunner:
    def test_shadow_run_benchmark_mode(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            state = runner.run()

            assert state.status == "completed"
            expected_phases = registry["modes"]["benchmark"]
            assert state.phases_completed == expected_phases

            progress_path = os.path.join(tmpdir, "progress.json")
            assert os.path.isfile(progress_path)

            parity_path = os.path.join(tmpdir, "results", "parity", "parity-manifest.json")
            assert os.path.isfile(parity_path)

    def test_shadow_run_profile_mode(self):
        registry = _load_registry()
        config = _minimal_config("profile")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            state = runner.run()
            assert state.status == "completed"
            assert state.phases_completed == registry["modes"]["profile"]

    def test_handoff_generation(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            runner.run()

            handoff_dir = os.path.join(tmpdir, "handoff")
            assert os.path.isdir(handoff_dir)
            files = os.listdir(handoff_dir)
            assert len(files) == len(registry["modes"]["benchmark"])

    def test_context_budget_high_cap_avoids_practical_truncation(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)

            context = {"KEY_" + str(i): "x" * 100 for i in range(600)}
            handoff = runner.build_handoff("env", context, 1)
            lines = handoff.split("\n")
            assert runner.max_context_lines == 8000
            assert "[truncated:" not in handoff
            assert any("KEY_599" in line for line in lines)

    def test_forced_context_budget_truncates_handoff(self):
        registry = _load_registry()
        registry["max_context_lines"] = 25
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            context = {f"KEY_{i}": "x" * 120 for i in range(120)}
            handoff = runner.build_handoff("env", context, 1)
            lines = handoff.split("\n")
            assert len(lines) == 25
            assert lines[-1].startswith("[truncated:")

    def test_structured_artifact_context_is_compacted(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            large_payload = {f"key_{i}": "x" * 40 for i in range(200)}
            handoff = runner.build_handoff(
                "problem-generate",
                {"TOP_KERNELS": large_payload},
                1,
                context_meta={
                    "TOP_KERNELS": {
                        "resolved_source": "artifact",
                        "path": "results/gap_analysis/gap_analysis.json",
                        "field": "top_kernels",
                    }
                },
            )
            assert '"type":"dict"' in handoff
            assert '"size":200' in handoff
            assert "source=results/gap_analysis/gap_analysis.json#top_kernels" in handoff
            assert "key_199" not in handoff

    def test_dispatch_with_failures(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")
        call_count = {}

        def failing_dispatch(phase_key, handoff_path):
            call_count.setdefault(phase_key, 0)
            call_count[phase_key] += 1
            if phase_key == "benchmark" and call_count[phase_key] <= 1:
                return {"verdict": "FAIL", "attempt": call_count[phase_key]}
            return {"verdict": "PASS", "attempt": call_count[phase_key]}

        def monitor_from_dispatch(phase_key, result_path, summary_path, checks):
            attempts = call_count.get(phase_key, 0)
            if phase_key == "benchmark" and attempts <= 1:
                return {"verdict": "FAIL", "attempt": attempts}
            return {"verdict": "PASS", "attempt": attempts}

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=failing_dispatch,
                monitor_fn=monitor_from_dispatch,
                rca_fn=_make_retry_rca_fn(),
            )

            assert state.status == "completed"
            assert state.retry_counts.get("benchmark") == 1
            assert state.total_reruns == 1

    def test_non_positive_rerun_limits_disable_budget_enforcement(self):
        registry = _load_registry()
        assert registry["rerun"]["max_per_phase"] == 0
        assert registry["rerun"]["max_total"] == 0
        config = _minimal_config("benchmark")
        call_count = {}

        def failing_then_passing_dispatch(phase_key, handoff_path):
            call_count.setdefault(phase_key, 0)
            call_count[phase_key] += 1
            if phase_key == "benchmark" and call_count[phase_key] <= 3:
                return {"verdict": "FAIL", "attempt": call_count[phase_key]}
            return {"verdict": "PASS", "attempt": call_count[phase_key]}

        def monitor_from_dispatch(phase_key, result_path, summary_path, checks):
            attempts = call_count.get(phase_key, 0)
            if phase_key == "benchmark" and attempts <= 3:
                return {"verdict": "FAIL", "attempt": attempts}
            return {"verdict": "PASS", "attempt": attempts}

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=failing_then_passing_dispatch,
                monitor_fn=monitor_from_dispatch,
                rca_fn=_make_retry_rca_fn(),
            )

            assert state.status == "completed"
            assert state.retry_counts.get("benchmark") == 3
            assert state.total_reruns == 3
            assert state.fallbacks_used == []

    def test_monitor_fn_required_for_non_shadow_dispatch(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        def always_pass_dispatch(phase_key, handoff_path):
            return {"verdict": "PASS", "attempt": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=always_pass_dispatch)

            assert state.status == "failed"
            failure_path = os.path.join(tmpdir, "runner_failure.json")
            assert os.path.isfile(failure_path)
            with open(failure_path) as f:
                failure = json.load(f)
            assert failure["error_type"] == "monitor_error"

    def test_rca_fn_required_for_non_shadow_dispatch_when_mode_requires_rca(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        def always_pass_dispatch(phase_key, handoff_path):
            return {"verdict": "PASS", "attempt": 1}

        def always_pass_monitor(phase_key, result_path, summary_path, checks):
            return {"verdict": "PASS", "attempt": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(
                dispatch_fn=always_pass_dispatch,
                monitor_fn=always_pass_monitor,
                rca_fn=None,
            )

            assert state.status == "failed"
            failure_path = os.path.join(tmpdir, "runner_failure.json")
            assert os.path.isfile(failure_path)
            with open(failure_path) as f:
                failure = json.load(f)
            assert failure["error_type"] == "rca_error"

    def test_skip_integration_removes_phase(self):
        registry = _load_registry()
        config = _minimal_config("optimize")
        config["SKIP_INTEGRATION"] = "true"

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            state = runner.run()

            assert "integration" not in state.phases_completed

    def test_optimize_large_context_promotes_warn_to_fail_rca(self):
        registry = _load_registry()
        config = _minimal_config("optimize")
        config["V2_MONITOR"] = "true"

        monitor_calls = []
        monitor_call_counts = {}
        rca_calls = []

        def dispatch_fn(phase_key, handoff_path):
            return {"verdict": "PASS", "phase": phase_key}

        def monitor_fn(phase_key, result_path, summary_path, checks):
            monitor_calls.append(phase_key)
            monitor_call_counts[phase_key] = monitor_call_counts.get(phase_key, 0) + 1
            if phase_key == "kernel-optimize" and monitor_call_counts[phase_key] == 1:
                return {"verdict": "WARN", "failure_type": "logic"}
            return {"verdict": "PASS"}

        def rca_fn(phase_key, manifest_dict):
            rca_calls.append((phase_key, manifest_dict))
            return {"terminal_action": None, "analysis": "advisory"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            os.makedirs(os.path.join(tmpdir, "results", "gap_analysis"), exist_ok=True)
            with open(os.path.join(tmpdir, "env_info.json"), "w") as f:
                json.dump({"gpu_arch": "mi300x"}, f)
            with open(os.path.join(tmpdir, "results", "sweep_configs.json"), "w") as f:
                json.dump({"framework": "sglang", "precision": "bf16"}, f)
            with open(os.path.join(tmpdir, "results", "profile_analysis.json"), "w") as f:
                json.dump({"status": "complete"}, f)
            with open(os.path.join(tmpdir, "results", "gap_analysis", "gap_analysis.json"), "w") as f:
                json.dump({"top_kernels": ["kernel_" + str(i) for i in range(1200)]}, f)

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn, monitor_fn=monitor_fn, rca_fn=rca_fn)

            assert state.status == "completed"
            assert len(monitor_calls) == len(registry["modes"]["optimize"]) + 1
            assert monitor_call_counts.get("kernel-optimize") == 2
            assert len(rca_calls) == 1
            phase_key, manifest = rca_calls[0]
            assert phase_key == "kernel-optimize"
            assert manifest["verdict_severity"] == "FAIL"

            handoff_path = os.path.join(tmpdir, "handoff", "to-phase-06.md")
            with open(handoff_path) as f:
                handoff = f.read()
            assert '"type":"list"' in handoff
            assert "source=results/gap_analysis/gap_analysis.json#top_kernels" in handoff

    def test_progress_is_sole_writer(self):
        """progress.json should only be written by the runner."""
        registry = _load_registry()
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            runner.run()

            progress_path = os.path.join(tmpdir, "progress.json")
            with open(progress_path) as f:
                data = json.load(f)
            assert data["schema_version"] == "1.0"
            assert data["status"] == "completed"

    def test_verdict_sequence_recorded(self):
        registry = _load_registry()
        config = _minimal_config("benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            state = runner.run()

            assert len(state.verdict_sequence) == len(registry["modes"]["benchmark"])
            for v in state.verdict_sequence:
                assert "phase" in v
                assert "attempt" in v
                assert "verdict" in v
