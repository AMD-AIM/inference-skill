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
            assert runner.max_context_lines == 20000
            assert "[truncated:" not in handoff
            assert any("KEY_599" in line for line in lines)

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
            state = runner.run(dispatch_fn=failing_dispatch, monitor_fn=monitor_from_dispatch)

            assert state.status == "completed"
            assert state.retry_counts.get("benchmark") == 1
            assert state.total_reruns == 1

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

    def test_skip_integration_removes_phase(self):
        registry = _load_registry()
        config = _minimal_config("optimize")
        config["SKIP_INTEGRATION"] = "true"

        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            runner = DeterministicRunner(config, registry, tmpdir, shadow=True)
            state = runner.run()

            assert "integration" not in state.phases_completed

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
