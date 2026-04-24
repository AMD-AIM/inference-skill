"""Regression tests for generate_optimization_summary.py.

These tests guard the summary generator against schema drift between
producer and consumer of `forks/manifest.json`. The producer
(`scripts/optimize/fork_upstream.py`) writes a top-level dict whose
`"forks"` key is a dict-of-dicts keyed by library name (NOT a list).
A previous bug had the consumer reading `manifest["libraries"]` (an
unknown key) and silently returning zero counts.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SCRIPT = os.path.join(
    REPO_ROOT,
    "skills",
    "inference-optimize",
    "scripts",
    "report",
    "generate_optimization_summary.py",
)


def _producer_shape_manifest():
    """Return a manifest matching scripts/optimize/fork_upstream.py output."""
    return {
        "ck_branch_merged_status": True,
        "ck_branch_merged_detail": "ahead=0 behind=12",
        "vllm_version": "v0.19.1",
        "forks": {
            "fla": {
                "repo_url": "https://github.com/sustcsonglin/flash-linear-attention.git",
                "pinned_commit": "abc1234567890",
                "fork_path": "forks/fla",
                "dirty": False,
                "rebuild_command": "pip install -e .",
            },
            "vllm": {
                "repo_url": "https://github.com/vllm-project/vllm.git",
                "pinned_commit": "def4567890123",
                "fork_path": "forks/vllm",
                "dirty": False,
                "rebuild_command": "pip install -e . --no-build-isolation",
            },
            "aiter": {
                "repo_url": "https://github.com/ROCm/aiter.git",
                "pinned_commit": "0001234567890",
                "fork_path": "forks/aiter",
                "dirty": True,
                "error": "fetch failed",
            },
        },
    }


def _write_workspace(tmpdir, manifest):
    """Lay out the directory tree that generate_optimization_summary.py expects."""
    output_dir = os.path.join(tmpdir, "out")
    results_dir = os.path.join(output_dir, "results")
    forks_dir = os.path.join(output_dir, "forks")
    report_dir = os.path.join(output_dir, "report")
    for d in (results_dir, forks_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    # Minimal comparison so the script exercises a happy path.
    with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
        json.dump({
            "baseline": {"total_token_throughput": 1000.0, "mean_ttft_ms": 100.0},
            "optimized": {"total_token_throughput": 1100.0, "mean_ttft_ms": 95.0},
            "speedup": 1.10,
            "e2e_speedup": 1.10,
            "validated": True,
            "artifacts_valid": True,
            "performance_valid": True,
            "performance_gate": "pass",
            "ttft_regression_pct": -5.0,
        }, f)

    with open(os.path.join(forks_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    return output_dir, results_dir, report_dir


def _run(report_dir, results_dir):
    out_path = os.path.join(report_dir, "optimization_summary.json")
    cmd = [
        sys.executable, SCRIPT,
        "--output", out_path,
        "--config-key", "test-key",
        "--framework", "vllm",
        "--results-dir", results_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    with open(out_path) as f:
        return json.load(f)


class TestForksManifestConsumption:
    def test_producer_shape_yields_nonzero_counts(self):
        """Dict-of-dicts shape (current producer output) must populate counts."""
        manifest = _producer_shape_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            _, results_dir, report_dir = _write_workspace(tmp, manifest)
            summary = _run(report_dir, results_dir)
        assert summary["forks_required_count"] == 3
        # 2 of 3 entries are dirty=False
        assert summary["forks_pinned_count"] == 2
        assert summary["ck_branch_merged_status"] is True

    def test_all_clean_pins_count_matches_required(self):
        manifest = _producer_shape_manifest()
        manifest["forks"]["aiter"]["dirty"] = False
        manifest["forks"]["aiter"].pop("error", None)
        with tempfile.TemporaryDirectory() as tmp:
            _, results_dir, report_dir = _write_workspace(tmp, manifest)
            summary = _run(report_dir, results_dir)
        assert summary["forks_required_count"] == 3
        assert summary["forks_pinned_count"] == 3

    def test_ck_branch_merged_status_read_from_top_level(self):
        manifest = _producer_shape_manifest()
        manifest["ck_branch_merged_status"] = False
        with tempfile.TemporaryDirectory() as tmp:
            _, results_dir, report_dir = _write_workspace(tmp, manifest)
            summary = _run(report_dir, results_dir)
        assert summary["ck_branch_merged_status"] is False

    def test_empty_forks_dict_yields_zero_counts(self):
        manifest = {
            "ck_branch_merged_status": False,
            "vllm_version": "v0.19.1",
            "forks": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            _, results_dir, report_dir = _write_workspace(tmp, manifest)
            summary = _run(report_dir, results_dir)
        assert summary["forks_required_count"] == 0
        assert summary["forks_pinned_count"] == 0
        assert summary["ck_branch_merged_status"] is False


class TestForksManifestSchemaRegression:
    """The producer is contracted to write {"forks": {<lib>: {...}}}.
    A list shape (or any non-dict) is a schema regression and must
    fail loudly via warnings.warn rather than silently zeroing out
    counts -- that was the prior HIGH bug class."""

    def _import_summary_module(self):
        # Import the script as a module so we can call main() in-process
        # and use pytest.warns. The script is not on sys.path normally.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_optimization_summary_under_test", SCRIPT
        )
        mod = importlib.util.module_from_spec(spec)
        # The script imports `integration_outcome` from its own dir.
        sys.path.insert(0, os.path.dirname(SCRIPT))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        return mod

    def test_list_shape_emits_warning_and_zero_counts(self, monkeypatch, tmp_path):
        # Manifest with list-shaped forks (the legacy shape that no
        # current producer writes). Must warn and yield zero counts.
        manifest = {
            "ck_branch_merged_status": False,
            "vllm_version": "v0.19.1",
            "forks": [
                {"library": "fla", "pinned_commit": "abc", "dirty": False},
                {"library": "vllm", "pinned_commit": "def", "dirty": True},
            ],
        }
        output_dir, results_dir, report_dir = _write_workspace(str(tmp_path), manifest)
        out_path = os.path.join(report_dir, "optimization_summary.json")
        mod = self._import_summary_module()

        argv = [
            "generate_optimization_summary.py",
            "--output", out_path,
            "--config-key", "test-key",
            "--framework", "vllm",
            "--results-dir", results_dir,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.warns(UserWarning, match="forks/manifest.json"):
            mod.main()
        with open(out_path) as f:
            summary = json.load(f)
        assert summary["forks_required_count"] == 0
        assert summary["forks_pinned_count"] == 0
