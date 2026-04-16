"""Tests for load_optimization_manifest.py -- scope filtering, mode filtering, priority sorting."""

import json
import os
import subprocess
import sys

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "optimize", "load_optimization_manifest.py"
)


def run(manifest_path, geak_mode, optimize_scope="all"):
    cmd = [
        sys.executable, SCRIPT,
        "--manifest", manifest_path,
        "--geak-mode", geak_mode,
        "--optimize-scope", optimize_scope,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def write_manifest(tmpdir, entries):
    path = os.path.join(tmpdir, "optimization_manifest.json")
    with open(path, "w") as f:
        json.dump({"optimizations": entries}, f)
    return path


SAMPLE_ENTRIES = [
    {"name": "fused_rmsnorm", "optimize": True, "enabled": True, "profiling_pct": 5.0,
     "geak_mode": "simple", "type": "fused", "kernel_type": "triton", "priority_score": 10.0},
    {"name": "gemm_bf16", "optimize": True, "enabled": True, "profiling_pct": 8.0,
     "geak_mode": "kernel-url", "type": "gemm", "kernel_type": "hip", "priority_score": 15.0},
    {"name": "skip_this", "optimize": True, "enabled": True, "profiling_pct": 2.0,
     "geak_mode": "skip", "type": "other", "kernel_type": "other", "priority_score": 1.0},
    {"name": "disabled_entry", "optimize": False, "enabled": False, "profiling_pct": 0.5,
     "geak_mode": "simple", "type": "other", "kernel_type": "triton", "priority_score": 0.5},
]


class TestLoadOptimizationManifest:
    def test_basic_load(self, tmp_path):
        """Loads manifest and filters skip entries."""
        path = write_manifest(str(tmp_path), SAMPLE_ENTRIES)
        result = run(path, "full")
        assert result.returncode == 0
        # skip_this should be filtered out
        assert "skip_this" not in result.stdout
        assert "fused_rmsnorm" in result.stdout
        assert "gemm_bf16" in result.stdout

    def test_fused_only_scope(self, tmp_path):
        """fused_only scope filters to only type=fused entries."""
        path = write_manifest(str(tmp_path), SAMPLE_ENTRIES)
        result = run(path, "full", optimize_scope="fused_only")
        assert result.returncode == 0
        assert "fused_rmsnorm" in result.stdout
        assert "gemm_bf16" not in result.stdout

    def test_triton_only_mode(self, tmp_path):
        """triton_only mode filters to simple-mode entries."""
        path = write_manifest(str(tmp_path), SAMPLE_ENTRIES)
        result = run(path, "triton_only")
        assert result.returncode == 0
        assert "fused_rmsnorm" in result.stdout
        assert "gemm_bf16" not in result.stdout

    def test_priority_sorting(self, tmp_path):
        """Entries are sorted descending by priority_score."""
        path = write_manifest(str(tmp_path), SAMPLE_ENTRIES)
        result = run(path, "full")
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        # gemm_bf16 (score=15) should appear before fused_rmsnorm (score=10) in entry lines
        entry_lines = [l for l in lines if "score=" in l]
        if len(entry_lines) >= 2:
            assert entry_lines[0].index("gemm_bf16") < len(entry_lines[0])

    def test_missing_file(self, tmp_path):
        """Non-existent manifest file exits non-zero."""
        result = run(os.path.join(str(tmp_path), "nonexistent.json"), "full")
        assert result.returncode != 0

    def test_geak_mode_override_warning(self, tmp_path):
        """simple mode on non-triton kernel emits warning and overrides to kernel-url."""
        entries = [
            {"name": "hip_kernel", "optimize": True, "enabled": True, "profiling_pct": 5.0,
             "geak_mode": "simple", "type": "other", "kernel_type": "hip", "priority_score": 5.0},
        ]
        path = write_manifest(str(tmp_path), entries)
        result = run(path, "full")
        assert result.returncode == 0
        assert "WARNING" in result.stdout
        assert "kernel-url" in result.stdout
