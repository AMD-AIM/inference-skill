"""Tests for library_test_driver.py.

Uses real subprocess invocation against a synthetic shell command rather
than mocking pytest, because the parser is the unit-under-test.
"""

import json
import os
import subprocess
import sys

import pytest
import yaml

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "optimize", "library_test_driver.py"
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimize"))
import library_test_driver  # noqa: E402


def _make_map(tmp_path, library_test_command="echo '5 passed in 1.23s'", bucket="A",
              library_test_path="tests/x.py"):
    path = tmp_path / "map.yaml"
    yaml.safe_dump({
        "entries": [
            {
                "symbol_pattern": "kernel_*",
                "library": "demo",
                "source_form": "triton",
                "bucket": bucket,
                "geak_strategy": "in_place_optimize",
                "library_test_path": library_test_path,
                "library_test_command": library_test_command,
            }
        ]
    }, path.open("w"))
    return path


def _run(symbol, fork, map_path, log_dir):
    return subprocess.run(
        [
            sys.executable, SCRIPT,
            "--kernel", symbol,
            "--fork", str(fork),
            "--map", str(map_path),
            "--log-dir", str(log_dir),
        ],
        capture_output=True,
        text=True,
    )


class TestParser:
    def test_parses_passed_failed_skipped(self):
        out = "==== 12 passed, 3 failed, 1 skipped in 4.5s ===="
        c = library_test_driver.parse_pytest_counts(out)
        assert c == {"passed": 12, "failed": 3, "skipped": 1, "errors": 0}

    def test_parses_passed_only(self):
        out = "==== 7 passed in 1.0s ===="
        c = library_test_driver.parse_pytest_counts(out)
        assert c["passed"] == 7
        assert c["failed"] == 0

    def test_parses_errors(self):
        out = "==== 1 failed, 2 errors in 0.5s ===="
        c = library_test_driver.parse_pytest_counts(out)
        assert c["failed"] == 1
        assert c["errors"] == 2


class TestEnd2End:
    def test_runs_library_command_and_parses_counts(self, tmp_path):
        # Echo a faux pytest summary so the parser produces non-zero counts.
        cmd = "echo '==== 4 passed, 1 skipped in 0.10s ===='"
        map_path = _make_map(tmp_path, library_test_command=cmd)
        result = _run("kernel_alpha", tmp_path, map_path, tmp_path / "logs")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["ran"] is True
        assert payload["pass_count"] == 4
        assert payload["fail_count"] == 0
        assert payload["skipped_count"] == 1
        assert payload["log_path"].endswith("library_test_kernel_alpha.log")

    def test_skips_bucket_b_kernels(self, tmp_path):
        map_path = _make_map(tmp_path, bucket="B", library_test_path=None,
                             library_test_command="echo 'should not run'")
        result = _run("kernel_alpha", tmp_path, map_path, tmp_path / "logs")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["ran"] is False
        assert payload["skip_reason"] == "no_harness_by_design"
        assert payload["pass_count"] is None
        assert payload["fail_count"] is None
        assert payload["skipped_count"] is None

    def test_unknown_kernel_returns_no_entry(self, tmp_path):
        map_path = _make_map(tmp_path)
        result = _run("totally_unknown", tmp_path, map_path, tmp_path / "logs")
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["ran"] is False
        assert payload["skip_reason"] == "no_kernel_source_map_entry"

    def test_failed_command_propagates_returncode(self, tmp_path):
        # `false` exits non-zero and emits no pytest summary.
        map_path = _make_map(tmp_path, library_test_command="false")
        result = _run("kernel_alpha", tmp_path, map_path, tmp_path / "logs")
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["ran"] is True
        assert payload["returncode"] != 0
