"""Tests for fork_upstream.py with mocked git and network."""

import json
import os
import subprocess
import sys
from unittest import mock

import pytest
import yaml

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "optimize", "fork_upstream.py"
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimize"))
import fork_upstream  # noqa: E402  (import after path injection)


def _write_minimal_inputs(tmp_path):
    map_path = tmp_path / "map.yaml"
    yaml.safe_dump({"entries": [
        {"symbol_pattern": "k*", "library": "fla", "source_form": "triton",
         "bucket": "A", "geak_strategy": "in_place_optimize"},
        {"symbol_pattern": "j*", "library": "vllm", "source_form": "hip_cpp",
         "bucket": "A", "geak_strategy": "in_place_optimize"},
    ]}, map_path.open("w"))
    pins_path = tmp_path / "pins.yaml"
    yaml.safe_dump({"pins": [
        {"vllm_version": "v0.19.1", "pins": {"fla": "abc123", "vllm": "def456"}}
    ]}, pins_path.open("w"))
    return map_path, pins_path


class TestLoadHelpers:
    def test_load_pins_returns_per_version_block(self, tmp_path):
        _, pins_path = _write_minimal_inputs(tmp_path)
        pins = fork_upstream.load_pins(str(pins_path), "v0.19.1")
        assert pins == {"fla": "abc123", "vllm": "def456"}

    def test_load_pins_unknown_version_returns_empty(self, tmp_path):
        _, pins_path = _write_minimal_inputs(tmp_path)
        assert fork_upstream.load_pins(str(pins_path), "v9.9.9") == {}

    def test_libraries_referenced_no_manifest_returns_all_libs(self, tmp_path):
        map_path, _ = _write_minimal_inputs(tmp_path)
        entries = fork_upstream.load_map(str(map_path))
        libs = fork_upstream.libraries_referenced(entries, manifest_path=None)
        assert libs == {"fla", "vllm"}

    def test_libraries_referenced_filters_to_optimize_true(self, tmp_path):
        map_path, _ = _write_minimal_inputs(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "optimizations": [
                {"library": "fla", "optimize": True},
                {"library": "vllm", "optimize": False},
            ]
        }))
        libs = fork_upstream.libraries_referenced(
            fork_upstream.load_map(str(map_path)), manifest_path=str(manifest_path)
        )
        assert libs == {"fla"}


class TestProbeCkBranch:
    def test_probe_handles_network_error_gracefully(self):
        with mock.patch("urllib.request.urlopen", side_effect=Exception("network")):
            merged, detail = fork_upstream.probe_ck_branch_merged()
        assert merged is False
        assert detail.startswith("probe_failed")

    def test_probe_returns_true_when_branch_merged(self):
        # ahead_by==0 + behind_by>0 means feature/ck-preprocess-main is fully
        # contained in main (i.e. merged).
        fake = mock.MagicMock()
        fake.read.return_value = json.dumps({"ahead_by": 0, "behind_by": 232}).encode()
        fake.__enter__ = lambda self: self
        fake.__exit__ = lambda self, *a: None
        with mock.patch("urllib.request.urlopen", return_value=fake):
            merged, detail = fork_upstream.probe_ck_branch_merged()
        assert merged is True
        assert "behind=232" in detail


class TestEnd2EndMockedGit:
    def test_main_creates_manifest_with_mocked_git(self, tmp_path, monkeypatch):
        map_path, pins_path = _write_minimal_inputs(tmp_path)
        out_dir = tmp_path / "out"

        called = []

        def fake_run(cmd, cwd=None, check=True, capture=False):
            called.append((cmd, cwd))
            return mock.MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(fork_upstream, "run", fake_run)
        monkeypatch.setattr(
            fork_upstream, "probe_ck_branch_merged", lambda: (False, "probe_skipped")
        )

        # Directly invoke main() by simulating CLI args.
        argv = [
            "fork_upstream.py",
            "--output-dir", str(out_dir),
            "--vllm-version", "v0.19.1",
            "--map", str(map_path),
            "--pins", str(pins_path),
            "--all",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        # Pretend the fork dirs exist so the script skips the clone branch
        # but still runs the fetch/checkout/branch steps.
        os.makedirs(out_dir / "forks" / "fla", exist_ok=True)
        os.makedirs(out_dir / "forks" / "vllm", exist_ok=True)

        rc = fork_upstream.main()
        assert rc == 0

        manifest = json.loads((out_dir / "forks" / "manifest.json").read_text())
        assert set(manifest["forks"].keys()) == {"fla", "vllm"}
        assert manifest["forks"]["fla"]["pinned_commit"] == "abc123"
        assert manifest["forks"]["vllm"]["pinned_commit"] == "def456"
        assert manifest["ck_branch_merged_status"] is False
        assert manifest["vllm_version"] == "v0.19.1"
