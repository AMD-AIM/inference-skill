"""Tests for hash-based baseline integrity verification."""
import json
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.orchestrate.integrity import (
    compute_file_hash,
    write_baseline_integrity,
    verify_baseline_integrity,
)


class TestIntegrity:
    def test_hash_computed_at_write_verified_at_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            # Create a baseline artifact
            artifact = os.path.join(tmpdir, "results/benchmark_results.json")
            with open(artifact, "w") as f:
                json.dump({"throughput": 100}, f)

            write_baseline_integrity(results_dir, ["results/benchmark_results.json"])
            match, mismatches = verify_baseline_integrity(results_dir)
            assert match is True
            assert mismatches == []

    def test_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            artifact = os.path.join(tmpdir, "results/benchmark_results.json")
            with open(artifact, "w") as f:
                json.dump({"throughput": 100}, f)

            write_baseline_integrity(results_dir, ["results/benchmark_results.json"])

            # Modify the artifact
            with open(artifact, "w") as f:
                json.dump({"throughput": 200}, f)

            match, mismatches = verify_baseline_integrity(results_dir)
            assert match is False
            assert len(mismatches) == 1
            assert mismatches[0]["file"] == "results/benchmark_results.json"

    def test_missing_integrity_file_returns_warn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            match, mismatches = verify_baseline_integrity(results_dir)
            assert match is True  # Missing = backward compat WARN, not FAIL
            assert mismatches == []

    def test_directory_hashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            raw_dir = os.path.join(tmpdir, "results/benchmark_raw")
            os.makedirs(raw_dir)
            with open(os.path.join(raw_dir, "a.json"), "w") as f:
                json.dump({"a": 1}, f)
            with open(os.path.join(raw_dir, "b.json"), "w") as f:
                json.dump({"b": 2}, f)

            write_baseline_integrity(results_dir, ["results/benchmark_raw/"])
            match, _ = verify_baseline_integrity(results_dir)
            assert match is True

    def test_missing_artifact_at_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            artifact = os.path.join(tmpdir, "results/benchmark_results.json")
            with open(artifact, "w") as f:
                json.dump({"throughput": 100}, f)

            write_baseline_integrity(results_dir, ["results/benchmark_results.json"])
            os.remove(artifact)

            match, mismatches = verify_baseline_integrity(results_dir)
            assert match is False
            assert mismatches[0]["actual"] == "missing"
