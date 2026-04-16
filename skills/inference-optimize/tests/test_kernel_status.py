"""Tests for JSONL kernel status tracking."""
import json
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.orchestrate.kernel_status import (
    append_kernel_event,
    replay_kernel_status,
    get_kernel_summary,
)


class TestKernelStatus:
    def test_truncated_last_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel-status.jsonl")
            append_kernel_event(path, "k1", "completed", 0, speedup=1.5)
            # Append a truncated line
            with open(path, "a") as f:
                f.write('{"kernel_id": "k2", "event_type": "compl')

            state = replay_kernel_status(path)
            assert "k1" in state
            assert "k2" not in state

    def test_event_replay_builds_correct_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel-status.jsonl")
            append_kernel_event(path, "k1", "started", 0)
            append_kernel_event(path, "k1", "completed", 0, speedup=1.5)
            append_kernel_event(path, "k2", "started", 0)
            append_kernel_event(path, "k2", "failed", 0, error="OOM")

            state = replay_kernel_status(path)
            assert state["k1"]["completed"]["speedup"] == 1.5
            assert state["k2"]["failed"]["error"] == "OOM"

    def test_replay_after_redirect(self):
        """Higher run_attempt overwrites lower."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel-status.jsonl")
            append_kernel_event(path, "k1", "completed", 0, speedup=1.2)
            append_kernel_event(path, "k1", "completed", 1, speedup=1.8)

            state = replay_kernel_status(path)
            assert state["k1"]["completed"]["speedup"] == 1.8
            assert state["k1"]["completed"]["run_attempt"] == 1

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel-status.jsonl")
            append_kernel_event(path, "k1", "completed", 0, speedup=1.5)
            append_kernel_event(path, "k2", "completed", 0, speedup=2.0)
            append_kernel_event(path, "k3", "failed", 0, error="OOM")
            append_kernel_event(path, "k4", "skipped", 0)

            summary = get_kernel_summary(path)
            assert summary["total"] == 4
            assert summary["completed"] == 2
            assert summary["failed"] == 1
            assert summary["skipped"] == 1
            assert summary["best_speedup"] == 2.0

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel-status.jsonl")
            state = replay_kernel_status(path)
            assert state == {}
