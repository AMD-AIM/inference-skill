"""Tests for signal-and-resume escalation."""
import json
import os
import sys
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.orchestrate.escalation import (
    build_escalation_context,
    write_escalation_request,
    read_escalation_response,
    is_escalation_stale,
    ESCALATION_WARN_SECONDS,
    ESCALATION_ABORT_SECONDS,
)


class TestEscalation:
    def test_context_includes_required_fields(self):
        ctx = build_escalation_context(
            phase_key="kernel-optimize",
            verdict="FAIL",
            failure_type="logic",
        )
        assert ctx["phase"] == "kernel-optimize"
        assert ctx["verdict"] == "FAIL"
        assert ctx["failure_type"] == "logic"
        assert "requested_at" in ctx
        assert "suggested_actions" in ctx
        assert "timeline" in ctx

    def test_write_and_read_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = build_escalation_context("kernel-optimize", "FAIL", "logic")
            path = write_escalation_request(tmpdir, "kernel-optimize", ctx)
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert data["phase"] == "kernel-optimize"

    def test_read_response_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor_dir = os.path.join(tmpdir, "monitor")
            os.makedirs(monitor_dir)
            resp = {"action": "retry", "notes": "fixed the bug"}
            with open(os.path.join(monitor_dir, "escalation-response-phase-kernel-optimize.json"), "w") as f:
                json.dump(resp, f)

            result = read_escalation_response(tmpdir, "kernel-optimize")
            assert result["action"] == "retry"
            assert result["notes"] == "fixed the bug"

    def test_read_response_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = read_escalation_response(tmpdir, "kernel-optimize")
            assert result is None

    def test_stale_escalation(self):
        ctx = build_escalation_context("kernel-optimize", "FAIL", "logic")
        # Set requested_at to 20 min ago
        old_time = datetime.now(timezone.utc) - timedelta(minutes=20)
        ctx["requested_at"] = old_time.isoformat()
        assert is_escalation_stale(ctx) is True

    def test_fresh_escalation_not_stale(self):
        ctx = build_escalation_context("kernel-optimize", "FAIL", "logic")
        assert is_escalation_stale(ctx) is False

    def test_timeline_values(self):
        ctx = build_escalation_context("kernel-optimize", "FAIL", "logic")
        assert ctx["timeline"]["warn_seconds"] == ESCALATION_WARN_SECONDS
        assert ctx["timeline"]["abort_seconds"] == ESCALATION_ABORT_SECONDS
