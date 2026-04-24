#!/usr/bin/env python3
"""Fixture-based tests for validate_optimization.py and integration_outcome.py.

Run:  python3 -m pytest tests/test_validate_optimization.py -v
  or: python3 tests/test_validate_optimization.py
"""

import json
import os
import subprocess
import sys
import tempfile

REPORT_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "report",
)
VALIDATE_OPTIMIZATION = os.path.join(REPORT_SCRIPTS_DIR, "validate_optimization.py")

sys.path.insert(0, REPORT_SCRIPTS_DIR)

from integration_outcome import (
    SCHEMA_VERSION,
    SEVERE_TTFT_REGRESSION_PCT,
    SPEEDUP_PASS_THRESHOLD,
    SPEEDUP_WARN_THRESHOLD,
    derive_fields,
    performance_gate,
    pipeline_status,
)
from validate_optimization import build_comparison, select_result_pair


def _write_benchmark_json(directory, filename, throughput, ttft_ms=100.0):
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump({
            "total_token_throughput": throughput,
            "mean_ttft_ms": ttft_ms,
            "mean_itl_ms": 5.0,
            "mean_tpot_ms": 10.0,
            "duration": 60,
        }, f)
    return path


def _make_pair(tmpdir, bl_tps, opt_tps, bl_ttft=100.0, opt_ttft=100.0):
    _write_benchmark_json(tmpdir, "run_fp4_tp8.json", bl_tps, bl_ttft)
    _write_benchmark_json(tmpdir, "optimized_run_fp4_tp8.json", opt_tps, opt_ttft)
    bl, opt, errors = select_result_pair(tmpdir)
    assert not errors, errors
    return build_comparison(bl, opt)


def _run_validate_cli(tmpdir):
    return subprocess.run(
        [sys.executable, VALIDATE_OPTIMIZATION, "--results-dir", tmpdir],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# integration_outcome unit tests
# ---------------------------------------------------------------------------

def test_gate_pass():
    gate, upgraded = performance_gate(1.05)
    assert gate == "pass"
    assert not upgraded


def test_gate_warn():
    gate, upgraded = performance_gate(0.98)
    assert gate == "warn"
    assert not upgraded


def test_gate_fail():
    gate, upgraded = performance_gate(0.90)
    assert gate == "fail"
    assert not upgraded


def test_gate_none_speedup():
    gate, upgraded = performance_gate(None)
    assert gate == "fail"
    assert not upgraded


def test_gate_ttft_upgrade():
    gate, upgraded = performance_gate(0.98, ttft_regression_pct=25.0)
    assert gate == "fail"
    assert upgraded


def test_gate_ttft_no_upgrade_when_pass():
    gate, upgraded = performance_gate(1.05, ttft_regression_pct=25.0)
    assert gate == "pass"
    assert not upgraded


def test_derive_fields_pass():
    fields = derive_fields(1.05, True)
    assert fields["performance_gate"] == "pass"
    assert fields["performance_valid"] is True
    assert fields["validated"] is True


def test_derive_fields_warn():
    fields = derive_fields(0.98, True)
    assert fields["performance_gate"] == "warn"
    assert fields["performance_valid"] is False
    assert fields["validated"] is False


def test_derive_fields_artifacts_invalid():
    fields = derive_fields(1.05, False)
    assert fields["performance_gate"] == "pass"
    assert fields["validated"] is False


def test_pipeline_status_no_blockers():
    assert pipeline_status([], "pass") == "completed"
    assert pipeline_status([], "warn") == "completed with warnings"
    assert pipeline_status([], "fail") == "completed with blockers"
    assert pipeline_status([], None) == "completed"


def test_pipeline_status_with_blockers():
    early = [{"phase": "benchmark", "summary": "x", "terminal_action": "stop"}]
    assert pipeline_status(early, "pass") == "pipeline incomplete"
    late = [{"phase": "integration", "summary": "x", "terminal_action": "retry"}]
    assert pipeline_status(late, "pass") == "completed with blockers"


def test_pipeline_status_integration_expected_but_missing():
    assert pipeline_status([], None, integration_expected=True) == "pipeline incomplete"
    assert pipeline_status([], None, integration_expected=False) == "completed"
    assert pipeline_status([], None) == "completed"
    assert pipeline_status([], "pass", integration_expected=True) == "completed"
    assert pipeline_status([], "fail", integration_expected=True) == "completed with blockers"


def test_pipeline_status_integration_skipped():
    assert pipeline_status(
        [], None, integration_expected=True, integration_skipped=True,
    ) == "completed"
    assert pipeline_status(
        [], None, integration_expected=False, integration_skipped=True,
    ) == "completed"


# ---------------------------------------------------------------------------
# validate_optimization fixture tests
# ---------------------------------------------------------------------------

def test_zero_baseline_throughput():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 0, 500)
        assert comp["artifacts_valid"] is True
        assert comp["speedup"] is None
        assert comp["performance_gate"] == "fail"
        assert comp["validated"] is False


def test_zero_optimized_throughput():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 1000, 0)
        assert comp["artifacts_valid"] is True
        assert comp["performance_gate"] == "fail"


def test_warn_band_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 1000, 980)
        assert comp["artifacts_valid"] is True
        assert comp["performance_gate"] == "warn"
        assert comp["performance_valid"] is False
        assert comp["validated"] is False
        assert comp["speedup"] == 0.98


def test_fail_band_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 1000, 900)
        assert comp["artifacts_valid"] is True
        assert comp["performance_gate"] == "fail"
        assert comp["speedup"] == 0.9


def test_pass_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 1000, 1100)
        assert comp["artifacts_valid"] is True
        assert comp["performance_gate"] == "pass"
        assert comp["performance_valid"] is True
        assert comp["validated"] is True
        assert comp["speedup"] == 1.1


def test_severe_ttft_regression():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 1000, 980, bl_ttft=100.0, opt_ttft=125.0)
        assert comp["ttft_regression_pct"] == 25.0
        assert comp["performance_gate"] == "fail"
        assert comp["ttft_upgraded"] is True


def test_validate_cli_warn_band_exits_successfully():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_benchmark_json(tmpdir, "run_fp4_tp8.json", 1000)
        _write_benchmark_json(tmpdir, "optimized_run_fp4_tp8.json", 980)

        result = _run_validate_cli(tmpdir)

        assert result.returncode == 0
        assert "VALIDATION COMPLETED (warn band)" in result.stdout

        with open(os.path.join(tmpdir, "optimization_comparison.json")) as f:
            comparison = json.load(f)
        assert comparison["performance_gate"] == "warn"


def test_validate_cli_fail_band_exits_nonzero():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_benchmark_json(tmpdir, "run_fp4_tp8.json", 1000)
        _write_benchmark_json(tmpdir, "optimized_run_fp4_tp8.json", 900)

        result = _run_validate_cli(tmpdir)

        assert result.returncode == 1
        assert "VALIDATION FAILED (performance regression)" in result.stdout

        with open(os.path.join(tmpdir, "optimization_comparison.json")) as f:
            comparison = json.load(f)
        assert comparison["performance_gate"] == "fail"


def test_schema_version_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = _make_pair(tmpdir, 1000, 1100)
        assert comp["schema_version"] == SCHEMA_VERSION


def test_corrupt_json_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "corrupt.json")
        with open(corrupt_path, "w") as f:
            f.write("{bad json")
        _write_benchmark_json(tmpdir, "run_fp4_tp8.json", 1000)
        _write_benchmark_json(tmpdir, "optimized_run_fp4_tp8.json", 1100)
        bl, opt, errors = select_result_pair(tmpdir)
        assert not errors
        comp = build_comparison(bl, opt)
        assert comp["artifacts_valid"] is True


def test_fallback_baseline_selection():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_benchmark_json(tmpdir, "some_other_name.json", 1000)
        _write_benchmark_json(tmpdir, "optimized_mismatched.json", 1100)
        summary = {"results": [{"file": "some_other_name.json"}]}
        with open(os.path.join(tmpdir, "benchmark_summary.json"), "w") as f:
            json.dump(summary, f)
        bl, opt, errors = select_result_pair(tmpdir)
        assert not errors
        assert bl["name"] == "some_other_name.json"


def test_no_matching_baseline():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_benchmark_json(tmpdir, "unrelated_baseline.json", 1000)
        _write_benchmark_json(tmpdir, "optimized_mismatched.json", 1100)
        bl, opt, errors = select_result_pair(tmpdir)
        assert errors
        assert bl is None


# ---------------------------------------------------------------------------
# Summary generation tests
# ---------------------------------------------------------------------------

GENERATE_SCRIPT = os.path.join(REPORT_SCRIPTS_DIR, "generate_optimization_summary.py")


def _run_generate_summary_with_skip(output_dir, results_dir):
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "optimization_summary.json")
    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--output", out_path,
        "--config-key", "test-key",
        "--framework", "sglang",
        "--results-dir", results_dir,
        "--skip-integration",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"generate_summary failed: {result.stderr}"
    with open(out_path) as f:
        return json.load(f)


def _run_generate_summary(output_dir, results_dir, problems_dir=""):
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "optimization_summary.json")
    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--output", out_path,
        "--config-key", "test-key",
        "--framework", "sglang",
        "--results-dir", results_dir,
    ]
    if problems_dir:
        cmd.extend(["--problems-dir", problems_dir])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"generate_summary failed: {result.stderr}"
    with open(out_path) as f:
        return json.load(f)


def test_summary_warn_band_no_blockers():
    """Warn band without blockers → completed with warnings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        comp = _make_pair(results_dir, 1000, 980)
        with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
            json.dump(comp, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["pipeline_status"] == "completed with warnings"
        assert summary["all_phases_completed"] is True
        assert summary["performance_gate"] == "warn"


def test_summary_fail_band_no_blockers():
    """Fail band without blockers → completed with blockers (fail gate drives it)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        comp = _make_pair(results_dir, 1000, 900)
        with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
            json.dump(comp, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["pipeline_status"] == "completed with blockers"
        assert summary["all_phases_completed"] is False
        assert summary["performance_gate"] == "fail"


def test_summary_blocker_driven_incomplete():
    """Early-phase blocker → pipeline incomplete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        blockers = {"blockers": [
            {"phase": "benchmark", "summary": "harness crash", "terminal_action": "stop"}
        ]}
        with open(os.path.join(results_dir, "pipeline_blockers.json"), "w") as f:
            json.dump(blockers, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["pipeline_status"] == "pipeline incomplete"
        assert summary["all_phases_completed"] is False
        assert summary["blocker_count"] == 1


def test_summary_late_blocker_with_pass_gate():
    """Late-phase blocker + pass gate → completed with blockers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        comp = _make_pair(results_dir, 1000, 1100)
        with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
            json.dump(comp, f)
        blockers = {"blockers": [
            {"phase": "integration", "summary": "adapter overhead", "terminal_action": "retry"}
        ]}
        with open(os.path.join(results_dir, "pipeline_blockers.json"), "w") as f:
            json.dump(blockers, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["pipeline_status"] == "completed with blockers"
        assert summary["all_phases_completed"] is False


def test_summary_missing_comparison_with_results_dir():
    """Results dir provided but no comparison → pipeline incomplete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["pipeline_status"] == "pipeline incomplete"
        assert summary["all_phases_completed"] is False


def test_summary_clean_pass():
    """Clean pass → completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        comp = _make_pair(results_dir, 1000, 1100)
        with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
            json.dump(comp, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["pipeline_status"] == "completed"
        assert summary["all_phases_completed"] is True
        assert summary["performance_gate"] == "pass"


def test_summary_integration_manifest_ingestion():
    """Integration manifest (schema 2.0) and dispatch verification fields are surfaced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        comp = _make_pair(results_dir, 1000, 1100)
        with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
            json.dump(comp, f)
        manifest = {
            "schema_version": "2.0",
            "libraries_rebuilt": [
                {"lib": "fla", "commit": "abc1234", "install_log_path": "results/rebuild_fla.log"},
                {"lib": "vllm", "commit": "def5678", "install_log_path": "results/rebuild_vllm.log"},
            ],
            "dispatch_verified": True,
            "e2e_ran": True,
            "artifacts_valid": True,
        }
        with open(os.path.join(results_dir, "integration_manifest.json"), "w") as f:
            json.dump(manifest, f)
        dispatch = {
            "mode": "post-rebuild",
            "expected_symbol_total_count": 12,
            "vendor_symbol_leaked_count": 0,
            "redirect_required_count": 1,
            "redirect_honored_count": 1,
            "dispatch_verified": True,
        }
        with open(os.path.join(results_dir, "dispatch_verification.json"), "w") as f:
            json.dump(dispatch, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert summary["integration_schema_version"] == "2.0"
        assert summary["libraries_rebuilt_count"] == 2
        assert summary["dispatch_verified"] is True
        assert summary["e2e_ran"] is True
        assert summary["expected_symbol_total_count"] == 12
        assert summary["vendor_symbol_leaked_count"] == 0
        assert summary["redirect_required_count"] == 1
        assert summary["redirect_honored_count"] == 1


def test_summary_skip_integration_completed():
    """SKIP_INTEGRATION=true with no comparison → completed (not incomplete)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        summary = _run_generate_summary_with_skip(tmpdir, results_dir)
        assert summary["pipeline_status"] == "completed"
        assert summary["all_phases_completed"] is True


def test_summary_no_phases_completed_field():
    """Summary should use all_phases_completed, not phases_completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        comp = _make_pair(results_dir, 1000, 1100)
        with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
            json.dump(comp, f)
        summary = _run_generate_summary(tmpdir, results_dir)
        assert "all_phases_completed" in summary
        assert "phases_completed" not in summary


# ---------------------------------------------------------------------------
# Run with plain python3 (no pytest required)
# ---------------------------------------------------------------------------

def _run_all():
    import inspect
    passed = 0
    failed = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(func):
            try:
                func()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
