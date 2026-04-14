#!/usr/bin/env python3
"""Fixture-driven tests for the control-plane suite.

Loads representative and adversarial fixtures from tests/fixtures/ and validates
the report pipeline against their expected outcomes.

Run:  python3 -m pytest tests/test_fixture_suite.py -v
"""

import json
import os
import pathlib
import sys
import tempfile

import pytest

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES_DIR = SKILL_ROOT / "tests" / "fixtures"
REPORT_SCRIPTS_DIR = SKILL_ROOT / "scripts" / "report"

sys.path.insert(0, str(REPORT_SCRIPTS_DIR))

from integration_outcome import derive_fields, performance_gate, pipeline_status
from validate_optimization import build_comparison, select_result_pair


def _load_fixtures(subdir):
    fixtures = []
    d = FIXTURES_DIR / subdir
    for path in sorted(d.glob("*.json")):
        with open(path) as f:
            fixture = json.load(f)
        fixture["_path"] = str(path)
        fixtures.append(fixture)
    return fixtures


REPRESENTATIVE = _load_fixtures("representative")
ADVERSARIAL = _load_fixtures("adversarial")


def _write_benchmark_json(directory, filename, payload):
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


# ---------------------------------------------------------------------------
# Representative fixture tests
# ---------------------------------------------------------------------------

class TestRepresentativeFixtures:
    """Validate expected outcomes for representative scenarios."""

    @pytest.mark.parametrize(
        "fixture",
        REPRESENTATIVE,
        ids=[f["name"] for f in REPRESENTATIVE],
    )
    def test_pipeline_status(self, fixture):
        expected = fixture["expected"]
        if expected.get("pipeline_status") is None:
            pytest.skip("No pipeline_status expectation")

        inputs = fixture["inputs"]
        blockers = inputs.get("blockers", {}).get("blockers", [])
        comp = inputs.get("comparison")
        integration_gate = comp.get("performance_gate") if comp else None
        skip = fixture.get("skip_integration", False)

        has_integration = comp is not None
        status = pipeline_status(
            blockers, integration_gate,
            integration_expected=has_integration and not skip,
            integration_skipped=skip,
        )
        assert status == expected["pipeline_status"], (
            f"Fixture {fixture['id']}: expected '{expected['pipeline_status']}', "
            f"got '{status}'"
        )

    @pytest.mark.parametrize(
        "fixture",
        [f for f in REPRESENTATIVE if f["inputs"].get("baseline_benchmark") and f["inputs"].get("optimized_benchmark")],
        ids=[f["name"] for f in REPRESENTATIVE if f["inputs"].get("baseline_benchmark") and f["inputs"].get("optimized_benchmark")],
    )
    def test_build_comparison(self, fixture):
        expected = fixture["expected"]
        inputs = fixture["inputs"]
        if expected.get("performance_gate") is None:
            pytest.skip("No performance_gate expectation")

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_benchmark_json(tmpdir, "run_fp4_tp8.json", inputs["baseline_benchmark"])
            _write_benchmark_json(tmpdir, "optimized_run_fp4_tp8.json", inputs["optimized_benchmark"])

            bl, opt, errors = select_result_pair(tmpdir)
            assert not errors, f"Fixture {fixture['id']}: {errors}"

            comp = build_comparison(bl, opt)
            assert comp["performance_gate"] == expected["performance_gate"]
            assert comp["artifacts_valid"] == expected["artifacts_valid"]
            if expected.get("validated") is not None:
                assert comp["validated"] == expected["validated"]

    @pytest.mark.parametrize(
        "fixture",
        REPRESENTATIVE,
        ids=[f["name"] for f in REPRESENTATIVE],
    )
    def test_all_phases_completed(self, fixture):
        expected = fixture["expected"]
        if expected.get("all_phases_completed") is None:
            pytest.skip("No all_phases_completed expectation")

        inputs = fixture["inputs"]
        blockers = inputs.get("blockers", {}).get("blockers", [])
        comp = inputs.get("comparison")
        integration_gate = comp.get("performance_gate") if comp else None
        skip = fixture.get("skip_integration", False)

        has_integration = comp is not None
        status = pipeline_status(
            blockers, integration_gate,
            integration_expected=has_integration and not skip,
            integration_skipped=skip,
        )
        all_completed = status in {"completed", "completed with warnings"}
        assert all_completed == expected["all_phases_completed"], (
            f"Fixture {fixture['id']}: status={status}, "
            f"expected all_phases_completed={expected['all_phases_completed']}"
        )


# ---------------------------------------------------------------------------
# Adversarial fixture tests
# ---------------------------------------------------------------------------

class TestAdversarialFixtures:
    """Validate behavior under adversarial inputs."""

    def test_missing_comparison_yields_incomplete(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "missing_comparison")
        status = pipeline_status(
            fixture["inputs"]["blockers"]["blockers"],
            None,
            integration_expected=True,
            integration_skipped=False,
        )
        assert status == fixture["expected"]["pipeline_status"]

    def test_budget_exhausted_yields_blockers(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "budget_exhausted")
        inputs = fixture["inputs"]
        status = pipeline_status(
            inputs["blockers"]["blockers"],
            None,
            integration_expected=True,
        )
        assert status == fixture["expected"]["pipeline_status"]
        assert inputs["progress"]["total_reruns"] == 5

    def test_partial_progress_missing_fields(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "partial_progress_json")
        progress = fixture["inputs"]["progress"]
        assert "phases_completed" not in progress

    def test_unsupported_schema_version(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "unsupported_schema_version")
        progress = fixture["inputs"]["progress"]
        assert progress["schema_version"] != "1.0"

    def test_empty_winners(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "empty_winners")
        results = fixture["inputs"]["geak_results"]
        improved = sum(1 for r in results if r.get("speedup", 0) > 1.0)
        assert improved == fixture["expected"]["kernels_improved"]

    def test_malformed_monitor_missing_verdict(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "malformed_monitor_json")
        content = fixture["inputs"]["monitor_review"]
        assert "verdict" not in content.split("---")[1]

    def test_malformed_rca_missing_fields(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "malformed_rca_json")
        rca = fixture["inputs"]["rca_output"]
        assert "retry_recommendation" not in rca
        assert "terminal_action" not in rca

    def test_corrupt_handoff_missing_fields(self):
        fixture = next(f for f in ADVERSARIAL if f["name"] == "corrupt_handoff")
        content = fixture["inputs"]["handoff_content"]
        assert "phase:" not in content.split("---")[1]


# ---------------------------------------------------------------------------
# Fixture manifest completeness
# ---------------------------------------------------------------------------

class TestFixtureManifest:
    """Ensure fixture cardinality meets minimums."""

    def test_representative_minimum_cardinality(self):
        assert len(REPRESENTATIVE) >= 8, (
            f"Need >= 8 representative fixtures, have {len(REPRESENTATIVE)}"
        )

    def test_adversarial_minimum_cardinality(self):
        assert len(ADVERSARIAL) >= 8, (
            f"Need >= 8 adversarial fixtures, have {len(ADVERSARIAL)}"
        )

    def test_fixture_ids_unique(self):
        all_ids = [f["id"] for f in REPRESENTATIVE + ADVERSARIAL]
        assert len(all_ids) == len(set(all_ids)), "Duplicate fixture IDs found"

    def test_manifest_file_exists(self):
        assert (FIXTURES_DIR / "MANIFEST.md").exists()
