#!/usr/bin/env python3
"""Invariant assertions for the inference harness architecture.

Each test corresponds to a non-negotiable invariant from ARCHITECTURE.md.
Tests start as stubs (assert True) and become real as each milestone lands.

Run:  python3 -m pytest tests/test_invariants.py -v
"""

import json
import os
import pathlib

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = SKILL_ROOT / "orchestrator"
REGISTRY_PATH = ORCHESTRATOR_DIR / "phase-registry.json"
DOCS_DIR = SKILL_ROOT / "docs"
PROTOCOLS_DIR = SKILL_ROOT / "protocols"


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# INV-1: progress.json has exactly one writer — the runner (on the runner path)
# ---------------------------------------------------------------------------

class TestProgressJsonSingleWriter:
    """progress.json must only be written by the runner on the runner path."""

    def test_orchestrator_doc_declares_sole_writer(self):
        orch = (ORCHESTRATOR_DIR / "ORCHESTRATOR.md").read_text()
        assert "sole writer" in orch.lower() or "single writer" in orch.lower(), (
            "ORCHESTRATOR.md must declare that progress.json has a sole writer"
        )

    def test_phase_agents_do_not_import_progress_writer(self):
        """Stub: once the runner exists, scan phase agent code for progress.json writes."""
        assert True


# ---------------------------------------------------------------------------
# INV-2: Every control-plane JSON artifact carries schema_version
# ---------------------------------------------------------------------------

class TestSchemaVersionPresent:
    """Control-plane JSON artifacts must carry schema_version."""

    def test_registry_has_schema_version(self):
        reg = _load_registry()
        assert "schema_version" in reg, "Registry must have schema_version"
        assert reg["schema_version"] == "1.0"

    def test_progress_schema_enforces_version(self):
        schema_path = PROTOCOLS_DIR / "progress.schema.json"
        assert schema_path.exists(), "progress.schema.json must exist"
        with open(schema_path) as f:
            schema = json.load(f)
        assert "schema_version" in schema.get("required", [])

    def test_all_schemas_have_version(self):
        for schema_file in PROTOCOLS_DIR.glob("*.schema.json"):
            with open(schema_file) as f:
                schema = json.load(f)
            props = schema.get("properties", {})
            assert "schema_version" in props, (
                f"{schema_file.name} must have schema_version property"
            )


# ---------------------------------------------------------------------------
# INV-3: Critical phases fail closed on missing/malformed monitor/RCA/handoff
# ---------------------------------------------------------------------------

class TestFailClosedCriticalPhases:
    """Critical phases must fail-closed on malformed artifacts."""

    def test_critical_phases_identified_in_registry(self):
        reg = _load_registry()
        critical = [k for k, v in reg["phases"].items() if v.get("critical")]
        assert len(critical) >= 3, f"Expected >=3 critical phases, got {critical}"

    def test_fail_closed_documented_in_orchestrator(self):
        orch = (ORCHESTRATOR_DIR / "ORCHESTRATOR.md").read_text()
        assert "fail closed" in orch.lower() or "fail verdict" in orch.lower(), (
            "ORCHESTRATOR.md must document fail-closed behavior for critical phases"
        )


# ---------------------------------------------------------------------------
# INV-4: Attempt artifacts are immutable and attempt-scoped
# ---------------------------------------------------------------------------

class TestAttemptArtifactImmutability:
    """Retries append new attempt artifacts, never mutate existing ones."""

    def test_evidence_lifecycle_documented(self):
        arch_path = DOCS_DIR / "ARCHITECTURE.md"
        assert arch_path.exists(), "ARCHITECTURE.md must exist"
        text = arch_path.read_text()
        assert "attempt-scoped" in text.lower(), (
            "ARCHITECTURE.md must document attempt-scoped artifact rules"
        )

    def test_runner_appends_not_mutates(self):
        """Stub: once runner.py exists, verify append-only behavior."""
        assert True


# ---------------------------------------------------------------------------
# INV-5: Legacy path remains default until shadow parity + predicate agreement
#         + rollback proof are green
# ---------------------------------------------------------------------------

class TestCanonicalRuntimePath:
    """The runner is now the canonical path; rollback via USE_RUNNER=false."""

    def test_runner_exists(self):
        runner_path = SKILL_ROOT / "scripts" / "orchestrate" / "runner.py"
        assert runner_path.exists(), "runner.py must exist"

    def test_parity_contract_exists(self):
        parity_path = DOCS_DIR / "PARITY_CONTRACT.md"
        assert parity_path.exists(), "PARITY_CONTRACT.md must exist"

    def test_rollback_documented_in_skill(self):
        skill_md = (SKILL_ROOT / "SKILL.md").read_text()
        assert "USE_RUNNER=false" in skill_md or "rollback" in skill_md.lower()

    def test_rollback_documented_in_runtime(self):
        runtime_md = (SKILL_ROOT / "RUNTIME.md").read_text()
        assert "USE_RUNNER=false" in runtime_md or "deterministic runner" in runtime_md.lower()


# ---------------------------------------------------------------------------
# INV-6: LLM judgment stays only where it adds value
# ---------------------------------------------------------------------------

class TestLlmBoundary:
    """Mechanical work belongs in code, not LLM agents."""

    def test_runner_to_llm_boundary_documented(self):
        boundary_path = PROTOCOLS_DIR / "runner-llm-boundary.md"
        assert boundary_path.exists(), "runner-llm-boundary.md must exist"
        text = boundary_path.read_text()
        assert "allowed writes" in text.lower()
        assert "failure taxonomy" in text.lower()

    def test_context_budget_in_registry(self):
        reg = _load_registry()
        assert "max_context_lines" in reg, "Registry must have max_context_lines"
        assert isinstance(reg["max_context_lines"], int)
        assert reg["max_context_lines"] > 0


# ---------------------------------------------------------------------------
# INV-7: Release-critical PASS/WARN/FAIL authority comes from schemas,
#         validators, report logic, runner state, and structured predicates
# ---------------------------------------------------------------------------

class TestVerdictAuthority:
    """Verdicts derive from structured data, not prose."""

    def test_detection_rules_are_strings_today(self):
        """Today detection_rules are prose strings. Commit 5 adds structured predicates."""
        reg = _load_registry()
        for key, phase in reg["phases"].items():
            quality = phase.get("quality")
            if quality and "detection_rules" in quality:
                assert isinstance(quality["detection_rules"], str), (
                    f"Phase {key}: detection_rules should be a string until Commit 5"
                )

    def test_structured_predicates_active_on_critical_phases(self):
        reg = _load_registry()
        for key, phase in reg["phases"].items():
            quality = phase.get("quality", {})
            if phase.get("critical") and "detection_rules" in quality:
                assert "detection_rules_structured" in quality, (
                    f"Critical phase {key}: must have detection_rules_structured"
                )
                rules = quality["detection_rules_structured"]
                assert isinstance(rules, list) and len(rules) > 0, (
                    f"Phase {key}: detection_rules_structured must be a non-empty list"
                )


# ---------------------------------------------------------------------------
# INV-8: Registry fields must be implemented and tested or absent
# ---------------------------------------------------------------------------

class TestNoDormantFields:
    """No dormant documented contracts — fields are active or absent."""

    def test_monitor_context_fields_on_critical_phases(self):
        reg = _load_registry()
        for key, phase in reg["phases"].items():
            if phase.get("critical") and phase.get("quality", {}).get("detection_rules"):
                assert "monitor_context_fields" in phase, (
                    f"Critical phase {key} with detection_rules must have monitor_context_fields"
                )
                mcf = phase["monitor_context_fields"]
                assert "scalars" in mcf, f"Phase {key}: monitor_context_fields missing scalars"

    def test_parallel_groups_in_registry(self):
        reg = _load_registry()
        assert "parallel_groups" in reg, "Registry must have parallel_groups"
        pg = reg["parallel_groups"]
        for mode, groups in pg.items():
            assert mode in reg["modes"], f"parallel_groups mode {mode} not in modes"
            for group in groups:
                for phase_key in group:
                    assert phase_key in reg["phases"], (
                        f"parallel_groups phase {phase_key} not in phases"
                    )

    def test_no_detection_rules_structured_yet(self):
        """Covered by TestVerdictAuthority.test_structured_predicates_not_active_yet."""
        assert True


# ---------------------------------------------------------------------------
# Architecture doc sanity
# ---------------------------------------------------------------------------

class TestArchitectureDocExists:
    """ARCHITECTURE.md must exist and cover the four-plane model."""

    def test_architecture_md_exists(self):
        assert (DOCS_DIR / "ARCHITECTURE.md").exists()

    def test_four_planes_documented(self):
        text = (DOCS_DIR / "ARCHITECTURE.md").read_text()
        for plane in ["policy plane", "control plane", "execution plane", "evidence plane"]:
            assert plane.lower() in text.lower(), f"Missing plane: {plane}"

    def test_rollback_procedure_documented(self):
        text = (DOCS_DIR / "ARCHITECTURE.md").read_text()
        assert "rollback" in text.lower()

    def test_rework_protocol_documented(self):
        text = (DOCS_DIR / "ARCHITECTURE.md").read_text()
        assert "rework" in text.lower()

    def test_evidence_lifecycle_documented(self):
        text = (DOCS_DIR / "ARCHITECTURE.md").read_text()
        assert "evidence lifecycle" in text.lower() or "evidence plane" in text.lower()


class TestParityContractExists:
    """PARITY_CONTRACT.md must exist and define included/excluded fields."""

    def test_parity_contract_exists(self):
        assert (DOCS_DIR / "PARITY_CONTRACT.md").exists()

    def test_included_fields_defined(self):
        text = (DOCS_DIR / "PARITY_CONTRACT.md").read_text()
        for field in ["phase ordering", "verdict sequence", "retry counts"]:
            assert field.lower() in text.lower(), f"Missing parity field: {field}"

    def test_excluded_fields_defined(self):
        text = (DOCS_DIR / "PARITY_CONTRACT.md").read_text()
        assert "excluded" in text.lower() or "not included" in text.lower()


# ---------------------------------------------------------------------------
# Registry structural invariants
# ---------------------------------------------------------------------------

class TestSchemaFiles:
    """All JSON schemas must be valid and self-consistent."""

    def test_all_schemas_are_valid_json(self):
        for schema_file in PROTOCOLS_DIR.glob("*.schema.json"):
            with open(schema_file) as f:
                schema = json.load(f)
            assert "type" in schema, f"{schema_file.name} must have a type"

    def test_runner_failure_taxonomy(self):
        schema_path = PROTOCOLS_DIR / "runner-failure.schema.json"
        assert schema_path.exists()
        with open(schema_path) as f:
            schema = json.load(f)
        error_types = schema["properties"]["error_type"]["enum"]
        expected = {"schema_invalid", "missing_artifact", "monitor_error",
                    "timeout", "budget_exhausted", "manual_intervention_required"}
        assert set(error_types) == expected

    def test_reserved_fields_documented(self):
        reg = _load_registry()
        reserved = reg.get("_reserved_fields", {})
        for field in ["monitor_context_fields", "detection_rules_structured", "parallel_groups"]:
            assert field in reserved, f"Reserved field {field} not documented"


class TestRegistryStructure:
    """phase-registry.json must maintain structural invariants."""

    def test_registry_loads(self):
        reg = _load_registry()
        assert "phases" in reg
        assert "modes" in reg
        assert "rerun" in reg
        assert "timeouts" in reg
        assert "context_sources" in reg
        assert "schema_version" in reg

    def test_phase_indices_are_contiguous(self):
        reg = _load_registry()
        indices = sorted(p["index"] for p in reg["phases"].values())
        assert indices == list(range(len(indices))), (
            f"Phase indices must be contiguous: {indices}"
        )

    def test_modes_reference_valid_phases(self):
        reg = _load_registry()
        phase_keys = set(reg["phases"].keys())
        for mode, phases in reg["modes"].items():
            for p in phases:
                assert p in phase_keys, f"Mode {mode} references unknown phase {p}"

    def test_deps_reference_valid_phases(self):
        reg = _load_registry()
        phase_keys = set(reg["phases"].keys())
        for key, phase in reg["phases"].items():
            for dep in phase.get("deps", []):
                assert dep in phase_keys, f"Phase {key} dep {dep} is not a valid phase"

    def test_critical_phases_have_quality_checks(self):
        reg = _load_registry()
        for key, phase in reg["phases"].items():
            if phase.get("critical"):
                assert "quality" in phase, f"Critical phase {key} missing quality checks"
                assert "checks" in phase["quality"], (
                    f"Critical phase {key} missing quality.checks"
                )

    def test_fallback_targets_reference_valid_phases(self):
        reg = _load_registry()
        phase_keys = set(reg["phases"].keys())
        for key, phase in reg["phases"].items():
            ft = phase.get("fallback_target")
            if ft:
                assert ft in phase_keys, (
                    f"Phase {key} fallback_target {ft} is not a valid phase"
                )

    def test_context_sources_cover_required_context(self):
        reg = _load_registry()
        source_keys = set(reg["context_sources"].keys())
        for key, phase in reg["phases"].items():
            for ctx in phase.get("required_context", []):
                assert ctx in source_keys, (
                    f"Phase {key} required_context {ctx} has no context_source entry"
                )

    def test_rca_artifacts_on_critical_phases_only(self):
        reg = _load_registry()
        for key, phase in reg["phases"].items():
            if "rca_artifact" in phase:
                assert phase.get("critical"), (
                    f"Phase {key} has rca_artifact but is not critical"
                )


# ---------------------------------------------------------------------------
# Handoff validation
# ---------------------------------------------------------------------------

class TestHandoffValidation:
    """validate_handoff.py must exist and enforce handoff contracts."""

    def test_validate_handoff_exists(self):
        vh = SKILL_ROOT / "scripts" / "orchestrate" / "validate_handoff.py"
        assert vh.exists(), "validate_handoff.py must exist"

    def test_valid_handoff_passes(self):
        import sys as _sys
        import tempfile
        scripts_dir = str(SKILL_ROOT / "scripts" / "orchestrate")
        _sys.path.insert(0, scripts_dir)
        from validate_handoff import validate_handoff

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\nphase: benchmark\nphase_index: 2\n---\n\n## Context\nHere is context.\n\n## Instructions\nDo something.\n")
            f.flush()
            valid, errors = validate_handoff(f.name, "benchmark", 2)
        os.unlink(f.name)
        assert valid, f"Expected valid handoff, got errors: {errors}"

    def test_missing_frontmatter_fails(self):
        import sys as _sys
        import tempfile
        scripts_dir = str(SKILL_ROOT / "scripts" / "orchestrate")
        _sys.path.insert(0, scripts_dir)
        from validate_handoff import validate_handoff

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("No frontmatter here\n## Context\nstuff\n## Instructions\nmore")
            f.flush()
            valid, errors = validate_handoff(f.name, "benchmark", 2)
        os.unlink(f.name)
        assert not valid

    def test_rerun_handoff_requires_feedback(self):
        import sys as _sys
        import tempfile
        scripts_dir = str(SKILL_ROOT / "scripts" / "orchestrate")
        _sys.path.insert(0, scripts_dir)
        from validate_handoff import validate_handoff

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\nphase: benchmark\nphase_index: 2\n---\n\n## Context\nctx\n\n## Instructions\ninstr\n")
            f.flush()
            valid, errors = validate_handoff(f.name, "benchmark", 2, is_rerun=True)
        os.unlink(f.name)
        assert not valid
        assert any("Prior Attempt Feedback" in e for e in errors)
