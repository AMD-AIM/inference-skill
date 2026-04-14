"""Validator for Phase 08 (integration) artifacts."""

import json
import os

from . import CheckResult


def validate(output_dir):
    results = []
    results_dir = os.path.join(output_dir, "results")

    comp_path = os.path.join(results_dir, "optimization_comparison.json")
    has_comp = os.path.isfile(comp_path)
    results.append(CheckResult(
        phase="integration",
        name="comparison_exists",
        passed=has_comp,
        detail=f"optimization_comparison.json {'found' if has_comp else 'missing'}",
    ))

    manifest_path = os.path.join(results_dir, "integration_manifest.json")
    has_manifest = os.path.isfile(manifest_path)
    results.append(CheckResult(
        phase="integration",
        name="integration_manifest_exists",
        passed=has_manifest,
        detail=f"integration_manifest.json {'found' if has_manifest else 'missing'}",
    ))

    if has_comp:
        try:
            with open(comp_path) as f:
                comp = json.load(f)
            results.append(CheckResult(
                phase="integration",
                name="artifacts_valid",
                passed=comp.get("artifacts_valid", False),
                detail=f"artifacts_valid={comp.get('artifacts_valid')}",
            ))
            gate = comp.get("performance_gate", "unknown")
            results.append(CheckResult(
                phase="integration",
                name="performance_gate",
                passed=gate in ("pass", "warn"),
                detail=f"performance_gate={gate}",
                severity="error" if gate == "fail" else "warning",
            ))
            speedup = comp.get("e2e_speedup") or comp.get("speedup")
            results.append(CheckResult(
                phase="integration",
                name="speedup_computed",
                passed=speedup is not None,
                detail=f"e2e_speedup={speedup}",
            ))
        except (json.JSONDecodeError, OSError) as e:
            results.append(CheckResult(
                phase="integration",
                name="comparison_readable",
                passed=False,
                detail=str(e),
            ))

    return results
