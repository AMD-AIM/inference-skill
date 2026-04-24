"""Validator for Phase 08 (integration) artifacts under the
library-rebuild contract."""

import json
import os

from . import CheckResult


def validate(output_dir):
    results = []
    results_dir = os.path.join(output_dir, "results")

    # ------------------------------------------------------------------
    # dispatch_verification.json — the rocprofv3 swap-verification gate
    # ------------------------------------------------------------------
    disp_path = os.path.join(results_dir, "dispatch_verification.json")
    has_disp = os.path.isfile(disp_path)
    results.append(CheckResult(
        phase="integration",
        name="dispatch_verification_exists",
        passed=has_disp,
        detail=f"dispatch_verification.json {'found' if has_disp else 'missing'}",
    ))

    if has_disp:
        try:
            with open(disp_path) as f:
                disp = json.load(f)
            verified = bool(disp.get("dispatch_verified"))
            results.append(CheckResult(
                phase="integration",
                name="dispatch_verified",
                passed=verified,
                detail=f"dispatch_verified={verified}",
            ))
            results.append(CheckResult(
                phase="integration",
                name="expected_symbols_present",
                passed=isinstance(disp.get("expected_symbols"), list),
                detail=f"expected_symbols entries={len(disp.get('expected_symbols') or [])}",
            ))
            results.append(CheckResult(
                phase="integration",
                name="vendor_symbols_present",
                passed=isinstance(disp.get("vendor_symbols"), list),
                detail=f"vendor_symbols entries={len(disp.get('vendor_symbols') or [])}",
            ))
        except (json.JSONDecodeError, OSError) as e:
            results.append(CheckResult(
                phase="integration",
                name="dispatch_verification_readable",
                passed=False,
                detail=str(e),
            ))

    # ------------------------------------------------------------------
    # integration_manifest.json — schema 2.0 (library-rebuild contract)
    # ------------------------------------------------------------------
    manifest_path = os.path.join(results_dir, "integration_manifest.json")
    has_manifest = os.path.isfile(manifest_path)
    results.append(CheckResult(
        phase="integration",
        name="integration_manifest_exists",
        passed=has_manifest,
        detail=f"integration_manifest.json {'found' if has_manifest else 'missing'}",
    ))

    libraries_rebuilt = []
    if has_manifest:
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            libraries_rebuilt = manifest.get("libraries_rebuilt", []) or []
            results.append(CheckResult(
                phase="integration",
                name="libraries_rebuilt_present",
                passed=isinstance(libraries_rebuilt, list),
                detail=f"libraries_rebuilt count={len(libraries_rebuilt)}",
            ))
            for field in ("dispatch_verified", "e2e_ran", "artifacts_valid"):
                results.append(CheckResult(
                    phase="integration",
                    name=f"manifest_{field}_present",
                    passed=field in manifest,
                    detail=f"{field}={manifest.get(field)}",
                ))
        except (json.JSONDecodeError, OSError) as e:
            results.append(CheckResult(
                phase="integration",
                name="integration_manifest_readable",
                passed=False,
                detail=str(e),
            ))

    # ------------------------------------------------------------------
    # rebuild_<lib>.log — one per rebuilt library
    # ------------------------------------------------------------------
    for entry in libraries_rebuilt:
        lib = entry.get("lib") if isinstance(entry, dict) else None
        if not lib:
            continue
        log_path = os.path.join(results_dir, f"rebuild_{lib}.log")
        present = os.path.isfile(log_path)
        results.append(CheckResult(
            phase="integration",
            name=f"rebuild_log_{lib}_exists",
            passed=present,
            detail=f"rebuild_{lib}.log {'found' if present else 'missing'}",
        ))

    # ------------------------------------------------------------------
    # optimization_comparison.json — schema preserved (e2e_speedup,
    # performance_gate, artifacts_valid). Validation logic preserved
    # from the prior contract.
    # ------------------------------------------------------------------
    comp_path = os.path.join(results_dir, "optimization_comparison.json")
    has_comp = os.path.isfile(comp_path)
    results.append(CheckResult(
        phase="integration",
        name="comparison_exists",
        passed=has_comp,
        detail=f"optimization_comparison.json {'found' if has_comp else 'missing'}",
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
