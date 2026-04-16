"""Validator for Phase 07 (kernel-optimize) artifacts."""

import json
import os

from . import CheckResult


def validate(output_dir):
    results = []

    optimized_dir = os.path.join(output_dir, "optimized")
    has_optimized = os.path.isdir(optimized_dir)
    results.append(CheckResult(
        phase="kernel-optimize",
        name="optimized_dir_exists",
        passed=has_optimized,
        detail=f"optimized/ {'found' if has_optimized else 'missing'}",
    ))

    geak_path = os.path.join(output_dir, "problems", "geak_results.json")
    has_geak = os.path.isfile(geak_path)
    results.append(CheckResult(
        phase="kernel-optimize",
        name="geak_results_exists",
        passed=has_geak,
        detail=f"geak_results.json {'found' if has_geak else 'missing'}",
    ))

    if has_geak:
        try:
            with open(geak_path) as f:
                geak_data = json.load(f)
            compiled = sum(1 for r in geak_data if r.get("status") != "error")
            winners = sum(1 for r in geak_data if r.get("speedup", 0) > 1.0)
            results.append(CheckResult(
                phase="kernel-optimize",
                name="compiled_count",
                passed=compiled > 0,
                detail=f"compiled={compiled}, winners={winners}",
            ))
        except (json.JSONDecodeError, OSError) as e:
            results.append(CheckResult(
                phase="kernel-optimize",
                name="geak_results_readable",
                passed=False,
                detail=str(e),
            ))

    return results
