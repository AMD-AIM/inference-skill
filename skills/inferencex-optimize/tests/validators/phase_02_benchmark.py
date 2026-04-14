"""Validator for Phase 02 (benchmark) artifacts."""

import json
import os

from . import CheckResult


def validate(output_dir):
    results = []
    results_dir = os.path.join(output_dir, "results")

    has_results = os.path.isdir(results_dir)
    results.append(CheckResult(
        phase="benchmark",
        name="results_dir_exists",
        passed=has_results,
        detail=f"results/ {'found' if has_results else 'missing'}",
    ))

    json_files = []
    if has_results:
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    results.append(CheckResult(
        phase="benchmark",
        name="benchmark_jsons_present",
        passed=len(json_files) > 0,
        detail=f"Found {len(json_files)} JSON files in results/",
    ))

    for jf in json_files:
        path = os.path.join(results_dir, jf)
        try:
            with open(path) as f:
                data = json.load(f)
            valid = isinstance(data, dict)
        except (json.JSONDecodeError, OSError):
            valid = False
        results.append(CheckResult(
            phase="benchmark",
            name=f"json_valid_{jf}",
            passed=valid,
            detail=f"{jf}: {'valid' if valid else 'invalid or unreadable'}",
        ))

    return results
