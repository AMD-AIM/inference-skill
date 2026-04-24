"""Validator for Phase 07 (kernel-optimize) artifacts under the
library-rebuild contract."""

import json
import os

from . import CheckResult


def _records(geak_data):
    if isinstance(geak_data, dict):
        return geak_data.get("kernels") or []
    if isinstance(geak_data, list):
        return geak_data
    return []


def validate(output_dir):
    results = []

    forks_dir = os.path.join(output_dir, "forks")
    has_forks = os.path.isdir(forks_dir)
    results.append(CheckResult(
        phase="kernel-optimize",
        name="forks_dir_exists",
        passed=has_forks,
        detail=f"forks/ {'found' if has_forks else 'missing'}",
    ))

    geak_path = os.path.join(output_dir, "problems", "geak_results.json")
    has_geak = os.path.isfile(geak_path)
    results.append(CheckResult(
        phase="kernel-optimize",
        name="geak_results_exists",
        passed=has_geak,
        detail=f"geak_results.json {'found' if has_geak else 'missing'}",
    ))

    preflight_path = os.path.join(output_dir, "results", "preflight_dispatch_trace.json")
    has_preflight = os.path.isfile(preflight_path)
    results.append(CheckResult(
        phase="kernel-optimize",
        name="preflight_dispatch_trace_exists",
        passed=has_preflight,
        detail=f"preflight_dispatch_trace.json {'found' if has_preflight else 'missing'}",
    ))

    if has_geak:
        try:
            with open(geak_path) as f:
                geak_data = json.load(f)
            records = _records(geak_data)
            in_place = sum(
                1 for r in records
                if r.get("geak_strategy") == "in_place_optimize"
                and (r.get("geak_speedup_lib_bench") or 0) > 1.0
            )
            redirects = sum(
                1 for r in records
                if str(r.get("geak_strategy", "")).startswith("dispatch_redirect_")
            )
            no_harness = sum(
                1 for r in records
                if r.get("geak_strategy") == "in_place_optimize_no_harness"
                and (r.get("geak_speedup_lib_bench") or 0) > 1.0
            )
            unverified = sum(
                1 for r in records if r.get("optimization_unverified_per_kernel") is True
            )
            total_winners = in_place + redirects + no_harness
            results.append(CheckResult(
                phase="kernel-optimize",
                name="winners_present",
                passed=total_winners > 0,
                detail=(
                    f"in_place={in_place} redirect_commits={redirects} "
                    f"no_harness={no_harness} unverified_per_kernel={unverified}"
                ),
            ))
        except (json.JSONDecodeError, OSError) as e:
            results.append(CheckResult(
                phase="kernel-optimize",
                name="geak_results_readable",
                passed=False,
                detail=str(e),
            ))

    return results
