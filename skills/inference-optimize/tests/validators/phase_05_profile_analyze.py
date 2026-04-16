"""Validator for Phase 05 (profile-analyze) artifacts."""

import json
import os

from . import CheckResult


def validate(output_dir):
    results = []

    profiles_dir = os.path.join(output_dir, "profiles")
    has_profiles = os.path.isdir(profiles_dir)
    results.append(CheckResult(
        phase="profile-analyze",
        name="profiles_dir_exists",
        passed=has_profiles,
        detail=f"profiles/ {'found' if has_profiles else 'missing'}",
    ))

    analysis_path = os.path.join(output_dir, "results", "profile_analysis.json")
    has_analysis = os.path.isfile(analysis_path)
    results.append(CheckResult(
        phase="profile-analyze",
        name="profile_analysis_exists",
        passed=has_analysis,
        detail=f"profile_analysis.json {'found' if has_analysis else 'missing'}",
    ))

    manifest_path = os.path.join(output_dir, "results", "trace_manifest.json")
    has_manifest = os.path.isfile(manifest_path)
    results.append(CheckResult(
        phase="profile-analyze",
        name="trace_manifest_exists",
        passed=has_manifest,
        detail=f"trace_manifest.json {'found' if has_manifest else 'missing'}",
    ))

    if has_manifest:
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            trace_count = manifest.get("trace_count", 0)
            results.append(CheckResult(
                phase="profile-analyze",
                name="trace_count_positive",
                passed=trace_count > 0,
                detail=f"trace_count={trace_count}",
            ))
        except (json.JSONDecodeError, OSError) as e:
            results.append(CheckResult(
                phase="profile-analyze",
                name="trace_manifest_readable",
                passed=False,
                detail=str(e),
            ))

    return results
