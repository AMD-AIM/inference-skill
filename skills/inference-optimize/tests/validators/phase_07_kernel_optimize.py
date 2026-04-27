"""Validator for Phase 07 (kernel-optimize) artifacts under the
library-rebuild contract.

Validation semantics align with phase-registry.json:

- Count only shipped winners (rows whose ``winner_strategy`` does not
  start with ``not_a_winner_``). A row's ``geak_strategy`` describes
  intent (``in_place_optimize`` / ``dispatch_redirect_*`` /
  ``in_place_optimize_no_harness``); whether it actually shipped is
  tracked by ``winner_strategy`` plus per-row signals such as
  ``py_export_shipped`` and ``redirect_count_within_tolerance``.
- ``redirect_commits_applied_count`` is audit-only and never counts as
  a winner on its own.
- If a winner is claimed but its required integration artifact under
  ``optimized/`` (or the redirect manifest entry) is missing, the
  validator reports failure.
"""

import json
import os

from . import CheckResult


_NOT_A_WINNER_PREFIX = "not_a_winner_"


def _records(geak_data):
    if isinstance(geak_data, dict):
        return geak_data.get("kernels") or []
    if isinstance(geak_data, list):
        return geak_data
    return []


def _is_shipped(record):
    """A row is shipped only when its winner_strategy does not signal
    an intentional non-shipment (``not_a_winner_*``). A missing
    ``winner_strategy`` falls back to legacy heuristics."""
    ws = str(record.get("winner_strategy", "")).strip()
    if ws.startswith(_NOT_A_WINNER_PREFIX):
        return False
    if ws:
        return True
    # Legacy fallback: shipped if an in-place strategy ran and reported
    # speedup>1. Redirects are intentionally excluded from this
    # fallback: a redirect commit without an explicit winner_strategy is
    # audit-only and must not be promoted to a shipped winner.
    strategy = record.get("geak_strategy")
    speedup = record.get("geak_speedup_lib_bench") or 0
    if strategy == "in_place_optimize" and speedup > 1.0:
        return True
    if strategy == "in_place_optimize_no_harness" and speedup > 1.0:
        return True
    return False


def _count_optimized_artifacts(output_dir):
    optimized_dir = os.path.join(output_dir, "optimized")
    if not os.path.isdir(optimized_dir):
        return 0, True
    count = 0
    for root, _dirs, files in os.walk(optimized_dir):
        for name in files:
            if name.startswith("."):
                continue
            count += 1
    return count, count == 0


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

    artifact_count, optimized_dir_empty = _count_optimized_artifacts(output_dir)

    if not has_geak:
        return results

    try:
        with open(geak_path) as f:
            geak_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        results.append(CheckResult(
            phase="kernel-optimize",
            name="geak_results_readable",
            passed=False,
            detail=str(e),
        ))
        return results

    records = _records(geak_data)
    in_place = 0
    redirect_winners = 0
    no_harness = 0
    invalid_artifact_rows = []
    redirect_attempted = False
    redirect_count_failures = []
    for r in records:
        strategy = r.get("geak_strategy")
        shipped = _is_shipped(r)
        if str(strategy or "").startswith("dispatch_redirect_"):
            redirect_attempted = True
            within = r.get("redirect_count_within_tolerance")
            if shipped and within is not True:
                redirect_count_failures.append(r.get("name", "?"))
        if not shipped:
            continue
        if strategy == "in_place_optimize":
            in_place += 1
        elif strategy == "in_place_optimize_no_harness":
            no_harness += 1
        elif str(strategy or "").startswith("dispatch_redirect_"):
            redirect_winners += 1
        # claimed_winner_artifacts_valid is computed per row when
        # available; otherwise we fall back to the optimized/ count.
        per_row_valid = r.get("claimed_winner_artifact_valid")
        if per_row_valid is False:
            invalid_artifact_rows.append(r.get("name", "?"))

    total_winners = in_place + redirect_winners + no_harness

    results.append(CheckResult(
        phase="kernel-optimize",
        name="winners_present",
        passed=total_winners > 0,
        detail=(
            f"winners_total_count={total_winners} "
            f"(in_place={in_place} redirect_winners={redirect_winners} "
            f"no_harness={no_harness})"
        ),
    ))

    results.append(CheckResult(
        phase="kernel-optimize",
        name="optimized_artifact_count",
        passed=not (total_winners > 0 and artifact_count == 0),
        detail=(
            f"optimized_artifact_count={artifact_count} "
            f"optimized_dir_empty={optimized_dir_empty} "
            f"winners_total_count={total_winners}"
        ),
    ))

    results.append(CheckResult(
        phase="kernel-optimize",
        name="redirect_count_within_tolerance",
        passed=len(redirect_count_failures) == 0,
        detail=(
            "all shipped redirects within tolerance"
            if not redirect_count_failures
            else f"out of tolerance: {redirect_count_failures}"
        ),
    ))

    results.append(CheckResult(
        phase="kernel-optimize",
        name="claimed_winner_artifacts_valid",
        passed=len(invalid_artifact_rows) == 0,
        detail=(
            "all shipped winner artifacts valid"
            if not invalid_artifact_rows
            else f"invalid artifacts: {invalid_artifact_rows}"
        ),
    ))

    return results
