#!/usr/bin/env python3
"""Compare baseline and optimized benchmark results to validate optimization.

Usage: python3 validate_optimization.py --results-dir <dir>
Outputs optimization_comparison.json with speedup metrics.
"""

import argparse
import glob
import json
import os
import sys

from integration_outcome import (
    SCHEMA_VERSION,
    derive_fields,
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_throughput(payload):
    return payload.get("total_token_throughput", payload.get("output_throughput"))


def discover_benchmark_results(results_dir):
    """Return benchmark-like JSON payloads keyed by filename."""
    candidates = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable JSON: {os.path.basename(path)} ({exc})")
            continue

        if not isinstance(payload, dict):
            continue
        if get_throughput(payload) is None:
            continue

        candidates.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "payload": payload,
            }
        )

    return candidates


def iter_strings(payload):
    if isinstance(payload, dict):
        for value in payload.values():
            yield from iter_strings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_strings(item)
    elif isinstance(payload, str):
        yield payload


def pick_summary_baseline(results_dir, baseline_by_name):
    """Best-effort fallback when exact optimized_->baseline pairing is missing."""
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    if not os.path.isfile(summary_path):
        return None

    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return None

    matches = []
    for text in iter_strings(summary):
        basename = os.path.basename(text)
        if basename in baseline_by_name:
            matches.append(basename)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return baseline_by_name[unique_matches[0]]
    return None


def select_result_pair(results_dir):
    candidates = discover_benchmark_results(results_dir)
    baseline_by_name = {
        candidate["name"]: candidate
        for candidate in candidates
        if not candidate["name"].startswith("optimized_")
    }
    optimized_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["name"].startswith("optimized_")
        ),
        key=lambda candidate: os.path.getmtime(candidate["path"]),
        reverse=True,
    )

    errors = []
    if not baseline_by_name:
        errors.append("No baseline benchmark result found")
    if not optimized_candidates:
        errors.append(
            "No optimized benchmark result found — the patched server may have failed"
        )
    if errors:
        return None, None, errors

    for optimized in optimized_candidates:
        expected_baseline = optimized["name"][len("optimized_"):]
        baseline = baseline_by_name.get(expected_baseline)
        if baseline:
            return baseline, optimized, []

    fallback_baseline = pick_summary_baseline(results_dir, baseline_by_name)
    if fallback_baseline:
        return fallback_baseline, optimized_candidates[0], []

    expected = ", ".join(
        optimized["name"][len("optimized_"):] for optimized in optimized_candidates
    )
    available = ", ".join(sorted(baseline_by_name)) or "<none>"
    return None, None, [
        "No baseline benchmark result matches the optimized artifact set. "
        f"Expected one of: {expected}. Available baselines: {available}"
    ]


def detect_no_improvements(problems_dir):
    """Return True when geak_results.json exists but no kernel has speedup > 1.0."""
    if not problems_dir:
        return False
    results_path = os.path.join(problems_dir, "geak_results.json")
    if not os.path.isfile(results_path):
        return False
    try:
        raw = load_json(results_path)
    except (OSError, json.JSONDecodeError):
        return False
    if isinstance(raw, dict):
        results = raw.get("kernels", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return False
    if not results:
        return True
    return all(r.get("speedup", 0) <= 1.0 for r in results)


def build_comparison(baseline_entry, optimized_entry, no_improvements=False):
    """Build the full comparison dict.  This is the single source of truth
    for E2E health — every downstream consumer reads this JSON."""
    baseline = baseline_entry["payload"]
    optimized = optimized_entry["payload"]

    bl_tps = get_throughput(baseline) or 0
    opt_tps = get_throughput(optimized) or 0

    # artifacts_valid = files loaded and parsed without error (guaranteed by
    # reaching this point).  Throughput quality is a performance concern, not
    # an artifact-validity concern.
    artifacts_valid = True
    speedup = (opt_tps / bl_tps) if bl_tps > 0 else None

    bl_ttft = baseline.get("mean_ttft_ms", 0)
    opt_ttft = optimized.get("mean_ttft_ms", 0)
    ttft_regression_pct = None
    if bl_ttft and bl_ttft > 0 and opt_ttft and opt_ttft > 0:
        ttft_regression_pct = round((opt_ttft - bl_ttft) / bl_ttft * 100, 2)

    fields = derive_fields(speedup, artifacts_valid, ttft_regression_pct,
                           no_improvements=no_improvements)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts_valid": artifacts_valid,
        "performance_valid": fields["performance_valid"],
        "validated": fields["validated"],
        "performance_gate": fields["performance_gate"],
        "ttft_upgraded": fields["ttft_upgraded"],
        "baseline_file": baseline_entry["name"],
        "optimized_file": optimized_entry["name"],
        "baseline": {
            "total_token_throughput": bl_tps,
            "mean_ttft_ms": bl_ttft,
            "mean_itl_ms": baseline.get("mean_itl_ms", 0),
            "mean_tpot_ms": baseline.get("mean_tpot_ms", 0),
            "duration_s": baseline.get("duration", 0),
        },
        "optimized": {
            "total_token_throughput": opt_tps,
            "mean_ttft_ms": opt_ttft,
            "mean_itl_ms": optimized.get("mean_itl_ms", 0),
            "mean_tpot_ms": optimized.get("mean_tpot_ms", 0),
            "duration_s": optimized.get("duration", 0),
        },
        "speedup": round(speedup, 4) if speedup is not None else None,
        "e2e_speedup": round(speedup, 4) if speedup is not None else None,
        "ttft_regression_pct": ttft_regression_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate optimization results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--problems-dir", default="")
    args = parser.parse_args()

    results_dir = args.results_dir
    no_improvements = detect_no_improvements(args.problems_dir)

    baseline_entry, optimized_entry, errors = select_result_pair(results_dir)

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    comparison = build_comparison(baseline_entry, optimized_entry, no_improvements)

    gate = comparison["performance_gate"]
    speedup = comparison["speedup"]
    bl_tps = comparison["baseline"]["total_token_throughput"]
    opt_tps = comparison["optimized"]["total_token_throughput"]
    exit_code = 0

    if speedup is None:
        print("VALIDATION FAILED (cannot compute speedup — zero baseline throughput)")
        exit_code = 1
    elif gate == "pass":
        print("VALIDATION PASSED")
    elif gate == "warn":
        print("VALIDATION COMPLETED (warn band)")
    else:
        print("VALIDATION FAILED (performance regression)")
        exit_code = 1

    if no_improvements:
        print("  NOTE: no kernel improvements detected — performance delta treated as run-to-run noise")
    if comparison["ttft_upgraded"]:
        print("  NOTE: gate upgraded from warn to fail due to severe TTFT regression")

    print(f"  Baseline file:  {comparison['baseline_file']}")
    print(f"  Optimized file: {comparison['optimized_file']}")
    print(f"  Baseline:  {bl_tps:.1f} tok/s")
    print(f"  Optimized: {opt_tps:.1f} tok/s")
    if speedup is not None:
        print(f"  Speedup:   {speedup:.3f}x")
    else:
        print("  Speedup:   N/A (zero baseline throughput)")
    print(f"  Gate:      {gate}")
    ttft_pct = comparison["ttft_regression_pct"]
    if ttft_pct is not None:
        print(f"  TTFT Δ:    {ttft_pct:+.2f}%")

    output_path = os.path.join(results_dir, "optimization_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved {output_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
