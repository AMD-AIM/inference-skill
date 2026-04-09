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


def main():
    parser = argparse.ArgumentParser(description="Validate optimization results")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = args.results_dir

    baseline_files = [
        f for f in glob.glob(os.path.join(results_dir, "*.json"))
        if not any(
            skip in os.path.basename(f)
            for skip in [
                "benchmark_summary", "bottlenecks", "sweep_configs",
                "optimization", "optimized_",
            ]
        )
    ]
    optimized_files = glob.glob(os.path.join(results_dir, "optimized_*.json"))

    errors = []
    if not baseline_files:
        errors.append("No baseline benchmark result found")
    if not optimized_files:
        errors.append("No optimized benchmark result found — the patched server may have failed")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    baseline = json.load(open(baseline_files[0]))
    optimized = json.load(open(optimized_files[0]))

    bl_tps = baseline.get("total_token_throughput", baseline.get("output_throughput", 0))
    opt_tps = optimized.get("total_token_throughput", optimized.get("output_throughput", 0))
    speedup = opt_tps / bl_tps if bl_tps > 0 else 1.0

    if speedup < 1.0:
        print("WARNING: REGRESSION DETECTED — optimized throughput is lower than baseline")
        print("Investigate: kernel compatibility issues, plugin import failures, or server startup errors")

    print("VALIDATION PASSED" if speedup >= 1.0 else "VALIDATION COMPLETED (with regression)")
    print(f"  Baseline:  {bl_tps:.1f} tok/s")
    print(f"  Optimized: {opt_tps:.1f} tok/s")
    print(f"  Speedup:   {speedup:.3f}x")

    comparison = {
        "validated": True,
        "baseline_file": os.path.basename(baseline_files[0]),
        "optimized_file": os.path.basename(optimized_files[0]),
        "baseline": {
            "total_token_throughput": bl_tps,
            "mean_ttft_ms": baseline.get("mean_ttft_ms", 0),
            "mean_itl_ms": baseline.get("mean_itl_ms", 0),
            "mean_tpot_ms": baseline.get("mean_tpot_ms", 0),
            "duration_s": baseline.get("duration", 0),
        },
        "optimized": {
            "total_token_throughput": opt_tps,
            "mean_ttft_ms": optimized.get("mean_ttft_ms", 0),
            "mean_itl_ms": optimized.get("mean_itl_ms", 0),
            "mean_tpot_ms": optimized.get("mean_tpot_ms", 0),
            "duration_s": optimized.get("duration", 0),
        },
        "speedup": round(speedup, 4),
    }

    output_path = os.path.join(results_dir, "optimization_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()
