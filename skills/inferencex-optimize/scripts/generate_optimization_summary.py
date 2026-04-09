#!/usr/bin/env python3
"""Generate machine-readable optimization_summary.json from all phase artifacts.

Usage: python3 generate_optimization_summary.py \
    --output <path> --config-key <key> --framework <fw> \
    --env-info <path> --results-dir <dir> --problems-dir <dir>
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Generate optimization summary JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--env-info", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--problems-dir", default="")
    args = parser.parse_args()

    summary = {
        "config_key": args.config_key,
        "framework": args.framework,
        "phases_completed": True,
    }

    if args.env_info and os.path.isfile(args.env_info):
        env = json.load(open(args.env_info))
        summary["gpu_arch"] = env.get("gpu_arch", "unknown")
        summary["geak_mode"] = "auto" if env.get("geak_available") else "manual"

    comp_path = os.path.join(args.results_dir, "optimization_comparison.json") if args.results_dir else ""
    if comp_path and os.path.isfile(comp_path):
        comp = json.load(open(comp_path))
        summary["baseline_throughput"] = comp.get("baseline", {}).get("total_token_throughput", 0)
        summary["optimized_throughput"] = comp.get("optimized", {}).get("total_token_throughput", 0)
        summary["speedup"] = comp.get("speedup", 1.0)
        summary["validated"] = comp.get("validated", False)

    results_path = os.path.join(args.problems_dir, "geak_results.json") if args.problems_dir else ""
    if results_path and os.path.isfile(results_path):
        results = json.load(open(results_path))
        summary["kernels_attempted"] = len(results)
        summary["kernels_improved"] = sum(1 for r in results if r.get("speedup", 0) > 1.0)
        summary["patches_recovered"] = sum(1 for r in results if r.get("patch_recovered", False))
        summary["kernel_results"] = results

    manifest_path = os.path.join(args.problems_dir, "optimization_manifest.json") if args.problems_dir else ""
    if manifest_path and os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
        summary["total_problem_files"] = len(manifest.get("optimizations", []))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
