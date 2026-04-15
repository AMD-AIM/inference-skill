#!/usr/bin/env python3
"""Generate machine-readable optimization_summary.json from all phase artifacts.

Usage: python3 generate_optimization_summary.py \
    --output <path> --config-key <key> --framework <fw> \
    --env-info <path> --results-dir <dir> --problems-dir <dir>
"""

import argparse
import json
import os

from integration_outcome import SCHEMA_VERSION, pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Generate optimization summary JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--env-info", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--problems-dir", default="")
    parser.add_argument("--skip-integration", action="store_true", default=False)
    args = parser.parse_args()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_key": args.config_key,
        "framework": args.framework,
        "all_phases_completed": True,
        "pipeline_status": "completed",
        "blocker_count": 0,
    }

    if args.env_info and os.path.isfile(args.env_info):
        with open(args.env_info) as f:
            env = json.load(f)
        summary["gpu_arch"] = env.get("gpu_arch", "unknown")
        summary["geak_mode"] = env.get("effective_geak_mode", "manual")

    integration_gate = None
    comp_path = os.path.join(args.results_dir, "optimization_comparison.json") if args.results_dir else ""
    if comp_path and os.path.isfile(comp_path):
        with open(comp_path) as f:
            comp = json.load(f)
        summary["baseline_throughput"] = comp.get("baseline", {}).get("total_token_throughput", 0)
        summary["optimized_throughput"] = comp.get("optimized", {}).get("total_token_throughput", 0)
        summary["speedup"] = comp.get("speedup", 1.0)
        summary["e2e_speedup"] = comp.get("e2e_speedup", comp.get("speedup", 1.0))
        summary["validated"] = comp.get("validated", False)
        summary["artifacts_valid"] = comp.get("artifacts_valid", False)
        summary["performance_valid"] = comp.get("performance_valid", False)
        summary["performance_gate"] = comp.get("performance_gate", "unknown")
        summary["ttft_regression_pct"] = comp.get("ttft_regression_pct")
        integration_gate = comp.get("performance_gate")

    blocker_list = []
    blockers_path = os.path.join(args.results_dir, "pipeline_blockers.json") if args.results_dir else ""
    if blockers_path and os.path.isfile(blockers_path):
        with open(blockers_path) as f:
            blockers_data = json.load(f)
        blocker_list = blockers_data.get("blockers", [])
        summary["blocker_count"] = len(blocker_list)

    summary["pipeline_status"] = pipeline_status(
        blocker_list, integration_gate,
        integration_expected=bool(args.results_dir),
        integration_skipped=args.skip_integration,
    )
    summary["all_phases_completed"] = summary["pipeline_status"] in {
        "completed", "completed with warnings",
    }

    int_manifest_path = os.path.join(args.results_dir, "integration_manifest.json") if args.results_dir else ""
    if int_manifest_path and os.path.isfile(int_manifest_path):
        with open(int_manifest_path) as f:
            int_manifest = json.load(f)
        summary["integration_plugin_type"] = int_manifest.get("plugin_type")
        int_summary = int_manifest.get("summary", {})
        summary["integration_total_targets"] = int_summary.get("total_targets")
        summary["integration_integrated"] = int_summary.get("integrated")
        summary["integration_blocked"] = int_summary.get("blocked")
        summary["integration_coverage_pct"] = int_summary.get("coverage_pct")

    results_path = os.path.join(args.problems_dir, "geak_results.json") if args.problems_dir else ""
    if results_path and os.path.isfile(results_path):
        with open(results_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            results = raw.get("kernels", [])
        elif isinstance(raw, list):
            results = raw
        else:
            results = []
        summary["kernels_attempted"] = len(results)
        summary["kernels_improved"] = sum(1 for r in results if r.get("speedup", 0) > 1.0)
        summary["patches_recovered"] = sum(1 for r in results if r.get("patch_recovered", False))
        summary["kernel_results"] = results

    manifest_path = os.path.join(args.problems_dir, "optimization_manifest.json") if args.problems_dir else ""
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        summary["total_problem_files"] = len(manifest.get("optimizations", []))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
