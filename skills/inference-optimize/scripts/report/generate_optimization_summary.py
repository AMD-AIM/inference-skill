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
        summary["vllm_version"] = env.get("vllm_version", "unknown")

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
        summary["integration_schema_version"] = int_manifest.get("schema_version")
        libs_rebuilt = int_manifest.get("libraries_rebuilt", []) or []
        summary["libraries_rebuilt_count"] = len(libs_rebuilt)
        summary["libraries_rebuilt"] = libs_rebuilt
        summary["dispatch_verified"] = int_manifest.get("dispatch_verified")
        summary["e2e_ran"] = int_manifest.get("e2e_ran")

    dispatch_path = os.path.join(args.results_dir, "dispatch_verification.json") if args.results_dir else ""
    if dispatch_path and os.path.isfile(dispatch_path):
        with open(dispatch_path) as f:
            dispatch = json.load(f)
        summary["expected_symbol_total_count"] = dispatch.get("expected_symbol_total_count", 0)
        summary["vendor_symbol_leaked_count"] = dispatch.get("vendor_symbol_leaked_count", 0)
        summary["redirect_required_count"] = dispatch.get("redirect_required_count", 0)
        summary["redirect_honored_count"] = dispatch.get("redirect_honored_count", 0)

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
        # New scalar names use lib-bench speedup; preserve legacy "speedup" key for fallback.
        def _speedup(r):
            return r.get("geak_speedup_lib_bench", r.get("speedup", 0)) or 0
        summary["kernels_improved"] = sum(1 for r in results if _speedup(r) > 1.0)
        summary["unverified_per_kernel_count"] = sum(
            1 for r in results if r.get("optimization_unverified_per_kernel") is True
        )
        summary["bucket_b_winners"] = [
            {
                "name": r.get("name"),
                "library": r.get("library"),
                "source_form": r.get("source_form"),
                "gating_reason": r.get("gating_reason"),
                "geak_speedup_lib_bench": r.get("geak_speedup_lib_bench"),
                "no_harness_warning": r.get("no_harness_warning"),
            }
            for r in results
            if r.get("optimization_unverified_per_kernel") is True
        ]
        summary["kernel_results"] = results

    manifest_path = os.path.join(args.problems_dir, "optimization_manifest.json") if args.problems_dir else ""
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        opts = manifest.get("optimizations", []) if isinstance(manifest, dict) else (
            manifest if isinstance(manifest, list) else []
        )
        summary["total_optimization_targets"] = len(opts)
        summary["bucket_a_count"] = sum(1 for o in opts if o.get("bucket") == "A")
        summary["bucket_b_count"] = sum(1 for o in opts if o.get("bucket") == "B")
        summary["bucket_c_count"] = sum(1 for o in opts if o.get("bucket") == "C")

    forks_manifest_path = os.path.join(
        args.results_dir, "..", "forks", "manifest.json"
    ) if args.results_dir else ""
    if forks_manifest_path and os.path.isfile(forks_manifest_path):
        with open(forks_manifest_path) as f:
            forks = json.load(f)
        # Producer (scripts/optimize/fork_upstream.py) writes:
        #   {"forks": {<lib>: {repo_url, pinned_commit, fork_path, dirty,
        #                      rebuild_command, ...}, ...},
        #    "ck_branch_merged_status": bool, "vllm_version": ..., ...}
        fork_entries = forks.get("forks", {}) if isinstance(forks, dict) else {}
        if isinstance(fork_entries, dict):
            entries_iter = fork_entries.values()
            summary["forks_required_count"] = len(fork_entries)
        else:  # tolerate legacy/list shape so the report still renders
            entries_iter = [e for e in fork_entries if isinstance(e, dict)]
            summary["forks_required_count"] = len(fork_entries) if isinstance(fork_entries, list) else 0
        summary["forks_pinned_count"] = sum(
            1 for e in entries_iter if isinstance(e, dict) and not e.get("dirty", False)
        )
        summary["ck_branch_merged_status"] = forks.get("ck_branch_merged_status") if isinstance(forks, dict) else None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
