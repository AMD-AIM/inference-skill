#!/usr/bin/env python3
"""Collect winning kernels from optimization attempts and produce geak_results.json.

Run INSIDE the Docker container:
  docker exec $CONTAINER python3 /workspace/scripts/collect_winning_kernels.py \
      --problems-dir /workspace/problems --optimized-dir /workspace/optimized
"""

import argparse
import glob
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Collect winning kernels")
    parser.add_argument("--problems-dir", required=True)
    parser.add_argument("--optimized-dir", required=True)
    args = parser.parse_args()

    os.chdir(args.problems_dir)

    manifest = json.load(open("optimization_manifest.json"))
    opt_lookup = {o["name"]: o for o in manifest.get("optimizations", [])}

    results = []
    for best_file in sorted(glob.glob("*_opt_best.json")):
        tracker = json.load(open(best_file))
        name = best_file.replace("_opt_best.json", "")
        speedup = tracker.get("best_speedup", 0)
        opt_info = opt_lookup.get(name, {})
        entry = {
            "name": name,
            "speedup": speedup,
            "ref_ms": tracker.get("best_ref_time", 0),
            "opt_ms": tracker.get("best_opt_time", 0),
            "geak_mode": opt_info.get("geak_mode", "unknown"),
            "kernel_type": opt_info.get("kernel_type", "unknown"),
            "profiling_pct": opt_info.get("profiling_pct", 0),
            "patch_recovered": tracker.get("patch_recovered", False),
        }
        results.append(entry)
        if speedup > 1.0:
            opt_file = name + "_opt.py"
            if os.path.isfile(opt_file):
                shutil.copy2(opt_file, args.optimized_dir)
                print(f"  Copied {opt_file} (speedup={speedup:.2f}x, mode={entry['geak_mode']})")
        else:
            print(f"  Skipped {name} (speedup={speedup:.2f}x < 1.0x)")

    with open("geak_results.json", "w") as f:
        json.dump(results, f, indent=2)

    winners = sum(1 for r in results if r["speedup"] > 1.0)
    print(f"\nTotal: {len(results)} kernels, {winners} with speedup > 1.0x")


if __name__ == "__main__":
    main()
