#!/usr/bin/env python3
"""Verify all winning kernels from GEAK are captured in the optimized directory.

Usage: python3 verify_winning_kernels.py --problems-dir <dir> --optimized-dir <dir>
Exit code 1 if any winning kernels are missing.
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Verify winning kernels are captured")
    parser.add_argument("--problems-dir", required=True)
    parser.add_argument("--optimized-dir", required=True)
    args = parser.parse_args()

    results_path = os.path.join(args.problems_dir, "geak_results.json")
    if not os.path.isfile(results_path):
        print("WARNING: geak_results.json not found — no kernels to integrate")
        return

    results = json.load(open(results_path))
    missing = []
    for r in results:
        if r["speedup"] > 1.0:
            opt_file = os.path.join(args.optimized_dir, r["name"] + "_opt.py")
            if not os.path.isfile(opt_file):
                missing.append(r["name"])

    if missing:
        print(f"ERROR: {len(missing)} winning kernels NOT captured in {args.optimized_dir}/: {missing}")
        print("Run Phase 7 Step 3.5 patch recovery before proceeding!")
        sys.exit(1)
    else:
        winners = sum(1 for r in results if r["speedup"] > 1.0)
        print(f"OK: All {winners} winning kernels captured in {args.optimized_dir}/")


if __name__ == "__main__":
    main()
