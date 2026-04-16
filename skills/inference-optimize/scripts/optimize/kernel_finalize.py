#!/usr/bin/env python3
"""
Kernel Finalize — Writes best optimization result to target file.

Reads the best tracker JSON and writes the best code to the target file.

Usage:
    python kernel_finalize.py --src problem.py --target problem_opt.py
    python kernel_finalize.py --tracker problem_opt_best.json --target problem_opt.py

Part of the inference-optimize skill. Can be used standalone.
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Finalize kernel optimization")
    parser.add_argument("--src", default=None, help="Source file (for log context)")
    parser.add_argument("--target", required=True, help="Target file to write best result")
    parser.add_argument("--tracker", default=None, help="Path to best tracker JSON")
    args = parser.parse_args()

    tgt_path = os.path.abspath(args.target)
    tracker_path = args.tracker or tgt_path.replace(".py", "_best.json")

    if not os.path.exists(tracker_path):
        print(f"ERROR: Tracker file not found: {tracker_path}")
        sys.exit(1)

    with open(tracker_path) as f:
        tracker = json.load(f)

    if tracker.get("best_speedup", 0) == 0:
        print("ERROR: No successful optimization found!")
        sys.exit(1)

    best_code = tracker["best_code"]
    best_speedup = tracker["best_speedup"]
    best_ref = tracker.get("best_ref_time", 0)
    best_opt = tracker.get("best_opt_time", 0)
    best_attempt = tracker.get("best_attempt", "?")

    # Write best code to target
    with open(tgt_path, "w") as f:
        f.write(best_code)

    print(f"\n{'='*50}")
    print(f"OPTIMIZATION FINALIZED")
    print(f"{'='*50}")
    print(f"Best Speedup: {best_speedup:.2f}x (Attempt {best_attempt})")
    print(f"Ref time: {best_ref:.4f} ms")
    print(f"Opt time: {best_opt:.4f} ms")
    print(f"Best code written to: {tgt_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

