#!/usr/bin/env python3
"""Phase 8 step 3: run the standard vLLM e2e benchmark with rebuilt forks.

This script is intentionally a thin wrapper -- it boots vLLM via the same
launcher Phase 2 used (no plugin injection, no PYTHONPATH override, no
inject_plugin.py). The editable installs from `rebuild_libraries.py` are
already on the import path.

Outputs are written next to the existing Phase-2 baseline so
`scripts/report/validate_optimization.py` can compare them with its existing
thresholds.

Usage:
  run_e2e.py --output-dir <path> --launcher '<phase-2-equivalent cmd>' \
      [--label after_rebuild] [--timeout-sec 3600]
"""

import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run vLLM e2e benchmark with rebuilt forks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--launcher", required=True,
                        help="Same shell command Phase 2 used to run the baseline benchmark")
    parser.add_argument("--label", default="after_rebuild")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()

    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"e2e_{args.label}.log")

    start = time.time()
    try:
        with open(log_path, "w") as logf:
            logf.write(f"# launcher: {args.launcher}\n\n")
            logf.flush()
            proc = subprocess.run(
                ["bash", "-lc", args.launcher],
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_sec,
            )
            rc = proc.returncode
            timed_out = False
    except subprocess.TimeoutExpired:
        rc = 124
        timed_out = True

    duration = time.time() - start
    with open(log_path, "a") as logf:
        logf.write(f"\n# rc: {rc}  duration: {duration:.1f}s  timed_out: {timed_out}\n")

    summary = {
        "label": args.label,
        "launcher": args.launcher,
        "log_path": log_path,
        "returncode": rc,
        "timed_out": timed_out,
        "duration_sec": round(duration, 2),
        "e2e_ran": rc == 0,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
