#!/usr/bin/env python3
"""Run a library's own test suite against a forked checkout.

Phase 7's library-suite validation step (Bucket A only). Operates on the
main fork tree (not GEAK's per-agent worktrees).

Reads the kernel's `library_test_path` and `library_test_command` from
`kernel_source_map.yaml`, executes the command inside the fork directory,
and reports a structured `{pass_count, fail_count, skipped_count,
log_path}`.

Bucket B (`in_place_optimize_no_harness`) kernels are not exercised here
- their `library_test_*_count` is reported as `null` to distinguish
"skipped by design" from "ran and all failed".

Usage:
  library_test_driver.py --kernel <name> --fork <dir> [--map <path>] \
      [--log-dir <dir>]
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MAP = os.path.join(
    REPO_ROOT, "skills", "inference-optimize", "resources", "kernel_source_map.yaml"
)


def load_entry(map_path, kernel_symbol):
    with open(map_path) as f:
        doc = yaml.safe_load(f)
    for entry in doc.get("entries", []):
        if fnmatch.fnmatchcase(kernel_symbol, entry.get("symbol_pattern", "")):
            return entry
    return None


# pytest summary line format examples:
#   "5 passed, 2 failed, 1 skipped in 3.21s"
#   "10 passed in 1.23s"
#   "1 failed in 0.5s"
PYTEST_COUNT_RE = re.compile(
    r"(?P<n>\d+)\s+(?P<kind>passed|failed|skipped|errors?|xfailed|xpassed)"
)


def parse_pytest_counts(stdout):
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in PYTEST_COUNT_RE.finditer(stdout or ""):
        kind = match.group("kind")
        n = int(match.group("n"))
        if kind == "passed":
            counts["passed"] = n
        elif kind == "failed":
            counts["failed"] = n
        elif kind == "skipped":
            counts["skipped"] = n
        elif kind.startswith("error"):
            counts["errors"] = n
    return counts


def main():
    parser = argparse.ArgumentParser(description="Run library-suite tests on a fork")
    parser.add_argument("--kernel", required=True, help="Runtime kernel symbol")
    parser.add_argument("--fork", required=True, help="Fork directory (e.g. forks/vllm)")
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--log-dir", default=None,
                        help="Where to write the captured stdout/stderr log")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    entry = load_entry(args.map, args.kernel)
    if entry is None:
        result = {
            "kernel": args.kernel,
            "ran": False,
            "skip_reason": "no_kernel_source_map_entry",
            "pass_count": None,
            "fail_count": None,
            "skipped_count": None,
            "log_path": None,
        }
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1

    if entry.get("bucket") == "B" or entry.get("library_test_path") in (None, ""):
        # Bucket B / no harness: skip-by-design; counts are null.
        result = {
            "kernel": args.kernel,
            "ran": False,
            "skip_reason": "no_harness_by_design",
            "pass_count": None,
            "fail_count": None,
            "skipped_count": None,
            "log_path": None,
        }
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    cmd = entry["library_test_command"].format(
        library_test_path=entry["library_test_path"]
    )
    log_dir = args.log_dir or os.path.join(args.fork, "_geak_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"library_test_{args.kernel.replace('/', '_')}.log"
    )

    start = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=args.fork,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
        )
        stdout, stderr = proc.stdout, proc.stderr
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n[timeout after {args.timeout_sec}s]\n"
        rc = 124
        timed_out = True

    duration = time.time() - start
    with open(log_path, "w") as f:
        f.write(f"# command: {cmd}\n")
        f.write(f"# cwd: {args.fork}\n")
        f.write(f"# rc: {rc}  duration: {duration:.1f}s\n")
        f.write("--- stdout ---\n")
        f.write(stdout)
        f.write("\n--- stderr ---\n")
        f.write(stderr)

    counts = parse_pytest_counts(stdout)
    result = {
        "kernel": args.kernel,
        "ran": True,
        "command": cmd,
        "returncode": rc,
        "timed_out": timed_out,
        "pass_count": counts["passed"],
        "fail_count": counts["failed"] + counts["errors"],
        "skipped_count": counts["skipped"],
        "log_path": log_path,
        "duration_sec": round(duration, 2),
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
