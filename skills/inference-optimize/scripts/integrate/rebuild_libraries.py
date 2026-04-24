#!/usr/bin/env python3
"""Phase 8 step 1: rebuild each fork in-place.

Reads `<output_dir>/forks/manifest.json` (produced by `fork_upstream.py`),
runs each fork's `rebuild_command` from inside its directory, and logs to
`<output_dir>/results/rebuild_<lib>.log`. Wheel-installed copies of the
library are shadowed by the editable install (Python's import order).

Usage:
  rebuild_libraries.py --output-dir <path> [--libs vllm,aiter,fla] \
      [--max-jobs N] [--timeout-sec 5400]
"""

import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Rebuild forked libraries in-place")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--libs", default=None,
                        help="Comma-separated subset of libs (default: all in manifest)")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Sets MAX_JOBS env for the build subprocess")
    parser.add_argument("--timeout-sec", type=int, default=5400)
    args = parser.parse_args()

    forks_root = os.path.join(args.output_dir, "forks")
    manifest_path = os.path.join(forks_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        sys.stderr.write(f"ERROR: forks/manifest.json not found at {manifest_path}\n")
        return 2

    with open(manifest_path) as f:
        manifest = json.load(f)
    forks = manifest.get("forks", {})

    if args.libs:
        wanted = {x.strip() for x in args.libs.split(",") if x.strip()}
    else:
        wanted = set(forks.keys())

    log_dir = os.path.join(args.output_dir, "results")
    os.makedirs(log_dir, exist_ok=True)

    libraries_rebuilt = []
    rebuild_failures = []

    for lib in sorted(wanted):
        info = forks.get(lib)
        if info is None:
            sys.stderr.write(f"WARNING: {lib} absent from forks/manifest.json; skipping\n")
            continue
        if info.get("dirty"):
            rebuild_failures.append({"lib": lib, "error": "fork_dirty"})
            continue
        cmd = info.get("rebuild_command", "")
        if not cmd:
            sys.stderr.write(f"WARNING: {lib} has no rebuild_command; skipping\n")
            continue

        env = os.environ.copy()
        if args.max_jobs is not None:
            env["MAX_JOBS"] = str(args.max_jobs)

        log_path = os.path.join(log_dir, f"rebuild_{lib}.log")
        sys.stdout.write(f"-- rebuilding {lib}: {cmd}\n")
        sys.stdout.flush()
        start = time.time()
        try:
            with open(log_path, "w") as logf:
                logf.write(f"# command: {cmd}\n")
                logf.write(f"# cwd: {info['fork_path']}\n\n")
                logf.flush()
                proc = subprocess.run(
                    ["bash", "-lc", cmd],
                    cwd=info["fork_path"],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
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

        record = {
            "lib": lib,
            "commit": info.get("pinned_commit"),
            "install_log_path": log_path,
            "duration_sec": round(duration, 2),
            "returncode": rc,
            "timed_out": timed_out,
        }
        if rc == 0:
            libraries_rebuilt.append(record)
        else:
            rebuild_failures.append(record)

    summary = {
        "libraries_rebuilt": libraries_rebuilt,
        "libraries_rebuild_failed": rebuild_failures,
        "libraries_rebuilt_ok_count": len(libraries_rebuilt),
        "libraries_rebuild_failed_count": len(rebuild_failures),
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if not rebuild_failures else 1


if __name__ == "__main__":
    sys.exit(main())
