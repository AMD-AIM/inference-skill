#!/usr/bin/env python3
"""
Run kernel_breakdown.py in parallel for all concurrency trace directories.

Reads profile_meta.json to discover trace dirs, launches one kernel_breakdown.py
process per concurrency level in parallel (all start simultaneously), then waits
for all to finish and prints output in concurrency order.

Usage:
    python run_breakdown.py \
        --profile-meta results/profile_meta.json \
        --scripts-dir  /path/to/scripts \
        --output-dir   results/gap_analysis
"""
import argparse
import glob
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-meta", required=True, help="profile_meta.json path")
    parser.add_argument("--scripts-dir",  required=True, help="directory containing kernel_breakdown.py")
    parser.add_argument("--output-dir",   required=True, help="where to write gap_cN.json files")
    parser.add_argument("--top-n",        default=30, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.profile_meta):
        print(f"FATAL: {args.profile_meta} not found", file=sys.stderr); sys.exit(1)

    kb_script = os.path.join(args.scripts_dir, "kernel_breakdown.py")
    if not os.path.exists(kb_script):
        print(f"FATAL: {kb_script} not found", file=sys.stderr); sys.exit(1)

    meta = json.load(open(args.profile_meta))
    os.makedirs(args.output_dir, exist_ok=True)

    # Launch all breakdown jobs in parallel
    procs = {}
    for cs, m in sorted(meta.items(), key=lambda x: int(x[0])):
        conc = int(cs)
        td = m["trace_dir"]
        traces = glob.glob(td + "/*.json*")
        if not traces:
            print(f"  SKIP conc={conc}: no traces in {td}/  [FAIL]", flush=True)
            continue
        out = os.path.join(args.output_dir, f"gap_c{conc}.json")
        mode = "decode-only" if m.get("decode_only") else "prefill+decode"
        print(f"  Launching: conc={conc} [{mode}] ({len(traces)} trace(s))...", flush=True)
        p = subprocess.Popen(
            [sys.executable, kb_script,
             "--trace-dir", td, "--output", out,
             "--label", f"conc={conc} ({mode})", "--top-n", str(args.top_n)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs[conc] = (p, out, mode)

    if not procs:
        print("FATAL: No trace directories found in profile_meta.json", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(procs)} jobs running in parallel...", flush=True)

    # Collect results in concurrency order (buffered to avoid interleaving)
    any_ok = False
    for conc in sorted(procs.keys()):
        p, out, mode = procs[conc]
        stdout_bytes, _ = p.communicate()
        for line in (stdout_bytes or b"").decode(errors="replace").splitlines():
            print(f"  [conc={conc}] {line}", flush=True)
        if p.returncode != 0:
            print(f"  ERROR: kernel_breakdown.py failed for conc={conc}  [FAIL]", flush=True)
        else:
            any_ok = True
            print(f"  [conc={conc}] Done  [OK]", flush=True)

    if not any_ok:
        print("FATAL: No concurrency level could be analyzed", file=sys.stderr)
        sys.exit(1)

    print(f"  All breakdown jobs complete. Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
