#!/usr/bin/env python3
"""
Filter untuned GEMM shapes to only those that appear in the profiler trace.

PYTORCH_TUNABLEOP_RECORD_UNTUNED collects shapes across all concurrencies (benchmark +
prefill + decode). The profiler trace at peak concurrency captures only the shapes that
actually execute during production serving. Tuning only those shapes reduces the shape
count by ~90% and saves 30+ minutes of offline tuning time.

Usage:
    python filter_shapes.py \
        --untuned results/untuned_shapes_final.csv \
        --real-shapes results/real_shapes.json \
        --output results/untuned_shapes_real_only.csv
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--untuned",     required=True, help="untuned_shapes_final.csv")
    parser.add_argument("--real-shapes", required=True, help="real_shapes.json from Phase 3")
    parser.add_argument("--output",      required=True, help="filtered output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.untuned):
        print(f"FATAL: {args.untuned} not found", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.real_shapes):
        print(f"WARNING: {args.real_shapes} not found — copying all shapes (no filter)")
        import shutil; shutil.copy(args.untuned, args.output)
        total = sum(1 for l in open(args.output) if l.startswith("Gemm"))
        print(f"Copied {total} shapes (unfiltered)"); sys.exit(0)

    rs = json.load(open(args.real_shapes))

    # Build set of exact (M, K, N) tuples from real shapes
    real_mkn = set()
    for entry in rs.get("shapes", []) + rs.get("top_shapes_by_calls", []):
        m, k, n = entry["MKN"]
        real_mkn.add((m, k, n))

    print(f"Real MKN tuples from profiler trace: {len(real_mkn)}")

    # CSV format: GemmTunableOp_BFloat16_TN,tn_N_M_K_stride_a0_...,
    # N=nums[0], M=nums[1], K=nums[2]  (MKN in real_shapes → M=MKN[0], K=MKN[1], N=MKN[2])
    kept, skipped = 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.untuned) as inp, open(args.output, "w") as out:
        for line in inp:
            if not line.startswith("Gemm"):
                out.write(line)   # header / blank lines pass through unchanged
                continue
            parts = line.strip().split(",")
            matched = False
            if len(parts) >= 2:
                nums = [x for x in parts[1].split("_") if x.isdigit()]
                if len(nums) >= 3:
                    n_val, m_val, k_val = int(nums[0]), int(nums[1]), int(nums[2])
                    if (m_val, k_val, n_val) in real_mkn:
                        out.write(line); kept += 1; matched = True
            if not matched:
                skipped += 1

    total = kept + skipped
    pct = 100 * kept / total if total > 0 else 0
    print(f"Filtered: {kept}/{total} shapes kept ({pct:.0f}% of original, {skipped} skipped)")
    print(f"Saved: {args.output}")

    if kept == 0:
        print("WARNING: 0 shapes kept — MKN tuples may not match CSV format. "
              "Check real_shapes.json MKN ordering.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
