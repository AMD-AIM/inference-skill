#!/usr/bin/env python3
"""
Offline GEMM tuning using PyTorch TunableOps.

Reads untuned GEMM shapes collected via PYTORCH_TUNABLEOP_RECORD_UNTUNED=1,
runs all hipBLASLt/rocBLAS candidates, writes the best algorithm per shape.

This is the standalone equivalent of the official ROCm vLLM gradlib/gemm_tuner.py
for dev/upstream vLLM builds that don't ship gradlib.

Usage:
    python tune_gemm_shapes.py \\
        --untuned  results/untuned_shapes_final.csv \\
        --output   optimized/tuned_gemm.csv \\
        --max-iter 50 \\
        --max-duration 20

Notes:
    - vLLM uses BFloat16_TN format (not NN): F.linear(x, weight) = x @ weight.T
    - Projections are fused: QKV merged, gate+up merged — shapes come from actual trace
    - rocBLAS typically beats hipBLASLt for decode shapes (M=1..128) on RDNA3/CDNA3
    - All Rocblas solution IDs may be negative (valid, arch-specific) — do NOT filter them
"""
import argparse
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Offline GEMM tuning via TunableOps")
    parser.add_argument("--untuned",      required=True, help="Untuned shapes CSV (from RECORD_UNTUNED)")
    parser.add_argument("--output",       required=True, help="Output tuned CSV path")
    parser.add_argument("--max-iter",     type=int, default=50,  help="Max tuning iterations per shape")
    parser.add_argument("--max-duration", type=int, default=20,  help="Max tuning seconds per shape")
    args = parser.parse_args()

    if not os.path.exists(args.untuned):
        print(f"ERROR: untuned file not found: {args.untuned}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available", file=sys.stderr)
        sys.exit(1)

    n_shapes = sum(1 for l in open(args.untuned) if l.startswith("Gemm"))
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"Untuned:  {args.untuned} ({n_shapes} shapes)")
    print(f"Output:   {args.output}")
    print(f"Config:   max_iter={args.max_iter}, max_duration={args.max_duration}s")
    print()

    # Configure TunableOps for tuning
    torch.cuda.tunable.enable(val=True)
    torch.cuda.tunable.tuning_enable(val=True)
    torch.cuda.tunable.set_max_tuning_iterations(args.max_iter)
    torch.cuda.tunable.set_max_tuning_duration(args.max_duration)
    torch.cuda.tunable.set_filename(args.output)

    t0 = time.time()
    torch.cuda.tunable.tune_gemm_in_file(args.untuned)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.cuda.tunable.write_file(args.output)

    n_tuned = sum(1 for l in open(args.output) if l.startswith("Gemm"))
    print(f"\nTuning complete: {n_tuned} shapes tuned in {elapsed:.0f}s")
    print()

    # Print results summary
    print(f"{'Shape':55s}  {'Algorithm':30s}  {'Time(ms)':>8}")
    print("-" * 100)
    for line in open(args.output):
        if not line.startswith("Gemm"):
            continue
        parts = line.strip().split(",")
        if len(parts) >= 4:
            shape = parts[1][:55]
            algo  = parts[2][:30]
            tms   = parts[3] if len(parts) > 3 else ""
            print(f"{shape:<55}  {algo:<30}  {tms:>8}")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
