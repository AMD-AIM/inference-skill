#!/usr/bin/env python3
"""
Offline GEMM implementation benchmarker for vLLM ROCm unquantized GEMM dispatch.

For each (n, k, m) shape from real profiler traces, benchmarks all implementations
that vllm::rocm_unquantized_gemm_impl could dispatch to and picks the fastest:

  - wvSplitK  : custom vLLM rocBLAS SplitK kernel (default for n<=4, m>8)
  - llmm1     : custom vLLM LLMM1 kernel (for n=1, k<=8192, m%4==0, no bias)
  - linear    : torch.nn.functional.linear (default for n>4; TunableOps intercepts here)

The winner is written to routing_table.json and used by Phase 5 injection to
override dispatch decisions at runtime.

Why this matters:
  - vLLM hard-codes n<=4 → wvSplitK, bypassing TunableOps entirely.
  - Benchmarking reveals whether wvSplitK is actually optimal for every shape,
    or whether forcing through linear (+ TunableOps tuning) would be faster.
  - Covers conc=1 (n=1) and conc=4 (n=4) which TunableOps-only approach misses.

Usage:
    python3 benchmark_gemm_impls.py \\
        --shapes real_shapes.json \\
        --output routing_table.json \\
        [--warmup 20] [--iters 100] [--dtype bfloat16]
"""
import argparse
import json
import os
import sys
import time

import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def timed_run(fn, warmup: int, iters: int) -> float:
    """Return median execution time in microseconds."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]  # median


def get_cu_count() -> int:
    try:
        from vllm._custom_ops import get_cu_count  # type: ignore
        return get_cu_count()
    except Exception:
        return torch.cuda.get_device_properties(0).multi_processor_count


# ── implementations ──────────────────────────────────────────────────────────

def impl_wvsplitk(x_view, weight, cu_count, bias):
    """wvSplitK: custom vLLM SplitK kernel. Requires n<=4, m>8."""
    from vllm import _custom_ops as ops
    return ops.wvSplitK(weight, x_view, cu_count, bias)


def impl_llmm1(x_view, weight, bias):
    """LLMM1: custom vLLM kernel. Requires n=1, k<=8192, m%4==0, bias=None."""
    from vllm import _custom_ops as ops
    return ops.LLMM1(weight, x_view, 4)


def impl_linear(x, weight, bias):
    """torch.nn.functional.linear — goes through TunableOps if enabled."""
    return torch.nn.functional.linear(x, weight, bias)


# ── per-shape benchmark ───────────────────────────────────────────────────────

def benchmark_shape(n: int, k: int, m: int, dtype: torch.dtype,
                    warmup: int, iters: int, cu_count: int) -> dict:
    """
    Benchmark all applicable impls for one (n, k, m) shape.
    n = batch size (tokens), k = input features, m = output features.
    Returns dict with timings and the winner.
    """
    device = "cuda"
    x      = torch.randn(n, k, device=device, dtype=dtype)
    weight = torch.randn(m, k, device=device, dtype=dtype)
    x_view = x  # already 2-D

    results = {}

    # ── wvSplitK: valid when m > 8 and 0 < n <= 4 ──
    if m > 8 and 0 < n <= 4:
        try:
            impl_wvsplitk(x_view, weight, cu_count, None)  # smoke test
            t = timed_run(lambda: impl_wvsplitk(x_view, weight, cu_count, None),
                          warmup, iters)
            results["wvSplitK"] = t
        except Exception as e:
            results["wvSplitK"] = f"ERROR: {e}"

    # ── LLMM1: valid when n=1, k<=8192, m%4==0, no bias ──
    if n == 1 and k <= 8192 and m % 4 == 0:
        try:
            impl_llmm1(x_view, weight, None)
            t = timed_run(lambda: impl_llmm1(x_view, weight, None), warmup, iters)
            results["llmm1"] = t
        except Exception as e:
            results["llmm1"] = f"ERROR: {e}"

    # ── linear (always valid; TunableOps may intercept) ──
    try:
        impl_linear(x, weight, None)
        t = timed_run(lambda: impl_linear(x, weight, None), warmup, iters)
        results["linear"] = t
    except Exception as e:
        results["linear"] = f"ERROR: {e}"

    # Pick winner (lowest numeric time)
    valid = {k: v for k, v in results.items() if isinstance(v, float)}
    if not valid:
        winner = "linear"
        baseline = results.get("linear", 0)
    else:
        winner = min(valid, key=valid.__getitem__)
        baseline = valid.get("linear", min(valid.values()))

    winner_time = valid.get(winner, 0)
    speedup = baseline / winner_time if winner_time > 0 else 1.0

    return {
        "n": n, "k": k, "m": m,
        "timings_us": {k: round(v, 2) if isinstance(v, float) else v
                       for k, v in results.items()},
        "winner": winner,
        "winner_time_us": round(winner_time, 2),
        "speedup_vs_linear": round(speedup, 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM implementations")
    parser.add_argument("--shapes",  required=True, help="real_shapes.json")
    parser.add_argument("--output",  required=True, help="routing_table.json output")
    parser.add_argument("--warmup",  type=int, default=20)
    parser.add_argument("--iters",   type=int, default=100)
    parser.add_argument("--dtype",   default="bfloat16",
                        choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    shapes_data = json.load(open(args.shapes))
    # Support both {shapes: [{MKN:[n,k,m],...}]} and {benchmark_shapes: [[[n,k],[k,m]],...]}
    raw_shapes = shapes_data.get("shapes") or []
    mkn_list = [s["MKN"] for s in raw_shapes if "MKN" in s]

    if not mkn_list:
        # fallback: benchmark_shapes format
        for pair in shapes_data.get("benchmark_shapes", []):
            n, k = pair[0]
            k2, m = pair[1]
            mkn_list.append([n, k, m])

    if not mkn_list:
        print("ERROR: no shapes found in input file", file=sys.stderr)
        sys.exit(1)

    cu_count = get_cu_count()
    print(f"GPU CU count: {cu_count}")
    print(f"Benchmarking {len(mkn_list)} shapes  "
          f"warmup={args.warmup}  iters={args.iters}  dtype={args.dtype}")
    print()

    print(f"{'n':>4} {'k':>6} {'m':>6}  {'wvSplitK':>10}  {'llmm1':>10}  "
          f"{'linear':>10}  {'winner':<12}  speedup")
    print("-" * 80)

    routing = {}
    results_list = []

    for n, k, m in mkn_list:
        res = benchmark_shape(n, k, m, dtype, args.warmup, args.iters, cu_count)
        t = res["timings_us"]
        wv  = f"{t.get('wvSplitK', '-'):>10}" if isinstance(t.get('wvSplitK'), float) \
              else f"{'N/A':>10}"
        ll  = f"{t.get('llmm1', '-'):>10}" if isinstance(t.get('llmm1'), float) \
              else f"{'N/A':>10}"
        ln  = f"{t.get('linear', '-'):>10}" if isinstance(t.get('linear'), float) \
              else f"{'ERR':>10}"
        print(f"{n:>4} {k:>6} {m:>6}  {wv}  {ll}  {ln}  "
              f"{res['winner']:<12}  {res['speedup_vs_linear']:.3f}x")

        key = f"{n},{k},{m}"
        routing[key] = {
            "impl":    res["winner"],
            "time_us": res["winner_time_us"],
            "speedup_vs_linear": res["speedup_vs_linear"],
        }
        results_list.append(res)

    # Summary
    winners = {}
    for r in results_list:
        w = r["winner"]
        winners[w] = winners.get(w, 0) + 1
    print()
    print("Winner distribution:", winners)

    avg_speedup = sum(r["speedup_vs_linear"] for r in results_list) / len(results_list)
    print(f"Avg speedup vs linear: {avg_speedup:.3f}x")

    # Save routing table
    output = {
        "dtype": args.dtype,
        "n_shapes": len(routing),
        "routing": routing,
        "details": results_list,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRouting table saved: {args.output}")


if __name__ == "__main__":
    main()
