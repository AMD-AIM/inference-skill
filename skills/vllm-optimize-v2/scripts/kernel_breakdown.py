#!/usr/bin/env python3
"""
GPU Kernel Breakdown Analyzer for vLLM Profiler Traces.

Correctly separates GPU-side execution from CPU-side events.
Provides GPU utilization, idle gap analysis, and kernel category breakdown.

Key design principles:
- Only `cat='kernel'`, `cat='gpu_memcpy'`, `cat='gpu_memset'` events represent
  actual GPU execution. Their `dur` = real GPU time.
- `cat='cpu_op'`, `cat='cuda_runtime'` events are CPU-side. Their `dur` = CPU time.
  Mixing these with GPU durations is the #1 source of wrong profiling results.
- GPU utilization = union(GPU kernel intervals) / trace wall time
- GPU idle = wall time - GPU active (gaps where CPU hasn't dispatched next kernel)
- H2D/D2H transfers are GPU memcpy events, reported separately.

Usage:
    python kernel_breakdown.py --trace-dir ./profiles --output ./results/gap_analysis.json
    python kernel_breakdown.py --trace-dir ./profiles --output ./results/gap_analysis.json --top-n 30
"""

import argparse
import collections
import gzip
import json
import os
import sys


# ─── Kernel classifier ─────────────────────────────────────────────────────

# Classification order matters: more specific patterns first.
# GEMM patterns must NOT contain "mfma" — paged_attention kernels on AMD also
# use MFMA instructions and have "mfma" in their name (e.g. ll4mi_QKV_mfma16).
# Use library/function name prefixes instead.
KERNEL_CATEGORIES_ORDERED = [
    # Attention BEFORE GEMM — paged_attention names can contain "mfma"
    ("Attention",   ["paged_attention", "flash_attn", "_fwd_kernel", "mha_", "xformers", "fmha",
                     "attention_ll4mi", "vllm_flash"]),
    # GEMM: library-specific patterns, not hardware instruction names
    ("GEMM",        ["cijk_", "hipblaslt", "hipblas_", "wvsplitk", "ck_tile", "tensile",
                     "s_hgemm", "s_sgemm", "s_bgemm", "rocblas_gemm",
                     "Cijk_", "HipBlasLt", "WvSplitK"]),
    ("RMSNorm",     ["rms_norm", "rmsnorm", "layernorm", "fused_add_rms"]),
    ("Activation",  ["act_and_mul", "silu_and_mul", "swiglu", "gelu_and_mul"]),
    ("RoPE",        ["rotary_embedding", "rope_kernel"]),
    ("KV Cache",    ["reshape_and_cache", "cache_kernel", "copy_blocks"]),
    ("Sampling",    ["reduce_kernel", "topk", "sample", "softmax", "argmax"]),
    ("Element-wise",["elementwise_kernel", "vectorized_element", "unary_op"]),
]


def classify_kernel(name: str) -> str:
    n = name.lower()
    for cat, patterns in KERNEL_CATEGORIES_ORDERED:
        if any(p.lower() in n for p in patterns):
            return cat
    return "Other"


def classify_memop(name: str) -> str:
    n = name.lower()
    if "h2d" in n or "htod" in n:
        return "H2D"
    if "d2h" in n or "dtoh" in n:
        return "D2H"
    return "D2D"


# ─── Trace loading ─────────────────────────────────────────────────────────

def load_trace(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def find_traces(trace_dir: str):
    """Find valid trace files containing GPU kernel events."""
    candidates = []
    for name in sorted(os.listdir(trace_dir)):
        if not (name.endswith(".json") or name.endswith(".json.gz")):
            continue
        full = os.path.join(trace_dir, name)
        opener = gzip.open if name.endswith(".gz") else open
        try:
            with opener(full, "rt") as fh:
                peek = fh.read(4096)
            if '"traceEvents"' in peek:
                candidates.append(full)
        except Exception:
            continue
    return candidates


# ─── Core analysis ─────────────────────────────────────────────────────────

def analyze_events(events: list) -> dict:
    """
    Analyze trace events and return structured breakdown.

    Returns dict with:
      wall_us         : trace coverage (min GPU start → max GPU end)
      gpu_active_us   : union of all GPU kernel intervals (true compute time)
      gpu_idle_us     : wall - active (GPU waiting for CPU dispatch)
      h2d_us, d2h_us, d2d_us : transfer times
      by_category     : {cat: {us, calls}} for compute kernels
      top_kernels     : top N kernels by total duration
      gap_stats       : {median_us, max_us, total_us, n_gaps, n_gaps_gt_1ms}
      cpu_launch_us   : total cuda_runtime API call time (runs parallel to GPU)
      n_kernel_events : total GPU kernel event count
    """
    # Separate GPU execution events from CPU-side events
    gpu_kernels = []   # actual compute
    gpu_memcpy  = []   # H2D / D2H / D2D
    gpu_memset  = []   # memset

    for e in events:
        if e.get("ph") != "X" or e.get("dur", 0) <= 0:
            continue
        cat = e.get("cat", "")
        if cat == "kernel":
            gpu_kernels.append(e)
        elif cat == "gpu_memcpy":
            gpu_memcpy.append(e)
        elif cat == "gpu_memset":
            gpu_memset.append(e)

    all_gpu = gpu_kernels + gpu_memcpy + gpu_memset
    if not all_gpu:
        return {}

    # Trace wall time (based on GPU stream coverage)
    ts_min = min(e["ts"] for e in all_gpu)
    ts_max = max(e["ts"] + e["dur"] for e in all_gpu)
    wall_us = ts_max - ts_min

    # GPU active time: union of all GPU intervals (handles multi-stream overlap)
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in all_gpu)
    merged = []
    for s, end in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append([s, end])
    gpu_active_us = sum(end - s for s, end in merged)
    gpu_idle_us   = max(0, wall_us - gpu_active_us)

    # Transfer breakdown
    h2d_us = sum(e["dur"] for e in gpu_memcpy if "h2d" in e.get("name","").lower() or "htod" in e.get("name","").lower())
    d2h_us = sum(e["dur"] for e in gpu_memcpy if "d2h" in e.get("name","").lower() or "dtoh" in e.get("name","").lower())
    d2d_us = sum(e["dur"] for e in gpu_memcpy + gpu_memset
                 if "h2d" not in e.get("name","").lower()
                 and "htod" not in e.get("name","").lower()
                 and "d2h" not in e.get("name","").lower()
                 and "dtoh" not in e.get("name","").lower())

    # Per-category breakdown (compute kernels only, not memcpy)
    by_cat = collections.defaultdict(lambda: {"us": 0, "calls": 0})
    for e in gpu_kernels:
        cat = classify_kernel(e.get("name", ""))
        by_cat[cat]["us"]    += e["dur"]
        by_cat[cat]["calls"] += 1

    # Top kernels by total duration
    top_by_name = collections.defaultdict(lambda: {"total_us": 0, "calls": 0, "avg_us": 0.0})
    for e in gpu_kernels:
        name = e.get("name", "?")
        top_by_name[name]["total_us"] += e["dur"]
        top_by_name[name]["calls"]    += 1

    top_kernels = []
    for name, stats in sorted(top_by_name.items(), key=lambda x: -x[1]["total_us"]):
        stats["avg_us"] = round(stats["total_us"] / max(stats["calls"], 1), 1)
        stats["pct_active"] = round(stats["total_us"] / max(gpu_active_us, 1) * 100, 2)
        stats["category"] = classify_kernel(name)
        top_kernels.append({"name": name, **stats})

    # Inter-kernel gap analysis
    sorted_gpu = sorted(all_gpu, key=lambda e: e["ts"])
    gaps = []
    for i in range(1, len(sorted_gpu)):
        prev_end = sorted_gpu[i-1]["ts"] + sorted_gpu[i-1]["dur"]
        gap      = sorted_gpu[i]["ts"] - prev_end
        if gap > 0:
            gaps.append(gap)

    gap_stats = {}
    if gaps:
        gaps_sorted = sorted(gaps)
        n = len(gaps_sorted)
        big_gaps = [g for g in gaps_sorted if g > 1000]
        gap_stats = {
            "n_gaps":           n,
            "median_us":        round(gaps_sorted[n // 2], 1),
            "p95_us":           round(gaps_sorted[int(n * 0.95)], 1),
            "max_us":           round(gaps_sorted[-1], 1),
            "total_idle_us":    round(sum(gaps_sorted), 1),
            "n_gaps_gt_1ms":    len(big_gaps),
            "total_gt_1ms_us":  round(sum(big_gaps), 1),
        }

    # CPU launch overhead (runs in parallel with GPU — informational only)
    cpu_launch_us = sum(
        e.get("dur", 0) for e in events
        if e.get("cat") == "cuda_runtime" and e.get("ph") == "X" and e.get("dur", 0) > 0
    )

    return {
        "wall_us":       round(wall_us, 1),
        "gpu_active_us": round(gpu_active_us, 1),
        "gpu_idle_us":   round(gpu_idle_us, 1),
        "gpu_util_pct":  round(gpu_active_us / max(wall_us, 1) * 100, 1),
        "h2d_us":        round(h2d_us, 1),
        "d2h_us":        round(d2h_us, 1),
        "d2d_us":        round(d2d_us, 1),
        "by_category":   {
            cat: {
                "us":          round(d["us"], 1),
                "calls":       d["calls"],
                "pct_active":  round(d["us"] / max(gpu_active_us, 1) * 100, 1),
                "pct_wall":    round(d["us"] / max(wall_us, 1) * 100, 1),
            }
            for cat, d in sorted(by_cat.items(), key=lambda x: -x[1]["us"])
        },
        "top_kernels":      top_kernels[:50],
        "gap_stats":        gap_stats,
        "cpu_launch_us":    round(cpu_launch_us, 1),
        "n_kernel_events":  len(gpu_kernels),
    }


def build_gap_analysis(result: dict, top_n: int = 50) -> dict:
    """Convert analysis result to the gap_analysis.json schema used by Phase 3+."""
    cat = result.get("by_category", {})
    total_kernel_us = sum(d["us"] for d in cat.values())

    top_kernels = []
    for k in result.get("top_kernels", [])[:top_n]:
        top_kernels.append({
            "name":        k["name"],
            "total_us":    k["total_us"],
            "avg_us":      k["avg_us"],
            "calls":       k["calls"],
            "pct_total":   k["pct_active"],   # % of GPU active time
            "category":    k["category"],
        })

    return {
        "total_kernel_time_us":  total_kernel_us,
        "wall_us":               result["wall_us"],
        "gpu_active_us":         result["gpu_active_us"],
        "gpu_idle_us":           result["gpu_idle_us"],
        "gpu_util_pct":          result["gpu_util_pct"],
        "h2d_us":                result["h2d_us"],
        "d2h_us":                result["d2h_us"],
        "gap_stats":             result.get("gap_stats", {}),
        "cpu_launch_us":         result["cpu_launch_us"],
        "top_kernels":           top_kernels,
        "category_breakdown":    cat,
        "n_kernel_events":       result["n_kernel_events"],
    }


def print_report(result: dict, label: str = ""):
    hdr = f"  {label}" if label else "  Analysis"
    print(f"\n{'='*62}")
    print(hdr)
    print(f"{'='*62}")

    wall_s   = result["wall_us"] / 1e6
    active_s = result["gpu_active_us"] / 1e6
    idle_s   = result["gpu_idle_us"] / 1e6
    util_pct = result["gpu_util_pct"]

    print(f"\n  Trace wall:        {wall_s:6.3f}s  ({result['n_kernel_events']} kernel events)")
    print(f"\n  GPU utilization")
    print(f"    Active:          {active_s:6.3f}s  ({util_pct:5.1f}%)")
    print(f"    Idle (CPU gap):  {idle_s:6.3f}s  ({100-util_pct:5.1f}%)")

    gs = result.get("gap_stats", {})
    if gs:
        print(f"      median gap={gs.get('median_us',0):.0f}us  "
              f"max={gs.get('max_us',0):.0f}us  "
              f"N>1ms={gs.get('n_gaps_gt_1ms',0)} "
              f"({gs.get('total_gt_1ms_us',0)/1e6:.3f}s)")
        if gs.get("n_gaps_gt_1ms", 0) > 0 and gs.get("total_gt_1ms_us", 0) / max(result["wall_us"], 1) > 0.05:
            print(f"      ⚠ Large gaps detected — possible CPU scheduling bottleneck")

    h2d = result.get("h2d_us", 0)
    d2h = result.get("d2h_us", 0)
    if h2d > 0: print(f"    H2D transfer:    {h2d/1e6:.3f}s  ({h2d/result['wall_us']*100:.1f}%)")
    if d2h > 0: print(f"    D2H transfer:    {d2h/1e6:.3f}s  ({d2h/result['wall_us']*100:.1f}%)")

    print(f"\n  GPU kernel breakdown (of {active_s:.3f}s active):")
    print(f"    {'Category':<18s}  {'Time':>7}  {'% active':>8}  {'% wall':>7}")
    print(f"    {'-'*46}")
    for cat, d in result.get("by_category", {}).items():
        print(f"    {cat:<18s}  {d['us']/1e6:>6.3f}s  {d['pct_active']:>7.1f}%  {d['pct_wall']:>6.1f}%")

    cpu_launch_s = result.get("cpu_launch_us", 0) / 1e6
    print(f"\n  CPU launch API:    {cpu_launch_s:.3f}s  (async, parallel with GPU — not a gap source)")

    # ── Top-N individual kernels ─────────────────────────────────────────────
    top = result.get("top_kernels", [])
    if top:
        print(f"\n  Top {min(len(top), 20)} kernels by GPU active time:")
        print(f"    {'#':>3}  {'%active':>7}  {'total_s':>8}  {'calls':>7}  {'avg_us':>8}  {'cat':<12}  name")
        print(f"    {'-'*3}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*12}  {'-'*30}")
        for i, k in enumerate(top[:20], 1):
            name_short = k["name"][:80]
            print(f"    {i:>3}  {k['pct_active']:>6.1f}%  "
                  f"{k['total_us']/1e6:>7.3f}s  "
                  f"{k['calls']:>7d}  "
                  f"{k['avg_us']:>7.1f}us  "
                  f"{k.get('category','?'):<12}  "
                  f"{name_short}")


# ─── Multi-concurrency comparison ──────────────────────────────────────────

def bottleneck_verdict(result: dict) -> str:
    util    = result["gpu_util_pct"]
    gs      = result.get("gap_stats", {})
    big_gap = gs.get("total_gt_1ms_us", 0) / max(result["wall_us"], 1) * 100
    cat     = result.get("by_category", {})
    gemm_pct = cat.get("GEMM", {}).get("pct_active", 0)
    attn_pct = cat.get("Attention", {}).get("pct_active", 0)

    verdicts = []
    if util < 85:
        verdicts.append(f"GPU underutilized ({util:.0f}%) — CPU scheduling overhead")
        if big_gap > 5:
            verdicts.append(f"Large idle gaps ({big_gap:.0f}% of wall) — Python/scheduler latency")
    if gemm_pct > 70:
        verdicts.append(f"GEMM bound ({gemm_pct:.0f}% of active) — TunableOps / Triton optimization")
    if attn_pct > 20:
        verdicts.append(f"Attention bound ({attn_pct:.0f}% of active) — KV cache growth, flash-attn tuning")
    if not verdicts:
        verdicts.append("Balanced or mixed — further kernel-level profiling needed")
    return "; ".join(verdicts)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU kernel breakdown from PyTorch profiler traces (ROCm/CUDA)")
    parser.add_argument("--trace-dir",  required=True, help="Directory with .json or .json.gz trace files")
    parser.add_argument("--output",     required=True, help="Output JSON path (gap_analysis.json)")
    parser.add_argument("--top-n",      type=int, default=50, help="Top N kernels to include in output")
    parser.add_argument("--label",      default="",  help="Label for this trace (e.g. 'conc=64')")
    args = parser.parse_args()

    traces = find_traces(args.trace_dir)
    if not traces:
        print(f"ERROR: No valid trace files in {args.trace_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(traces)} trace file(s) in {args.trace_dir}")

    # Aggregate events across all trace files
    all_events = []
    for path in traces:
        evs = load_trace(path)
        sz  = os.path.getsize(path) / 1e6
        print(f"  {os.path.basename(path)}: {sz:.1f}MB, {len(evs)} events")
        all_events.extend(evs)

    result = analyze_events(all_events)
    if not result:
        print("ERROR: No GPU kernel events found in traces.", file=sys.stderr)
        print("  Verify the profiler was running during inference.", file=sys.stderr)
        sys.exit(1)

    print_report(result, label=args.label or f"Trace: {os.path.basename(args.trace_dir)}")

    # Print bottleneck verdict
    verdict = bottleneck_verdict(result)
    print(f"\n  Bottleneck: {verdict}")

    # Save structured output
    gap_analysis = build_gap_analysis(result, args.top_n)
    gap_analysis["label"] = args.label

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(gap_analysis, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
