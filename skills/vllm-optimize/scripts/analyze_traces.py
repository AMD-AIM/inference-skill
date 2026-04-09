#!/usr/bin/env python3
"""Analyze vLLM profiler traces: gap analysis + kernel categorization.

Usage: python3 analyze_traces.py --trace <path> [--output-dir <dir>]
"""

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

KERNEL_PATTERNS = [
    "wvSplitK", "ck_tile", "ck::kernel", "Cijk_", "__amd_", "rocclr",
    "gemm", "mm", "matmul", "flash_attn", "attention", "paged_attention",
    "allreduce", "allgather", "broadcast", "reduce_scatter",
    "softmax", "layernorm", "rmsnorm", "norm",
    "moe", "experts", "topk", "routing", "sorting",
    "triton_", "triton_red", "triton_fused",
    "vectorized_elementwise", "elementwise_kernel",
    "copyBuffer", "memcpy", "memset",
    "fused_recurrent", "causal_conv", "mamba",
    "gdn_", "act_and_mul", "silu", "swiglu",
]

EXCLUDE_PATTERNS = [
    "execute_context", "profiler", "frontend", "python", "Module:",
    "vllm/model_executor", "vllm::moe", "vllm::gdn",
    "triton/runtime/jit", "pybind11", "builtin method",
    "nn.Module", "forward", "layer.py", "runner",
]

CATEGORIES = {
    "MoE": ["wvSplitK", "moe", "ck_tile", "topk", "sorting", "Cijk_"],
    "Attention": ["attention", "paged_attention", "flash"],
    "Normalization": ["layernorm", "rmsnorm", "norm", "triton_red"],
    "Memory": ["copyBuffer", "memcpy", "memset"],
    "Activation": ["silu", "swiglu", "act_and_mul", "gdn", "mamba", "fused_recurrent"],
    "Elementwise": ["vectorized_elementwise", "elementwise_kernel"],
}


def find_trace(profile_dir):
    for f in sorted(glob.glob(f"{profile_dir}/*.json*")):
        if "rocprof" in f.lower():
            continue
        try:
            with gzip.open(f, "rt") as fh:
                content = fh.read(65536)
            if '"traceEvents"' in content:
                return f
        except Exception:
            pass
    return None


def analyze(trace_file):
    opener = gzip.open if trace_file.endswith(".gz") else open
    with opener(trace_file, "rt") as f:
        data = json.load(f)

    kernel_times = defaultdict(lambda: {"total_us": 0, "count": 0})

    for e in data.get("traceEvents", []):
        if "dur" not in e or e.get("dur", 0) <= 0:
            continue
        name = e.get("name", "")
        cat = e.get("cat", "").lower()
        if cat not in ("kernel", "cuda", "gpu"):
            continue
        if any(p.lower() in name.lower() for p in EXCLUDE_PATTERNS):
            continue
        if any(p.lower() in name.lower() for p in KERNEL_PATTERNS):
            kernel_times[name]["total_us"] += e["dur"]
            kernel_times[name]["count"] += 1

    sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1]["total_us"], reverse=True)
    total_time = sum(s["total_us"] for _, s in sorted_kernels)

    # Categorize
    category_time = {}
    for name, stats in sorted_kernels:
        cat = "Other"
        for c, patterns in CATEGORIES.items():
            if any(p.lower() in name.lower() for p in patterns):
                cat = c
                break
        category_time[cat] = category_time.get(cat, 0) + stats["total_us"]

    return sorted_kernels, total_time, category_time


def main():
    parser = argparse.ArgumentParser(description="Analyze vLLM traces")
    parser.add_argument("--trace", help="Direct path to trace file")
    parser.add_argument("--profile-dir", help="Directory to search for traces")
    parser.add_argument("--output-dir", default="./vllm_results/gap_analysis")
    args = parser.parse_args()

    trace_file = args.trace
    if not trace_file and args.profile_dir:
        trace_file = find_trace(args.profile_dir)
    if not trace_file:
        print("ERROR: No valid trace file found")
        return

    print(f"Analyzing: {trace_file}")
    sorted_kernels, total_time, category_time = analyze(trace_file)

    print(f"\nTotal GPU kernel time: {total_time/1000:.2f} ms")
    print(f"Unique kernels: {len(sorted_kernels)}\n")

    kernels_json = []
    for i, (name, stats) in enumerate(sorted_kernels[:25]):
        pct = stats["total_us"] / total_time * 100 if total_time > 0 else 0
        print(f"{i+1}. {name[:55]} | {stats['count']} calls | {stats['total_us']/1000:.2f}ms | {pct:.2f}%")
        kernels_json.append({
            "name": name, "calls": stats["count"],
            "total_us": stats["total_us"], "pct": round(pct, 2),
        })

    print("\n=== Kernel Category Breakdown ===")
    for cat, time_us in sorted(category_time.items(), key=lambda x: -x[1]):
        print(f"{cat}: {time_us/1000:.2f}ms ({time_us/total_time*100:.1f}%)")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "gap_analysis.json"), "w") as f:
        json.dump({"kernels": kernels_json, "categories": category_time, "total_us": total_time}, f, indent=2)
    print(f"\nSaved to {args.output_dir}/gap_analysis.json")


if __name__ == "__main__":
    main()
