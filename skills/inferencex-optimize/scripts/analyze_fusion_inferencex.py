#!/usr/bin/env python3
"""
Operator Fusion Analysis for InferenceX pipeline.

Reads gap_analysis.json and optional TraceLens CSVs to detect fusable operator
patterns and generate fusion_opportunities.json + bottleneck_kernels.json.
When --gpu-arch is provided, also produces roofline_bottlenecks.json.

Usage:
    python analyze_fusion_inferencex.py \
        --gap-analysis results/gap_analysis/gap_analysis.json \
        --tracelens-dir results/tracelens_rank0_csvs \
        --framework vllm \
        --output-dir problems/

Part of the inferencex-optimize skill. Can be used standalone.
"""
import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)

from classify_kernel import (
    classify_kernel,
    load_unified_perf_summary,
    build_roofline_bottlenecks,
)


def load_gap_analysis(path):
    """Load gap_analysis.json and return structured kernel data."""
    with open(path) as f:
        data = json.load(f)
    kernels = []
    for k in data.get("top_kernels", []):
        kernels.append({
            "name": k["name"],
            "calls": k.get("calls", 0),
            "total_us": k.get("self_cuda_total_us", k.get("total_us", 0)),
            "avg_us": k.get("avg_time_us", k.get("avg_us", 0)),
            "pct": k.get("pct_total", k.get("pct", 0)),
        })
    category_breakdown = data.get("category_breakdown", {})
    return kernels, category_breakdown


def load_tracelens_ops_by_category(tracelens_dir):
    """Load ops_summary_by_category.csv if available."""
    path = os.path.join(tracelens_dir, "ops_summary_by_category.csv")
    if not os.path.isfile(path):
        return {}
    categories = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cat = row.get("op category", "")
            pct = float(row.get("Percentage (%)", 0))
            categories[cat] = pct
    return categories


def load_tracelens_kernel_summary(tracelens_dir):
    """Load kernel_summary.csv if available."""
    path = os.path.join(tracelens_dir, "kernel_summary.csv")
    if not os.path.isfile(path):
        return []
    kernels = []
    with open(path) as f:
        for row in csv.DictReader(f):
            kernels.append({
                "name": row.get("name", ""),
                "parent_op": row.get("Parent cpu_op", ""),
                "parent_category": row.get("Parent op category", ""),
                "calls": int(float(row.get("Calls", 0) or 0)),
                "total_us": float(row.get("Total time (us)", 0) or 0),
            })
    return kernels


def detect_fusion_opportunities(kernels, category_breakdown, tracelens_categories, framework):
    """Detect fusable operator patterns from kernel and category data."""
    fusions = []

    categories = {}
    if tracelens_categories:
        categories = tracelens_categories
    elif category_breakdown:
        total = sum(v["pct"] if isinstance(v, dict) else v for v in category_breakdown.values())
        if total > 0:
            categories = {k: (v["pct"] if isinstance(v, dict) else v) / total * 100 for k, v in category_breakdown.items()}

    kernel_names_lower = [k["name"].lower() for k in kernels]
    kernel_pcts = {k["name"].lower(): k["pct"] for k in kernels}

    has_add = any("add" in n for n in kernel_names_lower)
    has_norm = any("norm" in n or "mean" in n or "rsqrt" in n for n in kernel_names_lower)
    norm_cat_pct = categories.get("Normalization", 0) + categories.get("rmsnorm", 0)
    if (has_add and has_norm) or norm_cat_pct > 2:
        add_pct = sum(p for n, p in kernel_pcts.items() if "add" in n and "norm" not in n)
        norm_pct = sum(p for n, p in kernel_pcts.items()
                       if "norm" in n or "mean" in n or "rsqrt" in n)
        combined = max(add_pct + norm_pct, norm_cat_pct)
        entry = {
            "name": "fused_residual_rmsnorm",
            "operators": ["residual add", "RMSNorm (mean, rsqrt, mul)"],
            "combined_percent": round(combined, 1),
            "expected_speedup": "1.3-1.5x",
            "priority": "HIGH" if combined > 5 else "MEDIUM",
        }
        # Check if already fused by a vendor kernel (e.g., aiter::add_rmsnorm)
        for n, p in kernel_pcts.items():
            if "add_rmsnorm" in n:
                entry["already_fused"] = True
                entry["fused_kernel"] = n
                entry["fused_kernel_pct"] = round(p, 2)
                break
        fusions.append(entry)

    has_silu = any("silu" in n for n in kernel_names_lower)
    has_mul = any("mul" in n and "norm" not in n for n in kernel_names_lower)
    if has_silu and has_mul:
        combined = sum(p for n, p in kernel_pcts.items()
                       if "silu" in n or ("mul" in n and "norm" not in n))
        entry = {
            "name": "fused_swiglu",
            "operators": ["silu", "mul"],
            "combined_percent": round(combined, 1),
            "expected_speedup": "1.3-1.8x",
            "priority": "MEDIUM",
        }
        # Check if already fused by a vendor kernel (e.g., aiter::fused_moe_,
        # ck::kernel_moe_mxgemm, MoeFlatmmKernel with MoeSilu)
        for n, p in kernel_pcts.items():
            if "fused_moe" in n or "swiglu" in n or "kernel_moe" in n or "moeflatmm" in n:
                entry["already_fused"] = True
                entry["fused_kernel"] = n
                entry["fused_kernel_pct"] = round(p, 2)
                break
        fusions.append(entry)

    gemm_count = sum(1 for n in kernel_names_lower if "mm" in n or "gemm" in n)
    if gemm_count >= 3:
        fusions.append({
            "name": "fused_qkv_proj",
            "operators": ["3x GEMM for Q, K, V projections"],
            "combined_percent": 0,
            "expected_speedup": "1.2-1.4x",
            "priority": "LOW",
        })

    return fusions


def build_bottleneck_kernels(kernels, tracelens_kernels, threshold_pct):
    """Build classified bottleneck kernel list."""
    parent_cats = {}
    for tk in tracelens_kernels:
        parent_cats[tk["name"]] = tk.get("parent_category", "")

    bottlenecks = []
    for k in kernels:
        if k["pct"] < threshold_pct:
            continue
        parent_cat = parent_cats.get(k["name"], "")
        kernel_type, reason = classify_kernel(k["name"], parent_cat)
        bottlenecks.append({
            "name": k["name"],
            "calls": k["calls"],
            "total_us": k["total_us"],
            "pct": k["pct"],
            "kernel_type": kernel_type,
            "reason": reason,
            "parent_category": parent_cat,
            "optimizable": kernel_type not in ("communication",),
        })

    return bottlenecks


def main():
    parser = argparse.ArgumentParser(description="Analyze fusion opportunities from InferenceX profiling data")
    parser.add_argument("--gap-analysis", required=True, help="Path to gap_analysis.json")
    parser.add_argument("--tracelens-dir", default="", help="Path to tracelens_rank0_csvs/ directory")
    parser.add_argument("--decode-tracelens-dir", default="", help="Path to tracelens_decode_only_csvs/ directory")
    parser.add_argument("--framework", default="vllm", choices=["vllm", "sglang"], help="Inference framework")
    parser.add_argument("--threshold", type=float, default=1.0, help="Min %% of total time for bottleneck (default: 1.0)")
    parser.add_argument("--gpu-arch", default="", help="Path to gpu_arch.json for roofline analysis")
    parser.add_argument("--model-precision", default="", help="Model precision (e.g. fp4, bf16, fp8) for spec inference")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading gap analysis: {args.gap_analysis}")
    kernels, category_breakdown = load_gap_analysis(args.gap_analysis)
    print(f"  Found {len(kernels)} kernels, {len(category_breakdown)} categories")

    tracelens_categories = {}
    tracelens_kernels = []
    tl_dir = None
    if args.decode_tracelens_dir and os.path.isdir(args.decode_tracelens_dir):
        tl_dir = args.decode_tracelens_dir
        print(f"Loading decode-only TraceLens data: {tl_dir}")
    elif args.tracelens_dir and os.path.isdir(args.tracelens_dir):
        tl_dir = args.tracelens_dir
        print(f"Loading TraceLens data: {tl_dir}")
    if tl_dir:
        tracelens_categories = load_tracelens_ops_by_category(tl_dir)
        tracelens_kernels = load_tracelens_kernel_summary(tl_dir)
        print(f"  Categories: {len(tracelens_categories)}, Kernels: {len(tracelens_kernels)}")
    else:
        print("  No TraceLens data available, using gap analysis categories only")

    fusions = detect_fusion_opportunities(kernels, category_breakdown, tracelens_categories, args.framework)
    fusions_path = os.path.join(args.output_dir, "fusion_opportunities.json")
    with open(fusions_path, "w") as f:
        json.dump(fusions, f, indent=2)

    print(f"\n=== Fusion Opportunities ({len(fusions)}) ===")
    for fu in sorted(fusions, key=lambda x: x["combined_percent"], reverse=True):
        print(f"  [{fu['priority']}] {fu['name']} ({fu['combined_percent']:.1f}%)")
        print(f"    Operators: {', '.join(fu['operators'])}")
        print(f"    Expected speedup: {fu['expected_speedup']}")
    print(f"  Saved to {fusions_path}")

    bottlenecks = build_bottleneck_kernels(kernels, tracelens_kernels, args.threshold)
    bottlenecks_path = os.path.join(args.output_dir, "bottleneck_kernels.json")
    with open(bottlenecks_path, "w") as f:
        json.dump(bottlenecks, f, indent=2)

    print(f"\n=== Bottleneck Kernels ({len(bottlenecks)}) ===")
    print(f"  {'#':>3s}  {'Kernel':50s} {'%':>6s}  {'Type':15s}  {'Reason'}")
    print(f"  {'─' * 90}")
    for i, b in enumerate(bottlenecks[:20], 1):
        flag = "✗ " if not b["optimizable"] else "  "
        name = b["name"][:48]
        print(f"  {i:3d}. {flag}{name:<48s} {b['pct']:5.1f}%  {b['kernel_type']:15s}  {b['reason']}")
    print(f"  Saved to {bottlenecks_path}")

    gpu_arch = None
    if args.gpu_arch and os.path.isfile(args.gpu_arch):
        with open(args.gpu_arch) as f:
            gpu_arch = json.load(f)
        print(f"\nLoaded gpu_arch: {gpu_arch.get('name', 'unknown')}")

    if gpu_arch and tl_dir:
        unified_ops = load_unified_perf_summary(tl_dir)
        if unified_ops:
            roofline_bots = build_roofline_bottlenecks(
                unified_ops, gpu_arch, args.model_precision or None
            )
            roofline_path = os.path.join(args.output_dir, "roofline_bottlenecks.json")
            with open(roofline_path, "w") as f:
                json.dump(roofline_bots, f, indent=2)

            print(f"\n=== Roofline Bottlenecks ({len(roofline_bots)}) ===")
            print(f"  {'#':>3s}  {'Op':45s} {'%':>6s}  {'Spec':18s}  {'Conf':14s}  {'Eff':>8s}")
            print(f"  {'─' * 100}")
            for i, b in enumerate(sorted(roofline_bots, key=lambda x: x["pct"], reverse=True)[:20], 1):
                eff = f"{b['roofline_efficiency']:.1f}%" if b["roofline_efficiency"] is not None else "N/A"
                name = b["name"][:43]
                print(f"  {i:3d}. {name:<45s} {b['pct']:5.1f}%  {b['compute_spec']:18s}  {b['spec_confidence']:14s}  {eff:>8s}")
            print(f"  Saved to {roofline_path}")
        else:
            print("\n  No unified_perf_summary.csv found — skipping roofline bottlenecks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
