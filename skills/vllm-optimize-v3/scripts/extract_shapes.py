#!/usr/bin/env python3
"""
Extract real GEMM shapes (M, K, N) from PyTorch profiler trace files.

The profiler must have been started with record_shapes=True. Looks for
aten::mm, aten::addmm, aten::linear, aten::matmul events that carry
input shape metadata.

This satisfies Constraint 1: all shapes used in kernel optimization
come from actual profiler data, not model config guesses.

Usage:
    python extract_shapes.py --trace-dir ./profiles --output ./real_shapes.json
"""
import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(path: str):
    """Load a Chrome-trace JSON file (.json or .json.gz)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    if isinstance(data, list):
        return data
    return []


def find_traces(trace_dir: str):
    """Find valid trace files, filtering out frontend/log files."""
    candidates = []
    for name in sorted(os.listdir(trace_dir)):
        if not (name.endswith(".json") or name.endswith(".json.gz")):
            continue
        if "async_llm" in name.lower() or "_docker.log" in name:
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


def extract_shapes_from_events(events):
    """
    Extract GEMM-relevant (M, K, N) shapes from trace events.

    PyTorch records shapes as a list in event["args"]["Input Dims"] or
    event["args"]["input_dims"] when record_shapes=True.
    """
    # Target ops that correspond to GEMM
    GEMM_OPS = {"aten::mm", "aten::addmm", "aten::linear", "aten::matmul",
                "aten::bmm", "aten::baddbmm"}

    shape_counts = defaultdict(int)  # (M, K, N) -> count

    for ev in events:
        name = ev.get("name", "")
        if name not in GEMM_OPS:
            continue
        args = ev.get("args", {})
        # Shape is stored under various keys depending on PyTorch version
        dims = args.get("Input Dims") or args.get("input_dims") or args.get("shapes") or []
        if not dims or not isinstance(dims, list):
            continue

        # aten::mm: inputs = [A (M,K), B (K,N)]
        # aten::addmm: inputs = [bias, A (M,K), B (K,N)]  → skip bias
        # aten::linear: inputs = [input (M,K), weight (N,K)] → B is transposed
        try:
            if name == "aten::mm":
                if len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
                    M, K = dims[0]
                    K2, N = dims[1]
                    if K == K2 and M > 0 and K > 0 and N > 0:
                        shape_counts[(M, K, N)] += 1

            elif name == "aten::addmm":
                if len(dims) >= 3 and len(dims[1]) == 2 and len(dims[2]) == 2:
                    M, K  = dims[1]
                    K2, N = dims[2]
                    if K == K2 and M > 0 and K > 0 and N > 0:
                        shape_counts[(M, K, N)] += 1

            elif name == "aten::linear":
                if len(dims) >= 2 and len(dims[0]) >= 2 and len(dims[1]) == 2:
                    # input: (..., K), weight: (N, K) — weight is transposed
                    K  = dims[0][-1]
                    N, K2 = dims[1]
                    M  = 1
                    for d in dims[0][:-1]:
                        M *= d
                    if K == K2 and M > 0 and K > 0 and N > 0:
                        shape_counts[(M, K, N)] += 1

            elif name in ("aten::matmul", "aten::bmm", "aten::baddbmm"):
                if len(dims) >= 2 and len(dims[0]) >= 2 and len(dims[1]) >= 2:
                    K  = dims[0][-1]
                    N  = dims[1][-1]
                    M  = 1
                    for d in dims[0][:-1]:
                        M *= d
                    if M > 0 and K > 0 and N > 0:
                        shape_counts[(M, K, N)] += 1
        except (IndexError, TypeError, ValueError):
            continue

    return shape_counts


def build_output(shape_counts):
    """Format shape data for use by kernel_agent.py."""
    if not shape_counts:
        return {"benchmark_shapes": [], "unique_m_values": [],
                "shape_call_counts": {}, "total_shapes": 0}

    # Sort by call count descending
    sorted_shapes = sorted(shape_counts.items(), key=lambda x: -x[1])

    # benchmark_shapes: list of [[M, K], [K, N]] pairs (what kernel_agent uses)
    benchmark_shapes = []
    shape_call_counts = {}
    seen = set()
    for (M, K, N), count in sorted_shapes:
        key = (M, K, N)
        if key in seen:
            continue
        seen.add(key)
        benchmark_shapes.append([[M, K], [K, N]])
        shape_call_counts[f"[{M}, {K}]"] = count

    unique_m = sorted(set(s[0][0] for s in benchmark_shapes))

    return {
        "benchmark_shapes": benchmark_shapes,
        "unique_m_values": unique_m,
        "shape_call_counts": shape_call_counts,
        "total_shapes": len(benchmark_shapes),
        "top_shapes_by_calls": [
            {"MKN": list(mkn), "calls": cnt}
            for mkn, cnt in sorted_shapes[:20]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract real GEMM shapes from profiler traces")
    parser.add_argument("--trace-dir", required=True, help="Directory with .json.gz trace files")
    parser.add_argument("--output",    required=True, help="Output JSON path")
    args = parser.parse_args()

    traces = find_traces(args.trace_dir)
    if not traces:
        print(f"ERROR: No valid trace files found in {args.trace_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(traces)} trace file(s). Extracting shapes...")

    all_shapes = defaultdict(int)
    for trace_path in traces:
        events = load_trace(trace_path)
        shapes = extract_shapes_from_events(events)
        for shape, count in shapes.items():
            all_shapes[shape] += count
        print(f"  {os.path.basename(trace_path)}: {len(shapes)} shape types")

    result = build_output(all_shapes)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nExtracted {result['total_shapes']} unique shapes")
    print(f"M values: {result['unique_m_values']}")
    print(f"Saved: {args.output}")

    if result["total_shapes"] == 0:
        print("\nWARNING: No shapes found. Verify the profiler was started with")
        print("  torch_profiler_record_shapes=True in the --profiler-config JSON.")
        sys.exit(2)  # exit 2 = warning (not fatal)


if __name__ == "__main__":
    main()
