#!/usr/bin/env python3
"""
Extract real GEMM/linear shapes from torch profiler traces.

Reads traces collected with record_shapes=True and extracts
all (M, K, N) combinations actually dispatched during inference.

Output: JSON file mapping kernel type to list of {shape, count, pct}.

Usage:
    python extract_trace_shapes.py --trace-dir profiles/ --output results/real_shapes.json
"""

import argparse
import gzip
import glob
import json
import os
import re
from collections import Counter, defaultdict


def extract_shapes(trace_path):
    """Extract Input Dims from a trace file.
    
    The trace is a large JSON array of events. Each event that is an aten::mm/linear
    has "Input Dims" in its args. We search for these patterns in chunks, allowing
    the name and Input Dims to be in separate regex passes if needed.
    """
    shapes = Counter()  # (op, M, K, N) -> count
    chunk_size = 50_000_000  # 50MB

    opener = gzip.open if trace_path.endswith('.gz') else open
    with opener(trace_path, 'rt') as f:
        overlap = ""
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            text = overlap + data

            # Strategy: find each "Input Dims": [...] and look backwards for the op name.
            # This is more robust than trying to match both in one regex across event boundaries.
            for m in re.finditer(r'"Input Dims":\s*(\[\[[\d, ]+\](?:,\s*\[[\d, ]+\])*\])', text):
                pos = m.start()
                # Look backwards up to 1000 chars for the op name
                context = text[max(0, pos - 1000):pos]
                op_match = re.search(r'"name":\s*"(aten::(?:mm|addmm|linear|matmul))"', context)
                if not op_match:
                    continue
                op = op_match.group(1)
                try:
                    dims = json.loads(m.group(1))
                    if len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
                        M, K = dims[0]
                        K2, N = dims[1]
                        shapes[(op, M, K, N)] += 1
                except (json.JSONDecodeError, ValueError):
                    pass

            # Keep overlap for cross-boundary matches
            overlap = text[-5000:]

    return shapes


def main():
    parser = argparse.ArgumentParser(description="Extract real GEMM shapes from traces")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    traces = sorted(
        glob.glob(os.path.join(args.trace_dir, "*.json*")),
        key=lambda f: -os.path.getsize(f)
    )
    traces = [t for t in traces if not any(skip in os.path.basename(t)
              for skip in ['async_llm', '_docker.log'])]

    if not traces:
        print("ERROR: No trace files found")
        return

    print(f"Processing {len(traces)} trace file(s)...")

    all_shapes = Counter()
    for trace in traces:
        print(f"  {os.path.basename(trace)} ({os.path.getsize(trace)/1e6:.1f}MB)")
        shapes = extract_shapes(trace)
        all_shapes.update(shapes)

    total = sum(all_shapes.values())
    print(f"\nTotal GEMM calls: {total}")
    print(f"Unique (op, M, K, N) combos: {len(all_shapes)}")

    # Group by kernel type
    gemm_shapes = []
    for (op, M, K, N), count in all_shapes.most_common():
        pct = count / total * 100 if total > 0 else 0
        gemm_shapes.append({
            "op": op,
            "M": M, "K": K, "N": N,
            "count": count,
            "pct": round(pct, 2),
        })
        print(f"  {op:>15} M={M:>5} K={K:>5} N={N:>6}  {count:>6}x ({pct:.1f}%)")

    # Extract unique M values and shape pairs for benchmarking
    unique_ms = sorted(set(s["M"] for s in gemm_shapes))
    unique_nk = sorted(set((s["K"], s["N"]) for s in gemm_shapes))

    # Build benchmark shapes: each unique (M, K, N) actually observed
    bench_shapes = []
    for s in gemm_shapes:
        bench_shapes.append([[s["M"], s["K"]], [s["K"], s["N"]]])

    output = {
        "total_calls": total,
        "unique_shapes": len(all_shapes),
        "shapes": gemm_shapes,
        "unique_m_values": unique_ms,
        "unique_nk_pairs": [list(nk) for nk in unique_nk],
        "benchmark_shapes": bench_shapes,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")
    print(f"Unique M values: {unique_ms}")
    print(f"Unique (K,N) pairs: {unique_nk}")


if __name__ == "__main__":
    main()
