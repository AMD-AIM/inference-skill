#!/usr/bin/env python3
"""Discover and validate torch profiler trace files in a directory.

Outputs key=value pairs to stdout for consumption by the calling agent/script:
  TRACE_COUNT, WORLD_SIZE, MERGED_TRACE, RANK0_FULL_TRACE, RANK0_EXTEND_TRACE,
  RANK0_DECODE_TRACE, TRACELENS_PRIMARY_TRACE, TRACELENS_PRIMARY_ROLE,
  PHASE_SPLIT_INPUT_TRACE, PHASE_SPLIT_INPUT_ROLE, COLLECTIVE_TRACE_MODE

Optionally writes a JSON manifest with --output.
"""

import argparse
import gzip
import glob
import json
import os
import re
import sys

PEEK_BYTES = 65536  # 64 KB decompressed — covers all metadata before traceEvents


def discover_traces(trace_dir):
    valid_traces = []
    rank_ids = set()
    tp_extend_ranks = set()
    full_rank_ranks = set()
    merged_trace = ""
    rank0_full_trace = ""
    rank0_extend_trace = ""
    rank0_decode_trace = ""

    for f in sorted(glob.glob(os.path.join(trace_dir, "*.json*"))):
        basename = os.path.basename(f)
        lower_name = basename.lower()
        if "_docker.log" in basename:
            continue
        if "async_llm" in lower_name:
            print(f"SKIPPED (async_llm frontend trace): {f}")
            continue
        try:
            opener = gzip.open if f.endswith(".gz") else open
            with opener(f, "rt") as fh:
                prefix = fh.read(PEEK_BYTES)
            if '"traceEvents"' not in prefix:
                print(f"SKIPPED (no traceEvents key in first {PEEK_BYTES} bytes): {f}")
                continue

            trace_role = "other"
            rank = None
            phase = None

            merged_match = lower_name.startswith("merged-") or re.search(
                r"(^|[-_])merged(?:[-_.]|$)", lower_name
            )
            tp_phase_match = re.search(
                r"(?:^|[-_])tp[-_]?(\d+)[-_](decode|extend)(?:[-_.]|$)",
                basename,
                re.I,
            )
            rank_match = re.search(
                r"(?:^|[-_])rank[-_]?(\d+)(?:[-_.]|$)", basename, re.I
            )

            if merged_match:
                trace_role = "merged"
                if not merged_trace:
                    merged_trace = f
            elif tp_phase_match:
                trace_role = "tp-phase"
                rank = int(tp_phase_match.group(1))
                phase = tp_phase_match.group(2).upper()
                rank_ids.add(rank)
                if phase == "EXTEND":
                    tp_extend_ranks.add(rank)
                if rank == 0 and phase == "EXTEND" and not rank0_extend_trace:
                    rank0_extend_trace = f
                if rank == 0 and phase == "DECODE" and not rank0_decode_trace:
                    rank0_decode_trace = f
            elif rank_match:
                trace_role = "rank"
                rank = int(rank_match.group(1))
                rank_ids.add(rank)
                full_rank_ranks.add(rank)
                if rank == 0 and not rank0_full_trace:
                    rank0_full_trace = f

            valid_traces.append(f)
            size_mb = os.path.getsize(f) / (1024 * 1024)
            descriptor = [trace_role]
            if rank is not None:
                descriptor.append(f"rank {rank}")
            if phase:
                descriptor.append(phase)
            print(
                f"VALID torch trace ({', '.join(descriptor)}, {size_mb:.1f} MB compressed): {f}"
            )
        except Exception as e:
            print(f"ERROR reading {f}: {e}")

    # Determine primary trace for TraceLens
    tracelens_primary_trace = ""
    tracelens_primary_role = ""
    phase_split_input_trace = ""
    phase_split_input_role = "unavailable"
    collective_trace_mode = "unavailable"

    if valid_traces:
        if rank0_extend_trace:
            tracelens_primary_trace = rank0_extend_trace
            tracelens_primary_role = "rank0-extend"
        elif rank0_full_trace:
            tracelens_primary_trace = rank0_full_trace
            tracelens_primary_role = "rank0-full"
        elif merged_trace:
            tracelens_primary_trace = merged_trace
            tracelens_primary_role = "merged-fallback"
        else:
            tracelens_primary_trace = valid_traces[0]
            tracelens_primary_role = "first-valid-fallback"

        if merged_trace:
            phase_split_input_trace = merged_trace
            phase_split_input_role = "merged"
        elif rank0_full_trace:
            phase_split_input_trace = rank0_full_trace
            phase_split_input_role = "rank0-full"

        world_size = len(rank_ids) if rank_ids else 1
        if world_size > 1 and len(tp_extend_ranks) == world_size:
            collective_trace_mode = "tp-extend"
        elif world_size > 1 and len(full_rank_ranks) == world_size:
            collective_trace_mode = "rank-full"
    else:
        world_size = 0

    result = {
        "valid_traces": valid_traces,
        "trace_count": len(valid_traces),
        "world_size": world_size,
        "merged_trace": merged_trace,
        "rank0_full_trace": rank0_full_trace,
        "rank0_extend_trace": rank0_extend_trace,
        "rank0_decode_trace": rank0_decode_trace,
        "tracelens_primary_trace": tracelens_primary_trace,
        "tracelens_primary_role": tracelens_primary_role,
        "phase_split_input_trace": phase_split_input_trace,
        "phase_split_input_role": phase_split_input_role,
        "collective_trace_mode": collective_trace_mode,
    }

    # Print key=value summary for agent consumption
    print(f"TRACE_COUNT={result['trace_count']}")
    if valid_traces:
        print(f"WORLD_SIZE={world_size}")
        print(f"COLLECTIVE_TRACE_MODE={collective_trace_mode}")
        print(f"MERGED_TRACE={merged_trace}")
        print(f"RANK0_FULL_TRACE={rank0_full_trace}")
        print(f"RANK0_EXTEND_TRACE={rank0_extend_trace}")
        print(f"RANK0_DECODE_TRACE={rank0_decode_trace}")
        print(f"TRACELENS_PRIMARY_TRACE={tracelens_primary_trace}")
        print(f"TRACELENS_PRIMARY_ROLE={tracelens_primary_role}")
        print(f"PHASE_SPLIT_INPUT_TRACE={phase_split_input_trace}")
        print(f"PHASE_SPLIT_INPUT_ROLE={phase_split_input_role}")
    else:
        print(
            "WARNING: No valid torch profiler traces found. Trace analysis will be skipped."
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Discover and validate trace files")
    parser.add_argument("--trace-dir", required=True, help="Directory containing trace files")
    parser.add_argument("--output", help="Optional JSON manifest output path")
    args = parser.parse_args()

    result = discover_traces(args.trace_dir)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"MANIFEST_JSON={args.output}")


if __name__ == "__main__":
    main()
