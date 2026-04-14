#!/usr/bin/env python3
"""Discover and validate torch profiler trace files in a directory.

Outputs key=value pairs to stdout for consumption by the calling agent/script:
  TRACE_COUNT, WORLD_SIZE, MERGED_TRACE, RANK0_FULL_TRACE, RANK0_EXTEND_TRACE,
  RANK0_DECODE_TRACE, TRACELENS_PRIMARY_TRACE, TRACELENS_PRIMARY_ROLE,
  PHASE_SPLIT_INPUT_TRACE, PHASE_SPLIT_INPUT_ROLE, COLLECTIVE_TRACE_MODE

Always writes results/trace_manifest.json (structured manifest for monitor consumption).
Optionally writes a legacy JSON manifest with --output.
"""

import argparse
import datetime
import gzip
import glob
import json
import os
import re
import sys

PEEK_BYTES = 65536  # 64 KB decompressed — covers all metadata before traceEvents
ANNOTATION_PEEK_BYTES = 524288  # 512 KB — deeper peek for annotation markers
ANNOTATION_MARKERS = (
    "gpu_user_annotation",
    "user_annotation",
    "ForwardPass",
    "model_forward",
    "CUDAGraphRunner",
)


def _check_gzip_integrity(filepath):
    """Return 'valid' if gzip decompresses fully, 'corrupt' otherwise."""
    try:
        with gzip.open(filepath, "rb") as gz:
            while True:
                chunk = gz.read(1024 * 1024)
                if not chunk:
                    break
        return "valid"
    except (gzip.BadGzipFile, EOFError, OSError):
        return "corrupt"


def _has_phase_annotations(filepath):
    """Peek into a trace file and check for vLLM scheduling annotation markers.

    The phase-split script (split_vllm_trace_annotation.py) requires annotation
    events to determine prefill/decode boundaries.  Without them the split
    produces empty or meaningless output.
    """
    try:
        opener = gzip.open if filepath.endswith(".gz") else open
        with opener(filepath, "rt") as fh:
            content = fh.read(ANNOTATION_PEEK_BYTES)
        return any(marker in content for marker in ANNOTATION_MARKERS)
    except Exception:
        return False


def discover_traces(trace_dir):
    valid_traces = []
    trace_details = []
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
            integrity = "valid"
            if f.endswith(".gz"):
                integrity = _check_gzip_integrity(f)
                if integrity == "corrupt":
                    print(f"CORRUPT (gzip integrity failed): {f}")
                    trace_details.append({
                        "path": f,
                        "size_bytes": os.path.getsize(f),
                        "integrity": "corrupt",
                        "role": "unknown",
                    })
                    continue

            opener = gzip.open if f.endswith(".gz") else open
            with opener(f, "rt") as fh:
                prefix = fh.read(PEEK_BYTES)
            if '"traceEvents"' not in prefix:
                print(f"SKIPPED (no traceEvents key in first {PEEK_BYTES} bytes): {f}")
                trace_details.append({
                    "path": f,
                    "size_bytes": os.path.getsize(f),
                    "integrity": "missing",
                    "role": "unknown",
                })
                continue

            trace_role = "secondary"
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
                trace_role = "secondary"
                if not merged_trace:
                    merged_trace = f
            elif tp_phase_match:
                trace_role = "secondary"
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
                trace_role = "secondary"
                rank = int(rank_match.group(1))
                rank_ids.add(rank)
                full_rank_ranks.add(rank)
                if rank == 0 and not rank0_full_trace:
                    rank0_full_trace = f

            valid_traces.append(f)
            size_bytes = os.path.getsize(f)
            size_mb = size_bytes / (1024 * 1024)
            descriptor = [trace_role]
            if rank is not None:
                descriptor.append(f"rank {rank}")
            if phase:
                descriptor.append(phase)
            print(
                f"VALID torch trace ({', '.join(descriptor)}, {size_mb:.1f} MB compressed): {f}"
            )
            trace_details.append({
                "path": f,
                "size_bytes": size_bytes,
                "integrity": integrity,
                "role": trace_role,
            })
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

    # Mark the primary trace in trace_details
    for td in trace_details:
        if td["path"] == tracelens_primary_trace and td["integrity"] == "valid":
            td["role"] = "primary"

    # Determine phase_split_inputs_ready — requires both a valid trace
    # *and* the presence of scheduling annotation events that the split
    # script (split_vllm_trace_annotation.py) relies on.
    phase_split_annotations_detected = False
    if phase_split_input_trace:
        trace_valid = any(
            td["path"] == phase_split_input_trace and td["integrity"] == "valid"
            for td in trace_details
        )
        if trace_valid:
            phase_split_annotations_detected = _has_phase_annotations(
                phase_split_input_trace
            )
            if not phase_split_annotations_detected:
                print(
                    f"NOTE: phase-split input trace has no annotation markers "
                    f"in first {ANNOTATION_PEEK_BYTES // 1024} KB — "
                    f"split_vllm_trace_annotation.py may produce empty output"
                )
    phase_split_inputs_ready = bool(
        phase_split_input_trace and phase_split_annotations_detected
    )

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

    # Build structured trace manifest for monitor consumption.
    # `trace_count` remains the count of usable traces for downstream analysis,
    # while `traces` records every examined file so monitors/tests can inspect
    # corrupt or incomplete inputs without re-scanning the directory.
    trace_manifest = {
        "trace_count": len(valid_traces),
        "world_size": world_size,
        "traces": trace_details,
        "tracelens_primary_trace": tracelens_primary_trace,
        "phase_split_inputs_ready": phase_split_inputs_ready,
        "phase_split_annotations_detected": phase_split_annotations_detected,
        "validation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
        print(f"PHASE_SPLIT_INPUTS_READY={phase_split_inputs_ready}")
    else:
        print(
            "WARNING: No valid torch profiler traces found. Trace analysis will be skipped."
        )

    return result, trace_manifest


def main():
    parser = argparse.ArgumentParser(description="Discover and validate trace files")
    parser.add_argument("--trace-dir", required=True, help="Directory containing trace files")
    parser.add_argument("--output", help="Optional legacy JSON manifest output path")
    parser.add_argument("--results-dir", help="Directory for trace_manifest.json (defaults to sibling results/ of trace-dir)")
    args = parser.parse_args()

    result, trace_manifest = discover_traces(args.trace_dir)

    # Always write trace_manifest.json for monitor consumption
    if args.results_dir:
        manifest_dir = args.results_dir
    else:
        manifest_dir = os.path.join(os.path.dirname(args.trace_dir.rstrip("/")), "results")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "trace_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(trace_manifest, f, indent=2)
    print(f"TRACE_MANIFEST={manifest_path}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"MANIFEST_JSON={args.output}")


if __name__ == "__main__":
    main()
