#!/usr/bin/env bash
# Run TraceLens analysis: primary single-trace, multi-rank collective, and phase-split roofline.
#
# Usage: bash run_tracelens.sh \
#   --primary-trace <path> --primary-role <role> \
#   --output-dir <dir> --profile-dir <dir> \
#   [--world-size N] [--collective-mode <mode>] \
#   [--phase-split-trace <path>] [--phase-split-role <role>] \
#   [--gpu-arch-json <path>]
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

PRIMARY_TRACE=""
PRIMARY_ROLE=""
OUTPUT_DIR=""
PROFILE_DIR=""
WORLD_SIZE=1
COLLECTIVE_MODE="unavailable"
PHASE_SPLIT_TRACE=""
PHASE_SPLIT_ROLE=""
GPU_ARCH_JSON=""
TRACELENS_DIR="${TRACELENS_DIR:-$HOME/TraceLens-internal}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --primary-trace)   PRIMARY_TRACE="$2"; shift 2 ;;
        --primary-role)    PRIMARY_ROLE="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --profile-dir)     PROFILE_DIR="$2"; shift 2 ;;
        --world-size)      WORLD_SIZE="$2"; shift 2 ;;
        --collective-mode) COLLECTIVE_MODE="$2"; shift 2 ;;
        --phase-split-trace) PHASE_SPLIT_TRACE="$2"; shift 2 ;;
        --phase-split-role)  PHASE_SPLIT_ROLE="$2"; shift 2 ;;
        --gpu-arch-json)   GPU_ARCH_JSON="$2"; shift 2 ;;
        --tracelens-dir)   TRACELENS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

RESULTS_DIR="$OUTPUT_DIR/results"

# --- Primary single-trace report ---
if [ -n "$PRIMARY_TRACE" ] && [ -f "$PRIMARY_TRACE" ]; then
    mkdir -p "$RESULTS_DIR/tracelens_rank0_csvs"
    echo "TraceLens primary trace role: $PRIMARY_ROLE"
    TraceLens_generate_perf_report_pytorch \
        --profile_json_path "$PRIMARY_TRACE" \
        --output_csvs_dir "$RESULTS_DIR/tracelens_rank0_csvs" \
        --enable_kernel_summary \
        2>&1 | tee "$RESULTS_DIR/tracelens_rank0.log"
    echo "Primary single-trace report exit code: $?"
    ls -lh "$RESULTS_DIR/tracelens_rank0_csvs/"
else
    echo "No suitable single-trace input found — skipping primary TraceLens report"
fi

# --- Multi-rank collective report ---
if [ "$WORLD_SIZE" -gt 1 ] && [ "$COLLECTIVE_MODE" != "unavailable" ]; then
    COLLECTIVE_TRACE_DIR="$RESULTS_DIR/collective_trace_inputs"
    rm -rf "$COLLECTIVE_TRACE_DIR"
    mkdir -p "$COLLECTIVE_TRACE_DIR"

    COLLECTIVE_TRACE_COUNT=$(python3 -c "
import glob, os, re, sys
profile_dir = sys.argv[1]
stage_dir = sys.argv[2]
mode = sys.argv[3]
selected = {}
for f in sorted(glob.glob(os.path.join(profile_dir, '*.json*'))):
    name = os.path.basename(f)
    if mode == 'tp-extend':
        match = re.search(r'(?:^|[-_])tp[-_]?(\d+)[-_]extend(?:[-_.]|$)', name, re.I)
        if match:
            selected.setdefault(int(match.group(1)), f)
    elif mode == 'rank-full':
        if re.search(r'(^|[-_])merged(?:[-_.]|$)', name.lower()):
            continue
        if re.search(r'(?:^|[-_])tp[-_]?(\d+)[-_](decode|extend)(?:[-_.]|$)', name, re.I):
            continue
        match = re.search(r'(?:^|[-_])rank[-_]?(\d+)(?:[-_.]|$)', name, re.I)
        if match:
            selected.setdefault(int(match.group(1)), f)
for entry in os.listdir(stage_dir):
    path = os.path.join(stage_dir, entry)
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
for rank, src in sorted(selected.items()):
    os.symlink(src, os.path.join(stage_dir, f'rank{rank}_trace.json.gz'))
print(len(selected))
" "$PROFILE_DIR" "$COLLECTIVE_TRACE_DIR" "$COLLECTIVE_MODE")

    echo "Collective trace mode: $COLLECTIVE_MODE"
    echo "Collective staged trace count: $COLLECTIVE_TRACE_COUNT"
    ls -lh "$COLLECTIVE_TRACE_DIR"

    if [ "$COLLECTIVE_TRACE_COUNT" -eq "$WORLD_SIZE" ]; then
        mkdir -p "$RESULTS_DIR/tracelens_collective_csvs"
        TraceLens_generate_multi_rank_collective_report_pytorch \
            --trace_pattern "$COLLECTIVE_TRACE_DIR/rank*_trace.json.gz" \
            --world_size "$WORLD_SIZE" \
            --output_csvs_dir "$RESULTS_DIR/tracelens_collective_csvs" \
            2>&1 | tee "$RESULTS_DIR/tracelens_collective.log"
        echo "Multi-rank collective report exit code: ${PIPESTATUS[0]}"
        ls -lh "$RESULTS_DIR/tracelens_collective_csvs/"
    else
        echo "Collective trace staging mismatch (expected $WORLD_SIZE traces) — skipping"
    fi
else
    echo "No clean multi-rank trace set available — skipping multi-rank collective report"
fi

# --- Phase-split roofline analysis ---
SPLIT_SCRIPT="$TRACELENS_DIR/examples/custom_workflows/split_vllm_trace_annotation.py"
INFERENCE_REPORT_SCRIPT="$TRACELENS_DIR/TraceLens/Reporting/generate_perf_report_pytorch_inference.py"
PHASE_SPLIT_DIR="$RESULTS_DIR/phase_split"

if [ -n "$PHASE_SPLIT_TRACE" ] && [ -f "$PHASE_SPLIT_TRACE" ] && [ -f "$SPLIT_SCRIPT" ]; then
    mkdir -p "$PHASE_SPLIT_DIR"
    echo "Splitting trace into prefill-decode and decode-only phases..."
    echo "Phase-split input role: $PHASE_SPLIT_ROLE"
    python3 "$SPLIT_SCRIPT" "$PHASE_SPLIT_TRACE" \
        -o "$PHASE_SPLIT_DIR" \
        --find-steady-state \
        --num-steps 32 \
        2>&1 | tail -30
    echo "Phase split exit code: $?"
    ls -lh "$PHASE_SPLIT_DIR/"
else
    echo "PHASE_SPLIT_UNAVAILABLE=true"
    echo "Skipping phase-split roofline (input trace or split script unavailable)"
fi

# Identify phase-specific traces
PREFILL_DECODE_TRACE=""
DECODE_ONLY_TRACE=""
if [ -d "$PHASE_SPLIT_DIR" ] && ls "$PHASE_SPLIT_DIR"/*.json.gz &>/dev/null; then
    PREFILL_DECODE_TRACE=$(ls "$PHASE_SPLIT_DIR"/prefilldecode_*.json.gz 2>/dev/null | head -1)
    DECODE_ONLY_TRACE=$(ls "$PHASE_SPLIT_DIR"/decode_*.json.gz 2>/dev/null | head -1)
    echo "PREFILL_DECODE_TRACE=$PREFILL_DECODE_TRACE"
    echo "DECODE_ONLY_TRACE=$DECODE_ONLY_TRACE"
fi

ROOFLINE_FLAGS=""
if [ -n "$GPU_ARCH_JSON" ] && [ -f "$GPU_ARCH_JSON" ]; then
    ROOFLINE_FLAGS="--gpu_arch_json_path $GPU_ARCH_JSON"
fi

# Run prefill-decode roofline
if [ -n "$PREFILL_DECODE_TRACE" ] && [ -f "$PREFILL_DECODE_TRACE" ] && [ -f "$INFERENCE_REPORT_SCRIPT" ]; then
    mkdir -p "$RESULTS_DIR/tracelens_prefill_decode_csvs"
    echo "Running TraceLens roofline on prefill-decode phase..."
    python3 "$INFERENCE_REPORT_SCRIPT" \
        --profile_json_path "$PREFILL_DECODE_TRACE" \
        --output_csvs_dir "$RESULTS_DIR/tracelens_prefill_decode_csvs" \
        --enable_pseudo_ops --group_by_parent_module --enable_kernel_summary \
        $ROOFLINE_FLAGS \
        2>&1 | tail -30
    echo "Prefill-decode roofline exit code: $?"
    ls -lh "$RESULTS_DIR/tracelens_prefill_decode_csvs/"
else
    echo "Skipping prefill-decode roofline (trace or script unavailable)"
fi

# Run decode-only roofline
if [ -n "$DECODE_ONLY_TRACE" ] && [ -f "$DECODE_ONLY_TRACE" ] && [ -f "$INFERENCE_REPORT_SCRIPT" ]; then
    mkdir -p "$RESULTS_DIR/tracelens_decode_only_csvs"
    echo "Running TraceLens roofline on decode-only phase..."
    python3 "$INFERENCE_REPORT_SCRIPT" \
        --profile_json_path "$DECODE_ONLY_TRACE" \
        --output_csvs_dir "$RESULTS_DIR/tracelens_decode_only_csvs" \
        --enable_pseudo_ops --group_by_parent_module --enable_kernel_summary \
        $ROOFLINE_FLAGS \
        2>&1 | tail -30
    echo "Decode-only roofline exit code: $?"
    ls -lh "$RESULTS_DIR/tracelens_decode_only_csvs/"
else
    echo "Skipping decode-only roofline (trace or script unavailable)"
fi
