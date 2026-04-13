#!/usr/bin/env bash
# Display TraceLens analysis results summary to console.
# Usage: bash display_tracelens_results.sh <OUTPUT_DIR> [PRIMARY_ROLE]
set -euo pipefail

OUTPUT_DIR="$1"
PRIMARY_ROLE="${2:-unknown}"
RESULTS="$OUTPUT_DIR/results"

echo ""
echo "============================================"
echo "  TraceLens Analysis Summary (Primary Single Trace)"
echo "============================================"
echo "  Source role: $PRIMARY_ROLE"

for csv_name in gpu_timeline.csv ops_summary_by_category.csv; do
    csv_path="$RESULTS/tracelens_rank0_csvs/$csv_name"
    if [ -f "$csv_path" ]; then
        label=$(echo "$csv_name" | sed 's/.csv//' | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')
        echo ""
        echo "--- $label ---"
        cat "$csv_path"
    fi
done

for csv_name in ops_summary.csv kernel_summary.csv GEMM.csv; do
    csv_path="$RESULTS/tracelens_rank0_csvs/$csv_name"
    if [ -f "$csv_path" ]; then
        label=$(echo "$csv_name" | sed 's/.csv//' | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')
        echo ""
        echo "--- $label (first 25 lines) ---"
        head -25 "$csv_path"
    fi
done

echo ""
echo "============================================"
echo "  Phase-Split Roofline Analysis"
echo "============================================"

for PHASE_LABEL in "Prefill-Decode" "Decode-Only"; do
    if [ "$PHASE_LABEL" = "Prefill-Decode" ]; then
        PHASE_DIR="$RESULTS/tracelens_prefill_decode_csvs"
    else
        PHASE_DIR="$RESULTS/tracelens_decode_only_csvs"
    fi

    if [ -d "$PHASE_DIR" ] && ls "$PHASE_DIR"/*.csv &>/dev/null; then
        echo ""
        echo "--- $PHASE_LABEL Phase ---"

        for csv in gpu_timeline.csv ops_summary_by_category.csv; do
            [ -f "$PHASE_DIR/$csv" ] && { echo ""; echo "  $(echo $csv | sed 's/.csv//;s/_/ /g') ($PHASE_LABEL):"; cat "$PHASE_DIR/$csv"; }
        done

        for csv in unified_perf_summary.csv GEMM.csv; do
            [ -f "$PHASE_DIR/$csv" ] && { echo ""; echo "  $(echo $csv | sed 's/.csv//;s/_/ /g') ($PHASE_LABEL, first 25 lines):"; head -25 "$PHASE_DIR/$csv"; }
        done

        for ATTN_CSV in "$PHASE_DIR/SDPA_fwd.csv" "$PHASE_DIR/FLASH_ATTN_fwd.csv"; do
            if [ -f "$ATTN_CSV" ]; then
                echo ""
                echo "  Attention Roofline ($PHASE_LABEL, first 25 lines):"
                head -25 "$ATTN_CSV"
                break
            fi
        done
    else
        echo ""
        echo "  $PHASE_LABEL phase: no roofline data available"
    fi
done

echo ""
echo "============================================"
