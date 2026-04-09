# Phase 6: Problem Generation {{SKIP_LABEL}}

## Objective
Analyze profiling data to identify optimization targets, detect fusion opportunities, and generate problem files with PyTorch baselines for kernel optimization.

## Prerequisites
- Phase 5 must have completed successfully
- `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` MUST exist
- `{{OUTPUT_DIR}}/results/profile_analysis.json` SHOULD exist
- TraceLens CSVs in `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` are optional but improve GEMM shape extraction

## Steps

### 1. Verify Phase 5 Artifacts
```bash
[ -f "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" ] || { echo "ERROR: gap_analysis.json missing"; exit 1; }
GEMM_CSV=""
for dir in "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs" "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"; do
    [ -f "$dir/GEMM.csv" ] && { GEMM_CSV="$dir/GEMM.csv"; break; }
done
```

### 1.5. Classify Kernel Types

Read `profile_analysis.json` as PRIMARY input. Exclude communication time when computing optimization impact. Use `classify_kernel.py` for kernel type detection:

```bash
python3 << 'CLASSIFY'
import json, os, sys
sys.path.insert(0, "{{SCRIPTS_DIR}}")
from classify_kernel import classify_kernel

profile_path = "{{OUTPUT_DIR}}/results/profile_analysis.json"
if not os.path.isfile(profile_path):
    print("WARNING: profile_analysis.json not found"); exit(0)

profile = json.load(open(profile_path))
comm_pct = profile.get("category_breakdown", {}).get("communication", 0)
non_comm_total = 100.0 - comm_pct
classifications = []
for op in profile.get("top_ops", []):
    kernel_type, _ = classify_kernel(op.get("name", ""))
    if kernel_type == "communication": continue
    pct_opt = (op.get("pct", 0) / non_comm_total * 100.0) if non_comm_total > 0 else 0
    classifications.append({"name": op["name"], "kernel_type": kernel_type, "pct_optimizable": round(pct_opt, 2)})

os.makedirs("{{PROBLEMS_DIR}}", exist_ok=True)
with open("{{PROBLEMS_DIR}}/kernel_type_classification.json", "w") as f:
    json.dump({"comm_pct_excluded": round(comm_pct, 2), "classifications": sorted(classifications, key=lambda x: -x["pct_optimizable"])}, f, indent=2)
CLASSIFY
```

### 2. Create Directories
```bash
mkdir -p "{{PROBLEMS_DIR}}" "{{OPTIMIZED_DIR}}"
```

### 3. Extract Model Shapes

Run: `python3 "{{SCRIPTS_DIR}}/extract_model_shapes.py" --output "{{PROBLEMS_DIR}}/model_shapes.json" --gemm-csv "$GEMM_CSV" --config-json "{{OUTPUT_DIR}}/config.json"`

### 4. Run Fusion Analysis

```bash
python3 "{{SCRIPTS_DIR}}/analyze_fusion_inferencex.py" \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --tracelens-dir "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
    --decode-tracelens-dir "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" \
    --framework "{{FRAMEWORK}}" --threshold 1.0 \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json" \
    --model-precision "{{PRECISION}}" --output-dir "{{PROBLEMS_DIR}}"
```

### 5. Generate Problem Files

```bash
python3 "{{SCRIPTS_DIR}}/generate_problems_inferencex.py" \
    --fusion-opportunities "{{PROBLEMS_DIR}}/fusion_opportunities.json" \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --bottleneck-kernels "{{PROBLEMS_DIR}}/bottleneck_kernels.json" \
    --gemm-csv "$GEMM_CSV" \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json" \
    --model-shapes "{{PROBLEMS_DIR}}/model_shapes.json" \
    --framework "{{FRAMEWORK}}" \
    --priority-threshold {{OPTIMIZE_PRIORITY_THRESHOLD}} \
    --output-dir "{{PROBLEMS_DIR}}" \
    $([ -f "{{PROBLEMS_DIR}}/kernel_type_classification.json" ] && echo "--kernel-types {{PROBLEMS_DIR}}/kernel_type_classification.json") \
    $([ -f "{{PROBLEMS_DIR}}/roofline_bottlenecks.json" ] && echo "--roofline-bottlenecks {{PROBLEMS_DIR}}/roofline_bottlenecks.json --roofline-threshold 80.0")
```

### 5b-5e. Agent-Generated Problem Files

For specific kernel types, the agent creates additional problem files:

- **HIP/CK kernels** (`hip`, `ck`): `Model` uses actual HIP kernel via Python binding. Baseline is the real kernel.
- **Triton composite** (`triton_composite`): `Model` calls composite function directly with sub-kernel source paths.
- **Vendor kernels**: Source accessible → point to actual source. Source inaccessible → fallback to `torch.mm` with profiled shapes.
- **Roofline-gated targets**: Per-type strategy (MoE GEMM, FP4 GEMM, Attention, Normalization, Activation). Skip communication/moe_sort always. Skip if roofline >= 80%.

### 6. Review Generated Files
```bash
echo "Generated problem files:"
ls -la {{PROBLEMS_DIR}}/problem_*.py 2>/dev/null || echo "  (none)"
echo "Optimization manifest:"
cat {{PROBLEMS_DIR}}/optimization_manifest.json 2>/dev/null || echo "  (not generated)"
```

## Completion
Update progress.json:
```json
{
  "phase": "problem-generate",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze", "problem-generate"],
  "current_step": "problem generation complete",
  "details": {
    "problems_generated": "<count>",
    "fusion_opportunities": "<count>",
    "manifest_entries": "<count>"
  }
}
```
