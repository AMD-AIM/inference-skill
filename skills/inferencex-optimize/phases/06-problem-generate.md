# Phase 6: Problem Generation {{SKIP_LABEL}}

## Objective
Analyze profiling data to identify optimization targets, detect fusion opportunities, and generate problem files with PyTorch baselines for kernel optimization.

## Prerequisites
- Phase 5 (profile-analyze) must have completed successfully
- `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` MUST exist
- `{{OUTPUT_DIR}}/results/profile_analysis.json` SHOULD exist (primary input for profile-driven optimization targeting)
- TraceLens CSVs in `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` are optional but improve GEMM shape extraction and roofline prioritization

## Steps

### 1. Verify Phase 5 Artifacts

```bash
echo "Checking Phase 5 artifacts..."
[ -f "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" ] && echo "gap_analysis.json: OK" || { echo "ERROR: gap_analysis.json missing — run profile-analyze first"; exit 1; }
[ -f "{{OUTPUT_DIR}}/results/profile_analysis.json" ] && echo "profile_analysis.json: OK" || echo "WARNING: No profile_analysis.json — will use gap analysis only"
[ -d "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" ] && echo "tracelens_rank0_csvs/: OK" || echo "WARNING: No TraceLens CSVs — will use gap analysis only"
[ -f "{{OUTPUT_DIR}}/results/gpu_arch.json" ] && echo "gpu_arch.json: OK" || echo "WARNING: No gpu_arch.json — roofline efficiency unavailable"

# Check for GEMM.csv in decode-only or primary TraceLens dirs
GEMM_CSV=""
for dir in "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs" "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"; do
    if [ -f "$dir/GEMM.csv" ]; then
        GEMM_CSV="$dir/GEMM.csv"
        echo "GEMM.csv found: $GEMM_CSV"
        break
    fi
done
[ -z "$GEMM_CSV" ] && echo "WARNING: No GEMM.csv — GEMM problems will be skipped"
```

### 1.5. Load Profile Analysis and Classify Kernel Types

Read `profile_analysis.json` as the PRIMARY input for optimization targeting. Extract framework-level op names (`top_ops`), GEMM roofline data (`gemm_shapes`), category breakdown, and bottleneck recommendations. Exclude communication time when computing optimization impact percentages.

```bash
python3 << 'CLASSIFY'
import json, os, sys
sys.path.insert(0, "{{SCRIPTS_DIR}}")
from classify_kernel import classify_kernel

profile_path = "{{OUTPUT_DIR}}/results/profile_analysis.json"
output_dir = "{{PROBLEMS_DIR}}"
os.makedirs(output_dir, exist_ok=True)

if not os.path.isfile(profile_path):
    print("WARNING: profile_analysis.json not found — skipping kernel type classification")
    exit(0)

profile = json.load(open(profile_path))

category_breakdown = profile.get("category_breakdown", {})
comm_pct = category_breakdown.get("communication", 0)
non_comm_total = 100.0 - comm_pct
print(f"Communication time: {comm_pct:.1f}% (excluded from optimization targeting)")
print(f"Non-communication baseline: {non_comm_total:.1f}%")

gemm_roofline = {}
for g in profile.get("gemm_shapes", []):
    key = f"{g['M']}x{g['N']}x{g['K']}"
    gemm_roofline[key] = g.get("pct_roofline", None)

bottleneck_recs = {}
for b in profile.get("bottlenecks", []):
    cat = b.get("category", "").lower()
    bottleneck_recs[cat] = b.get("recommendation", "")

classifications = []
for op in profile.get("top_ops", []):
    name = op.get("name", "")
    pct_raw = op.get("pct", 0)

    kernel_type, _reason = classify_kernel(name)
    if kernel_type == "communication":
        continue

    pct_optimizable = (pct_raw / non_comm_total * 100.0) if non_comm_total > 0 else 0

    rec = ""
    lower = name.lower()
    for cat_key, cat_rec in bottleneck_recs.items():
        if cat_key in lower or any(k in lower for k in cat_key.split()):
            rec = cat_rec
            break

    classifications.append({
        "name": name,
        "kernel_type": kernel_type,
        "pct_total_raw": round(pct_raw, 2),
        "pct_optimizable": round(pct_optimizable, 2),
        "roofline_efficiency": None,
        "bottleneck_recommendation": rec,
        "source_file": "",
        "python_binding": "",
    })

classifications.sort(key=lambda x: -x["pct_optimizable"])

result = {
    "comm_pct_excluded": round(comm_pct, 2),
    "non_comm_baseline_pct": round(non_comm_total, 2),
    "classifications": classifications,
}

out_path = os.path.join(output_dir, "kernel_type_classification.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nKernel type classifications ({len(classifications)} ops, communication excluded):")
for c in classifications:
    print(f"  [{c['pct_optimizable']:5.1f}%] {c['name']:50s} type={c['kernel_type']}")
print(f"Saved to {out_path}")
CLASSIFY
```

**Source tracing** (inside optimization container in Phase 7): For each classified kernel, trace the source file by extracting the namespace prefix (e.g., `aiter::` → search in `aiter` package). This step is deferred to Phase 7 when the inference container is available with the actual framework packages installed.

### 2. Create Directories

```bash
mkdir -p "{{PROBLEMS_DIR}}" "{{OPTIMIZED_DIR}}"
```

### 3. Extract Model Shapes

If TraceLens GEMM.csv is available, infer model dimensions from GEMM shapes. Otherwise, attempt to read from the model's HuggingFace config.

```bash
python3 -c "
import csv, json, os, sys, collections
csv.field_size_limit(sys.maxsize)

shapes = {}
gemm_csv = os.environ.get('GEMM_CSV', '')

if gemm_csv and os.path.isfile(gemm_csv):
    # Infer hidden_size and intermediate_size from most common GEMM K and N dims
    k_counts = collections.Counter()
    n_counts = collections.Counter()
    with open(gemm_csv) as f:
        for row in csv.DictReader(f):
            try:
                K = int(float(row.get('param: K', 0) or 0))
                N = int(float(row.get('param: N', 0) or 0))
                if K > 0: k_counts[K] += 1
                if N > 0: n_counts[N] += 1
            except (ValueError, TypeError):
                continue
    if k_counts:
        shapes['hidden_size'] = k_counts.most_common(1)[0][0]
    if n_counts:
        # intermediate_size is typically the largest frequent N dimension
        top_n = [n for n, _ in n_counts.most_common(5)]
        shapes['intermediate_size'] = max(top_n) if top_n else 11008
    print(f'Inferred from GEMM.csv: {shapes}')
else:
    # Fallback: try HuggingFace config if available
    try:
        config_path = '{{OUTPUT_DIR}}/config.json'
        if os.path.isfile(config_path):
            cfg = json.load(open(config_path))
            model_name = cfg.get('model', cfg.get('hf_model', ''))
            if model_name:
                from transformers import AutoConfig
                hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()
                tc = hf_cfg.get('text_config', hf_cfg)
                shapes = {
                    'hidden_size': tc.get('hidden_size', 4096),
                    'intermediate_size': tc.get('intermediate_size', 11008),
                    'num_attention_heads': tc.get('num_attention_heads', 32),
                    'num_key_value_heads': tc.get('num_key_value_heads', 8),
                    'head_dim': tc.get('head_dim', tc.get('hidden_size', 4096) // max(tc.get('num_attention_heads', 1), 1)),
                }
                print(f'Loaded from HuggingFace config: {shapes}')
    except Exception as e:
        print(f'Could not load model config: {e}')
        shapes = {'hidden_size': 4096, 'intermediate_size': 11008}
        print(f'Using defaults: {shapes}')

with open('{{PROBLEMS_DIR}}/model_shapes.json', 'w') as f:
    json.dump(shapes, f, indent=2)
print(f'Saved model_shapes.json')
"
```

### 4. Run Fusion Analysis

```bash
cp "{{SCRIPTS_DIR}}/classify_kernel.py" "{{PROBLEMS_DIR}}/" 2>/dev/null || true
cp "{{SCRIPTS_DIR}}/analyze_fusion_inferencex.py" "{{PROBLEMS_DIR}}/" 2>/dev/null || true

python3 "{{SCRIPTS_DIR}}/analyze_fusion_inferencex.py" \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --tracelens-dir "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
    --decode-tracelens-dir "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" \
    --framework "{{FRAMEWORK}}" \
    --threshold 1.0 \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json" \
    --model-precision "{{PRECISION}}" \
    --output-dir "{{PROBLEMS_DIR}}"
```

### 5. Generate Problem Files

```bash
KERNEL_TYPES_ARG=""
[ -f "{{PROBLEMS_DIR}}/kernel_type_classification.json" ] && KERNEL_TYPES_ARG="--kernel-types {{PROBLEMS_DIR}}/kernel_type_classification.json"

ROOFLINE_ARG=""
[ -f "{{PROBLEMS_DIR}}/roofline_bottlenecks.json" ] && ROOFLINE_ARG="--roofline-bottlenecks {{PROBLEMS_DIR}}/roofline_bottlenecks.json --roofline-threshold 80.0"

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
    $KERNEL_TYPES_ARG \
    $ROOFLINE_ARG
```

The `--kernel-types` argument enriches the manifest with GEAK mode, source file, profiling percentages (communication-excluded), and roofline efficiency data from Phase 5.

### 5b. HIP Kernel Problem Files (agent-generated)

For kernels classified as `hip` or `ck` in `kernel_type_classification.json`, the agent should create problem files where `Model` uses the actual HIP kernel via Python binding (e.g., `torch.ops._rocm_C.wvSplitK`). Must import the registering package. Baseline is the real kernel, not a PyTorch approximation.

### 5c. Triton Composite Problem Files (agent-generated)

For kernels classified as `triton_composite`, `Model` calls the composite function directly. Include comments listing sub-kernel source paths for GEAK reference.

### 5d. Vendor Kernel Problem Files (agent-generated)

- **Source accessible** (primary): problem file points to actual vendor kernel source for GEAK kernel-url optimization. Goal: fix compatibility problems, dispatch paths, launch configs, shape specializations.
- **Source inaccessible** (fallback only): problem file uses `torch.mm(a, b)` with profiled shapes for Triton replacement.
- Priority: HIGH (source), MEDIUM (Triton fallback).

### 5e. Roofline-Gated Optimization Targets

When `roofline_bottlenecks.json` exists (generated by Step 4 when `gpu_arch.json` is available), the manifest includes `type: "roofline_gated"` entries for operators below the roofline threshold or lacking a perf model. Per-type optimization strategy:

- **MoE GEMM** (`moe_gemm`): CK source tuning via `kernel-url` or `gemm_moe_tune.py`. Typically `matrix_fp4` or `matrix_fp8` depending on model precision.
- **FP4 GEMM** (`gemm_fp4`): Triton autotuning via `kernel-url`. Always `matrix_fp4`.
- **Dense GEMM** (`aten_gemm`, `vendor`): Roofline against `matrix_bf16`/`matrix_fp16` peak from `compute_spec`. Only generated when no GEMM.csv (per-shape data is more actionable).
- **Attention** (`attention`, `ck`): CK/aiter source tuning via `kernel-url`. Typically `matrix_bf16`.
- **Normalization** (`normalization`): Memory-bound — roofline against `vector_bf16` peak. Triton fusion or skip if vendor kernel already fast.
- **Activation** (`activation`): Memory-bound — roofline against `vector_bf16` peak. Triton fusion via `simple` mode.
- Skip `communication` and `moe_sort` always.
- Skip if roofline >= 80% (kernel already near-optimal for its precision type).

Each entry carries `compute_spec`, `spec_confidence`, `peak_tflops`, and a human-readable `performance_note`.

### 6. Review Generated Files

```bash
echo ""
echo "============================================"
echo "  Problem Generation Summary"
echo "============================================"
echo "Problems directory: {{PROBLEMS_DIR}}"
echo ""
echo "Generated problem files:"
ls -la {{PROBLEMS_DIR}}/problem_*.py 2>/dev/null || echo "  (none)"
echo ""
echo "Optimization manifest:"
cat {{PROBLEMS_DIR}}/optimization_manifest.json 2>/dev/null || echo "  (not generated)"
echo ""
echo "Fusion opportunities:"
cat {{PROBLEMS_DIR}}/fusion_opportunities.json 2>/dev/null || echo "  (none detected)"
echo "============================================"
```

## Completion

Update progress.json:
```json
{
  "phase": "problem-generate",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze", "problem-generate"],
  "current_step": "problem generation complete",
  "details": {
    "problems_generated": "<count of problem_*.py files>",
    "fusion_opportunities": "<count from fusion_opportunities.json>",
    "manifest_entries": "<count from optimization_manifest.json>"
  }
}
```
