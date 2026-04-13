# Phase 6: Problem Generation

## Instructions

You are a phase agent responsible for generating optimization problem files from profiling data. You read exactly 2 files: this document and your handoff at `handoff/to-phase-06.md`.

**Tools**: Shell commands, Python, file I/O.
**Outputs**: Write `agent-results/phase-06-result.md`. Write problem files and manifest to `{PROBLEMS_DIR}`.
**Errors**: If `gap_analysis.json` is missing, report failure immediately -- this is a hard prerequisite.

## Runbook

### 1. Verify Prerequisites
```bash
[ -f "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" ] || { echo "ERROR: gap_analysis.json missing"; exit 1; }
```
Find GEMM CSV in tracelens output dirs.

### 1.5. Classify Kernel Types
Treat `{{OUTPUT_DIR}}/results/profile_analysis.json` as the **primary** input. When estimating optimization impact, **exclude communication time** from the denominator so comm-heavy runs do not drown out compute targets. Use `classify_kernel()` from `classify_kernel.py` on each relevant op name; **skip** any op whose classified type is `communication` (those are not kernel-optimization problems in this pipeline).

```python
import json, os, sys
sys.path.insert(0, "{{SCRIPTS_DIR}}/optimize")
from classify_kernel import classify_kernel
# Read profile_analysis.json, classify each kernel, exclude communication,
# write kernel_type_classification.json to PROBLEMS_DIR
```

Runnable reference (writes `{{PROBLEMS_DIR}}/kernel_type_classification.json` with `comm_pct_excluded` and `classifications` sorted by `pct_optimizable` descending):

```bash
python3 << 'CLASSIFY'
import json, os, sys
sys.path.insert(0, "{{SCRIPTS_DIR}}/optimize")
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
    if kernel_type == "communication":
        continue
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
```bash
python3 "{{SCRIPTS_DIR}}/optimize/extract_model_shapes.py" \
    --output "{{PROBLEMS_DIR}}/model_shapes.json" \
    --gemm-csv "$GEMM_CSV" --config-json "{{OUTPUT_DIR}}/config.json"
```

### 4. Run Fusion Analysis
```bash
python3 "{{SCRIPTS_DIR}}/optimize/analyze_fusion_inferencex.py" \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --tracelens-dir "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
    --decode-tracelens-dir "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" \
    --framework "{{FRAMEWORK}}" --threshold 1.0 \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json" \
    --model-precision "{{PRECISION}}" --output-dir "{{PROBLEMS_DIR}}"
```

### 5. Generate Problem Files
```bash
python3 "{{SCRIPTS_DIR}}/optimize/generate_problems_inferencex.py" \
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

**Conditional flags:** The trailing `$([ -f ... ] && echo ...)` clauses are intentional. When `kernel_type_classification.json` exists (step 1.5), pass `--kernel-types` so generation respects per-kernel typing and prioritization. When `roofline_bottlenecks.json` exists, pass `--roofline-bottlenecks` and `--roofline-threshold 80.0` so only sub-roofline targets (below 80% efficiency) are emphasized. Omit flags automatically when files are missing.

### 5b-5e. Agent-Generated Problem Files
For kernel types that need human/agent-authored scaffolding beyond the bulk generator, add problem files that satisfy the GEAK `Model` / `ModelNew` contract:

- **HIP / CK (`hip`, `ck`):** `Model` should invoke the real HIP kernel via Python bindings where possible so the baseline reflects production code, not a synthetic stub.
- **Triton composite (`triton_composite`):** `Model` calls the composite entry point directly and records paths to each sub-kernel source file GEAK must reason about.
- **Vendor kernels:** If vendor source is accessible in the workspace, point problem metadata at the true implementation. If source is inaccessible, fall back to a `torch.mm` (or equivalent) baseline using profiled shapes so optimization still has a defined target.
- **Roofline-gated targets:** Apply per-type strategy (MoE GEMM, FP4 GEMM, attention, normalization, activation, etc.). Always skip pure communication and `moe_sort`-class ops. Skip generation when roofline efficiency is already ≥ 80% (headroom too small).

### 6. Review
List generated `problem_*.py` files and `optimization_manifest.json`.

### Completion
Write `agent-results/phase-06-result.md` with problems_generated count, fusion_opportunities count, manifest_entries count.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
