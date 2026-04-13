> **ARCHIVE**: This file is a reference copy of the original phase runbook. The active
> agent docs are in `agents/phase-NN-*.md`. Script paths in this file reference the
> pre-reorganization flat layout (`scripts/*.py`); the actual scripts are now under
> `scripts/{env,container,profiling,optimize,plugin,report}/`.

# Phase 7: Kernel Optimization {{SKIP_LABEL}}

## Objective
Optimize each bottleneck kernel using GEAK (when available) or manual Triton/ATen kernel writing. Verify accuracy and speedup for each.

## Prerequisites
- Problem files from Phase 6 in `{{PROBLEMS_DIR}}/`
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (enriched with kernel types, profiling data)
- `{{ENV_INFO_FILE}}` (from Phase 0)
- Docker image with PyTorch + Triton (same IMAGE as Phase 02/04)

## Config Resolution
Read `{{OUTPUT_DIR}}/results/sweep_configs.json` and set: `RUNNER`, `IMAGE`, `FRAMEWORK`, `MODEL`, `PRECISION`, `TP`, `EP`.

## Steps

### 0. Resolve Effective GEAK Mode

Run: `python3 "{{SCRIPTS_DIR}}/resolve_geak_mode.py" --user-mode "{{GEAK_MODE}}" --env-info "{{ENV_INFO_FILE}}"`

Capture `EFFECTIVE_GEAK_MODE` from output. Modes:
- `full`: process both `simple` and `kernel-url` mode kernels
- `triton_only`: process only `simple` mode kernels (skip `kernel-url`)
- `manual`: skip GEAK, use manual optimization fallback

### 1. Load Manifest

Run: `python3 "{{SCRIPTS_DIR}}/load_optimization_manifest.py" --manifest "{{PROBLEMS_DIR}}/optimization_manifest.json" --geak-mode "$EFFECTIVE_GEAK_MODE" --optimize-scope "{{OPTIMIZE_SCOPE}}"`

Displays prioritized kernel list sorted by `priority_score` descending, grouped by GEAK mode.

### 2. Start Persistent Optimization Container

```bash
bash "{{SCRIPTS_DIR}}/start_profile_container.sh" \
    --name "inferencex-kernel-opt-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{PROBLEMS_DIR}}" \
    --mode optimize \
    --mount "{{PROBLEMS_DIR}}:/workspace/problems" \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --mount "{{SCRIPTS_DIR}}:/workspace/scripts" \
    --mount "{{GEAK_DIR}}:/workspace/geak" \
    --env "AMD_LLM_API_KEY=${AMD_LLM_API_KEY:-}" \
    --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}"
```

### 3. Detect Environment Inside Container

Verify PyTorch, Triton, GPU arch, and GEAK availability inside the container.

Use `select_gpus.py` for GPU selection: `CUDA_VISIBLE_DEVICES` if GPUs specified, else all available.

### 3a. C++/Vendor Kernel Optimization (`kernel-url` mode)

For entries with `geak_mode: kernel-url`. Process in `profiling_pct` descending order.

**Source tracing required** — before GEAK, trace and verify actual GPU kernel source:
- Vendor GEMM (`Cijk_*`): Tensile-generated, dispatched by hipBLASLt
- CK/aiter kernels: `aiter/csrc/` package
- Verify source matches profiled kernel by name substring

**Operator-specific guidance:**

| Type | Source Location | Compute Spec |
|------|----------------|-------------|
| MoE GEMM | `aiter/csrc/ck_gemm_moe_2stages_codegen/` | varies by precision |
| FP4 GEMM | `aiter/aiter/ops/triton/gemm_afp4wfp4.py` | `matrix_fp4` |
| Attention | `aiter/csrc/mla/` | `matrix_bf16` |
| Normalization | `aiter/csrc/` | memory-bound |

For each kernel:
1. Prepare isolated workspace: copy source, `git init && git add -A && git commit -m init`
2. Build task description with roofline data: `compute_spec`, `tflops_s`, `roofline_efficiency`, `peak_tflops`
3. Launch GEAK: `mini -m claude-opus-4.6 --config mini_kernel.yaml --repo /workspace/${NAME}_opt --gpu-ids 0,1 --yolo -t '${TASK_DESC}' -o traj_${NAME}.json`
4. For vendor GEMM (no external source): copy ONLY the problem `.py` file to isolated workspace to avoid multi-MB patches
5. If speedup > 1.0x: install via `AITER_REBUILD=1` or `pip install -e .`

### 3b. Triton Optimization (`simple` mode)

**SCOPE**: Simple mode is ONLY for originally-Triton kernels (`kernel_type: triton`). All other kernel types must use `kernel-url` mode.

For each kernel:
1. Build task description with shapes and roofline data
2. Launch GEAK: `mini -m claude-opus-4.6 --config geak.yaml --gpu-ids 0 --yolo -t '${TASK_DESC}' -o traj_${NAME}.json`
3. Always `git init && git add -A && git commit -m init` in problems dir before launch

### 3.5. Best Kernel Capture & Patch Recovery (CRITICAL)

Runs after EVERY GEAK attempt. GEAK's `[SelectPatch]` agent frequently fails — manual recovery is essential.

1. Check GEAK log for `[SelectPatch]` success
2. If failed: scan `optimization_logs/${NAME}_*/patch_*_test.txt` for best speedup
3. If best speedup > 1.0x: apply the corresponding `.diff` patch, re-verify with `kernel_test_runner.py`
4. Copy winning `_opt.py` to `{{OPTIMIZED_DIR}}/`

### 4. Iterate and Finalize

Up to 5 attempts per kernel. After each, run Step 3.5. Finalize with:
`docker exec $CONTAINER python3 /workspace/scripts/kernel_finalize.py --target /workspace/problems/${NAME}_opt.py`

### 5. Collect Winning Kernels

Run inside container:
`docker exec $CONTAINER python3 /workspace/scripts/collect_winning_kernels.py --problems-dir /workspace/problems --optimized-dir /workspace/optimized`

Produces `geak_results.json` and copies winning `_opt.py` files.

### 6. Clean Up Container
```bash
docker stop "$CONTAINER_NAME" 2>/dev/null; docker rm "$CONTAINER_NAME" 2>/dev/null
```

### 7. Print Summary
```bash
python3 -c "
import json, os
results_path = '{{PROBLEMS_DIR}}/geak_results.json'
if os.path.isfile(results_path):
    results = json.load(open(results_path))
    for r in sorted(results, key=lambda x: -x.get('profiling_pct', 0)):
        status = 'OK' if r['speedup'] > 1.0 else 'SKIP'
        print(f\"  {r['name']:40s} pct={r.get('profiling_pct',0):5.1f}% speedup={r['speedup']:.2f}x mode={r.get('geak_mode','?')} {status}\")
"
```

### Manual Fallback

When `GEAK_MODE=manual`:
- **HIP/CK kernels**: WARNING that in-place optimization requires GEAK; attempt Triton replacement
- **Vendor kernels**: manual compatibility fixes if source accessible; Triton replacement otherwise
- **Triton kernels**: AMD wave_size=64, BLOCK_SIZE 16-256, num_warps 4-8, `@triton.autotune` with 10-20 configs
- Test with `kernel_test_runner.py`, iterate up to 5 attempts, finalize

## Completion
Update progress.json:
```json
{
  "phase": "kernel-optimize",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze", "problem-generate", "kernel-optimize"],
  "current_step": "kernel optimization complete",
  "details": {
    "kernels_attempted": "<count>",
    "kernels_improved": "<count>",
    "geak_used": true,
    "geak_mode": "<mode>"
  }
}
```
