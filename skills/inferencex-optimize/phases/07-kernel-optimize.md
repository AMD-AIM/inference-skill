# Phase 7: Kernel Optimization {{SKIP_LABEL}}

## Objective
Optimize each bottleneck kernel using GEAK (when available) or manual Triton/ATen kernel writing. Verify accuracy and speedup for each. Optimization order is driven by Phase 5 profiling data (communication-excluded percentages).

## Prerequisites
- Problem files from Phase 6 in `{{PROBLEMS_DIR}}/`
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (enriched with kernel types, profiling data)
- `{{ENV_INFO_FILE}}` (from Phase 0)
- Docker image with PyTorch + Triton (same IMAGE as Phase 02/04)

## Config Resolution

Before executing this phase, read `{{OUTPUT_DIR}}/results/sweep_configs.json` (or `{{OUTPUT_DIR}}/config.json`) and set these shell variables from the config entry used for profiling:

```bash
# Extract from the profiling config (typically the first/only filtered config)
RUNNER="<runner field from config>"      # e.g., mi355x, h100
IMAGE="<image field from config>"        # Docker image URL
FRAMEWORK="<framework field from config>" # vllm or sglang
MODEL="<model field from config>"
PRECISION="<precision field from config>"
TP="<tp field from config>"
EP="<ep field from config, default 1>"
```

These variables are used in docker commands and GEAK task descriptions throughout this phase.

## Steps

### 0. Verify GEAK + API Key and Resolve Effective GEAK Mode

The user's `{{GEAK_MODE}}` preference (from INTAKE) is combined with runtime detection:

```bash
echo "Checking GEAK availability..."
python3 -c "
import json, os

user_mode = '{{GEAK_MODE}}'  # auto, full, triton_only, manual
env_info_path = '{{ENV_INFO_FILE}}'

geak = False
api_key = False
if os.path.isfile(env_info_path):
    env = json.load(open(env_info_path))
    geak = env.get('geak_available', False)
    api_key = env.get('llm_api_key_set', False)

# Resolve effective mode
if user_mode == 'manual':
    effective = 'manual'
elif not geak:
    print('WARNING: GEAK not available — falling back to manual mode')
    effective = 'manual'
elif not api_key:
    print('WARNING: GEAK installed but no LLM API key — falling back to manual mode')
    print('Set AMD_LLM_API_KEY, LLM_GATEWAY_KEY, or ANTHROPIC_API_KEY to enable GEAK')
    effective = 'manual'
elif user_mode == 'triton_only':
    effective = 'triton_only'
elif user_mode == 'full':
    effective = 'full'
else:  # auto
    effective = 'full'  # auto with GEAK available = full

print(f'User GEAK_MODE: {user_mode}')
print(f'Effective GEAK_MODE: {effective}')
print(f'  GEAK available: {geak}, API key: {api_key}')
"
```

Use the effective mode to filter kernels in Step 1:
- `full`: process both `simple` and `kernel-url` mode kernels
- `triton_only`: process only `simple` mode kernels (skip `kernel-url`)
- `manual`: skip GEAK, use manual optimization fallback for all kernels

### 1. Load Manifest — Profile-Driven Ordering

```bash
echo "Loading optimization manifest..."
python3 -c "
import json

manifest = json.load(open('{{PROBLEMS_DIR}}/optimization_manifest.json'))

# Filter to top-N optimization targets (ranked by priority_score in Phase 6)
# priority_score = profiling_pct * (1 - roofline_efficiency/100) for non-fusion
# priority_score = profiling_pct for fusion (no roofline concept)
enabled = [o for o in manifest['optimizations'] if o.get('optimize', False)]

# Fallback: if no optimize flag set (old manifest), use enabled + pct filter
if not enabled:
    enabled = [o for o in manifest['optimizations'] if o.get('enabled')]
    enabled = [o for o in enabled if o.get('profiling_pct', 100) >= 1.0]

# Always skip communication
enabled = [o for o in enabled if o.get('geak_mode', 'simple') != 'skip']

# Apply OPTIMIZE_SCOPE filter
optimize_scope = '{{OPTIMIZE_SCOPE}}'  # 'all' or 'fused_only'
if optimize_scope == 'fused_only':
    enabled = [o for o in enabled if o.get('type') == 'fused']
    print(f'OPTIMIZE_SCOPE=fused_only: filtered to {len(enabled)} fused entries')

# Apply GEAK_MODE filter for triton_only mode
effective_geak_mode = '{{GEAK_MODE}}'  # resolved in Step 0
if effective_geak_mode == 'triton_only':
    enabled = [o for o in enabled if o.get('geak_mode') == 'simple']
    print(f'GEAK_MODE=triton_only: filtered to {len(enabled)} simple-mode entries')

# GEAK mode policy validation: simple mode is ONLY for triton kernels
for o in enabled:
    kt = o.get('kernel_type', 'unknown')
    if o.get('geak_mode') == 'simple' and kt != 'triton':
        print(f'WARNING: {o[\"name\"]} has geak_mode=simple but kernel_type={kt} — overriding to kernel-url')
        o['geak_mode'] = 'kernel-url'
        o['geak_config'] = 'mini_kernel.yaml'

# Sort by priority_score (descending) — already computed in Phase 6
# Formula: profiling_pct * (1 - roofline_efficiency/100), or profiling_pct for fusion
enabled.sort(key=lambda o: -o.get('priority_score', o.get('profiling_pct', 0)))

# Group by geak_mode
simple = [o for o in enabled if o.get('geak_mode') == 'simple']
kernel_url = [o for o in enabled if o.get('geak_mode') == 'kernel-url']

print(f'Total entries: {len(manifest[\"optimizations\"])}')
print(f'Enabled (impact >= 1.0%): {len(enabled)}')
print(f'  simple mode: {len(simple)}')
print(f'  kernel-url mode: {len(kernel_url)}')
print()
for o in enabled:
    pct = o.get('profiling_pct', 0)
    mode = o.get('geak_mode', '?')
    score = o.get('priority_score', 0)
    eff = o.get('roofline_efficiency')
    eff_str = f' eff={eff:.0f}%' if eff is not None else ''
    print(f'  [score={score:5.1f}] {o[\"name\"]:40s} pct={pct:5.1f}%  {o.get(\"kernel_type\", \"?\"):15s}  mode={mode}{eff_str}')
"
```

### 2. Start Persistent Optimization Container

Detect GPU vendor and start a container with the inference framework image.

```bash
if [[ "$RUNNER" == mi* ]]; then
    GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"
else
    GPU_FLAGS="--gpus all"
fi

CONTAINER_NAME="inferencex-kernel-opt-{{CONFIG_KEY}}"

# Clean up any stale container from a previous run
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --label inferencex-pipeline=true \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 64g \
    --ipc=host \
    --network=host \
    -v {{PROBLEMS_DIR}}:/workspace/problems \
    -v {{OPTIMIZED_DIR}}:/workspace/optimized \
    -v {{SCRIPTS_DIR}}:/workspace/scripts \
    -v {{HF_CACHE}}:/root/.cache/huggingface \
    -v {{GEAK_DIR}}:/workspace/geak \
    -e AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-}" \
    -e LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-}" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -w /workspace/problems \
    $IMAGE \
    -c "sleep infinity"

echo "Container started: $CONTAINER_NAME"
```

### 3. Detect Environment Inside Container

```bash
docker exec "$CONTAINER_NAME" bash -c "
    python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\")'
    python3 -c 'import triton; print(f\"Triton: {triton.__version__}\")' 2>/dev/null || echo 'Triton: not installed'
    GPU_ARCH=\$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 || echo 'unknown')
    echo \"GPU_ARCH=\$GPU_ARCH\"
"

# Verify GEAK inside container
GEAK_AVAILABLE="false"
if [ -d "{{GEAK_DIR}}" ]; then
    docker exec "$CONTAINER_NAME" bash -c "
        export PYTHONPATH=/workspace/geak/src:\$PYTHONPATH
        python3 -c 'from minisweagent.run.mini import app; print(\"GEAK: available in container\")'" 2>/dev/null && GEAK_AVAILABLE="true" || echo "GEAK: not available in container"
fi
echo "GEAK_AVAILABLE=$GEAK_AVAILABLE"
```

**GPU selection**: Use `select_gpus.py` to pick the GPU(s) with most free memory:
```bash
MANUAL_GPUS="{{GPUS}}"
if [ -n "$MANUAL_GPUS" ]; then
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=$MANUAL_GPUS"
else
    GPU_ENV=""
fi
```

### 3a. C++ / Vendor Kernel Optimization — `mini --config mini_kernel.yaml`

For entries with `geak_mode: kernel-url` (hip, ck, asm, triton_composite, vendor, moe_gemm, gemm_fp4, attention, moe, normalization, activation, aten_gemm, aten_elementwise). Process in `profiling_pct` descending order.

**IMPORTANT — Source tracing for vendor kernels:**

Before launching GEAK, you MUST trace and verify the actual GPU kernel source:

1. **Vendor GEMM (`Cijk_*`)**: These are Tensile-generated kernels dispatched by hipBLASLt. The manifest `shapes` field lists the actual `Cijk_*` kernel names and their CK tile parameters (MT, MI, WG, ISA, SK). Source tracing:
   ```bash
   # Find Tensile source inside container
   find / -path "*/Tensile/*" -name "*.yaml" 2>/dev/null | head -5
   find / -name "Cijk_*" -path "*/library/*" 2>/dev/null | head -5
   ```
2. **CK/aiter kernels** (`ck_moe_*`, `ck_fmha_*`, `add_rmsnorm`, `fused_moe_`, MLA): Source is in the aiter package:
   ```bash
   pip show aiter | grep Location  # → aiter/csrc/
   ```
3. **Verify the found source matches the profiled kernel** by checking kernel name substrings match.

**Operator-specific guidance:**

- **MoE GEMM** (`moe_gemm`): CK tile tuning. Extract shapes from `kernel_details`. Source: `aiter/csrc/ck_gemm_moe_2stages_codegen/`. FP4 MoE GEMM has different tile constraints than BF16.
- **FP4 GEMM** (`gemm_fp4`): Triton autotuning (`BLOCK_SIZE_M/N/K`, `num_warps`, `waves_per_eu`). Source: `aiter/aiter/ops/triton/gemm_afp4wfp4.py`. Peak is `matrix_fp4`.
- **Attention** (`ck`, `attention`): aiter MLA kernel tuning. Source: `aiter/csrc/mla/`. Peak is typically `matrix_bf16`.
- **Normalization** (`normalization`): Memory-bound ops — target bandwidth. Source: `aiter/csrc/` (e.g., `add_rmsnorm_quant_kernel`).
- **Activation** (`activation`): Memory-bound ops — target bandwidth. Source: `aiter/` or framework activation kernels.
- **Vendor GEMM / ATen GEMM** (`vendor`, `aten_gemm`): The manifest `shapes` field contains all GEMM shapes and their actual `Cijk_*` GPU kernel names. Task description should include these shapes and CK tile parameters for tuning.

For each kernel:

1. Prepare workspace: copy source + dependencies, init git repo
2. Build precision-aware task description:
   ```
   "Optimize ${SOURCE_FILE} on AMD ${GPU_ARCH}.
    This kernel accounts for ${PROFILING_PCT}% of optimizable GPU time.
    Precision: ${COMPUTE_SPEC} (${SPEC_CONFIDENCE}).
    ${if TFLOPS_S: '${TFLOPS_S} TFLOPS/s = ${ROOFLINE_PCT}% of ${PEAK_TFLOPS} ${COMPUTE_SPEC} peak.'}
    ${if !TFLOPS_S: 'No perf model — pct-gated only.'}
    ${if vector_*: 'Memory-bound op. Target bandwidth, not compute.'}
    ${BOTTLENECK_RECOMMENDATION}"
   ```

   Use `compute_spec`, `spec_confidence`, `peak_tflops`, `tflops_s`, `roofline_efficiency`, and `performance_note` from the manifest entry when available. Fall back to `${BOTTLENECK_RECOMMENDATION}` for entries without roofline data.
3. Launch GEAK:
```bash
docker exec $GPU_ENV "$CONTAINER_NAME" bash -c "
    cd /workspace/problems
    # Prepare workspace for kernel-url mode
    mkdir -p /workspace/${NAME}_opt
    cp -r ${SOURCE_DIR}/* /workspace/${NAME}_opt/ 2>/dev/null || true
    cd /workspace/${NAME}_opt
    git init 2>/dev/null && git add -A && git commit -m init 2>/dev/null

    export PYTHONPATH=/workspace/geak/src:\$PYTHONPATH
    mini -m claude-opus-4.6 --config mini_kernel.yaml \
        --repo /workspace/${NAME}_opt --gpu-ids 0,1 --yolo \
        -t '${TASK_DESC}' \
        -o traj_${NAME}.json \
        &> log_${NAME}.txt
"
```

4. **Vendor GEMM (no external source):** For vendor/aten GEMM problems where there is no external kernel source directory, copy ONLY the single problem `.py` file into the isolated workspace. Do NOT use `--repo /workspace/problems` — this includes all problem files, analysis scripts, and test files, producing multi-MB patches (38k+ lines) with unrelated diffs. Example:
```bash
mkdir -p /workspace/${NAME}_opt
cp /workspace/problems/${NAME}.py /workspace/${NAME}_opt/
cd /workspace/${NAME}_opt
git init && git add -A && git commit -m init
```
Use `--repo /workspace/${NAME}_opt` for GEAK. For vendor kernels specifically: task description focuses on fixing compatibility issues, incorrect dispatch, suboptimal configs.

5. If speedup > 1.0x: install via `AITER_REBUILD=1` or `pip install -e .`

### 3b. Triton Optimization — `mini -t` (simple mode)

**SCOPE**: Simple mode is ONLY for originally-Triton kernels (`kernel_type: triton`). If any non-triton kernel appears here, it has been incorrectly classified — skip it and report the misclassification. GEMM, attention, MoE, vendor, CK, HIP, normalization, activation, and elementwise kernels must ALL use kernel-url mode (Step 3a).

For entries with `geak_mode: simple` (originally-triton kernels only). Process in `profiling_pct` descending order.

For each kernel:

1. Build profile-aware task description including roofline data:
   ```
   "Optimize problem_dense_gemm.py for AMD ${GPU_ARCH}.
    Shape: 705x2112x7168 (BF16). Currently at 26.4% of 1686 TFLOPS peak.
    Recommendation: ${BOTTLENECK_RECOMMENDATION}"
   ```

2. Launch GEAK in simple mode:
```bash
docker exec $GPU_ENV "$CONTAINER_NAME" bash -c "
    cd /workspace/problems
    git init 2>/dev/null && git add -A && git commit -m init 2>/dev/null

    export PYTHONPATH=/workspace/geak/src:\$PYTHONPATH
    mini -m claude-opus-4.6 --config geak.yaml \
        --gpu-ids 0 --yolo \
        -t '${TASK_DESC}' \
        -o traj_${NAME}.json \
        &> log_${NAME}.txt
"
```

**Critical notes:**
- Short `-o` path (avoid `OSError: File name too long`)
- `git init && git add -A && git commit -m init` before launching mini
- Kernel-type-aware task descriptions for best results

### 3.5. Best Kernel Capture & Patch Recovery (CRITICAL)

**This step runs after EVERY GEAK optimization attempt (Steps 3a/3b) to guarantee the best kernel is never lost.** GEAK's `[SelectPatch]` agent frequently fails to apply the best patch — manual recovery is essential.

**Best kernel capture procedure:**

1. **Check GEAK success**: Look in the GEAK log for `[SelectPatch]` success message
2. **If GEAK succeeded**: Verify the applied patch actually has the best speedup by reading `*_opt_best.json`
3. **If GEAK failed OR best patch wasn't applied**: Execute patch recovery:

```bash
docker exec "$CONTAINER_NAME" bash -c "
    cd /workspace/problems
    NAME='${NAME}'
    OPT_FILE='${NAME}_opt.py'
    SRC_FILE='${NAME}.py'

    # Find latest optimization_logs directory for this kernel
    LOG_DIR=\$(ls -dt optimization_logs/\${NAME}_* 2>/dev/null | head -1)
    if [ -z \"\$LOG_DIR\" ]; then
        echo 'No optimization logs found for \$NAME'
        exit 0
    fi

    PATCH_RECOVERED='false'

    # Check if SelectPatch succeeded
    if grep -q '\\[SelectPatch\\].*success' log_\${NAME}.txt 2>/dev/null; then
        echo 'GEAK SelectPatch succeeded for \$NAME'
    else
        echo 'GEAK SelectPatch FAILED — scanning for best patch...'

        BEST_SPEEDUP=0
        BEST_PATCH=''
        for test_log in \${LOG_DIR}/patch_*_test.txt; do
            [ -f \"\$test_log\" ] || continue
            speedup=\$(grep -oP '\"speedup\":\\s*\\K[\\d.]+' \"\$test_log\" 2>/dev/null || echo 0)
            if [ -z \"\$speedup\" ]; then
                speedup=\$(grep -oP 'GEAK_RESULT_LATENCY_MS=\\K[\\d.]+' \"\$test_log\" 2>/dev/null || echo 0)
            fi
            if (( \$(echo \"\$speedup > \$BEST_SPEEDUP\" | bc -l 2>/dev/null || echo 0) )); then
                BEST_SPEEDUP=\$speedup
                BEST_PATCH=\"\${test_log%_test.txt}.diff\"
            fi
        done

        if [ -n \"\$BEST_PATCH\" ] && [ -f \"\$BEST_PATCH\" ] && (( \$(echo \"\$BEST_SPEEDUP > 1.0\" | bc -l 2>/dev/null || echo 0) )); then
            echo \"Recovering best patch: \$BEST_PATCH (speedup=\${BEST_SPEEDUP}x)\"
            PATCH_RECOVERED='true'
            git checkout -- \"\$OPT_FILE\" 2>/dev/null || true
            git apply --include=\"\$OPT_FILE\" \"\$BEST_PATCH\" 2>/dev/null || {
                echo 'WARNING: git apply failed, trying patch -p1'
                patch -p1 < \"\$BEST_PATCH\" 2>/dev/null || echo 'ERROR: Could not apply patch'
            }
            # Re-verify
            python3 /workspace/scripts/kernel_test_runner.py --src \"\$SRC_FILE\" --target \"\$OPT_FILE\"
        else
            echo \"No viable patch found (best speedup: \${BEST_SPEEDUP}x)\"
        fi
    fi

    # Always update tracker (record patch_recovered flag) and copy winning kernel
    if [ -f \"\${NAME}_opt_best.json\" ]; then
        python3 -c \"
import json
t = json.load(open('\${NAME}_opt_best.json'))
t['patch_recovered'] = \$PATCH_RECOVERED
json.dump(t, open('\${NAME}_opt_best.json', 'w'), indent=2)
\" 2>/dev/null || true
        speedup=\$(python3 -c \"import json; print(json.load(open('\${NAME}_opt_best.json')).get('best_speedup', 0))\" 2>/dev/null || echo 0)
        if (( \$(echo \"\$speedup > 1.0\" | bc -l 2>/dev/null || echo 0) )); then
            cp -f \"\$OPT_FILE\" /workspace/optimized/ 2>/dev/null && echo \"Copied \$OPT_FILE to optimized/ (speedup=\${speedup}x)\"
        fi
    fi
"
```

### 4. Iterate and Finalize

For each kernel, iterate up to 5 attempts. After each attempt, run Step 3.5 to capture best result. Finalize with `kernel_finalize.py`:

```bash
docker exec "$CONTAINER_NAME" \
    python3 /workspace/scripts/kernel_finalize.py \
    --target /workspace/problems/problem_XXX_opt.py
```

### 5. Collect Winning Kernels

```bash
docker exec "$CONTAINER_NAME" bash -c "
    cd /workspace/problems
    python3 -c '
import json, shutil, os, glob

manifest = json.load(open(\"optimization_manifest.json\"))
opt_lookup = {o[\"name\"]: o for o in manifest.get(\"optimizations\", [])}

results = []
for best_file in sorted(glob.glob(\"*_opt_best.json\")):
    tracker = json.load(open(best_file))
    name = best_file.replace(\"_opt_best.json\", \"\")
    speedup = tracker.get(\"best_speedup\", 0)
    opt_info = opt_lookup.get(name, {})
    entry = {
        \"name\": name,
        \"speedup\": speedup,
        \"ref_ms\": tracker.get(\"best_ref_time\", 0),
        \"opt_ms\": tracker.get(\"best_opt_time\", 0),
        \"geak_mode\": opt_info.get(\"geak_mode\", \"unknown\"),
        \"kernel_type\": opt_info.get(\"kernel_type\", \"unknown\"),
        \"profiling_pct\": opt_info.get(\"profiling_pct\", 0),
        \"patch_recovered\": tracker.get(\"patch_recovered\", False),
    }
    results.append(entry)
    if speedup > 1.0:
        opt_file = name + \"_opt.py\"
        if os.path.isfile(opt_file):
            shutil.copy2(opt_file, \"/workspace/optimized/\")
            print(f\"  Copied {opt_file} (speedup={speedup:.2f}x, mode={entry[\\\"geak_mode\\\"]})\")
    else:
        print(f\"  Skipped {name} (speedup={speedup:.2f}x < 1.0x)\")

with open(\"geak_results.json\", \"w\") as f:
    json.dump(results, f, indent=2)
print(f\"\\nTotal: {len(results)} kernels, {sum(1 for r in results if r[\\\"speedup\\\"] > 1.0)} with speedup > 1.0x\")
'
"
```

### 6. Clean Up Container

```bash
docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm "$CONTAINER_NAME" 2>/dev/null
```

### 7. Print Summary

```bash
echo ""
echo "============================================"
echo "  Kernel Optimization Summary"
echo "============================================"
python3 -c "
import json, os
results_path = '{{PROBLEMS_DIR}}/geak_results.json'
if os.path.isfile(results_path):
    results = json.load(open(results_path))
    print(f'  {\"Kernel\":40s} {\"Opt.Time%\":>10s} {\"Speedup\":>8s} {\"Ref (ms)\":>10s} {\"Opt (ms)\":>10s} {\"Mode\":>10s}')
    print(f'  {\"─\" * 90}')
    for r in sorted(results, key=lambda x: -x.get('profiling_pct', 0)):
        status = '  OK' if r['speedup'] > 1.0 else 'SKIP'
        pct = r.get('profiling_pct', 0)
        mode = r.get('geak_mode', '?')
        print(f'  {r[\"name\"]:40s} {pct:9.1f}% {r[\"speedup\"]:7.2f}x {r[\"ref_ms\"]:10.4f} {r[\"opt_ms\"]:10.4f} {mode:>10s}  {status}')
else:
    print('  No results found')
"
echo "============================================"

echo "Optimized kernels:"
ls -la {{OPTIMIZED_DIR}}/*_opt.py 2>/dev/null || echo "  (none)"
```

### Manual Fallback

When `GEAK_MODE=manual` (GEAK not available or not desired):

- **HIP/CK kernels**: WARNING that in-place source optimization requires GEAK; attempt Triton replacement instead
- **Vendor kernels**: if source accessible, attempt manual compatibility fixes; Triton replacement only when source inaccessible
- **Triton kernels**: write manually with AMD wave_size=64, BLOCK_SIZE 16-256, num_warps 4-8, `@triton.autotune` with 10-20 configs
- Use profiling data (roofline %, bottleneck recommendations) to guide manual optimization focus
- Test with `kernel_test_runner.py`, iterate up to 5 attempts, finalize

Key manual optimization patterns:
- **AMD GPUs (CDNA3 gfx942 / CDNA4 gfx950)**: wave_size=64, BLOCK_SIZE 16-256, num_warps 4-8, num_stages 2-4
- **NVIDIA GPUs**: warp_size=32, standard Triton configs
- For RMSNorm fusion: single-pass reduction kernel that computes add+norm in one pass
- For SwiGLU fusion: single kernel that reads x, computes silu(gate)*up, writes output
- For GEMM: tiled matrix multiply with shared memory staging

## Completion

Update progress.json:
```json
{
  "phase": "kernel-optimize",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze", "problem-generate", "kernel-optimize"],
  "current_step": "kernel optimization complete",
  "details": {
    "kernels_attempted": "<count>",
    "kernels_improved": "<count with speedup > 1.0x>",
    "geak_used": "<true/false>",
    "geak_mode": "<auto/manual/full/triton_only>"
  }
}
```
