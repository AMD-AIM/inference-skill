# Phase 8: Integration & End-to-End Benchmarking {{SKIP_LABEL}}

## Objective
Integrate optimized kernels into the inference framework and measure ACTUAL end-to-end serving throughput using InferenceX Docker containers.

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
CONC="<conc field from config>"
ISL="<isl field from config>"
OSL="<osl field from config>"
MAX_MODEL_LEN="<max-model-len field from config>"
EXP_NAME="<exp-name field from config>"
BENCHMARK_SCRIPT="<resolved benchmark script from Phase 1>"
```

These variables are used in docker commands and benchmark execution throughout this phase.

## ⛔ MANDATORY: This phase requires REAL measured data

**This phase is NOT complete until:**
1. A baseline benchmark result exists (reused from Phase 2/3)
2. A patched server has ACTUALLY started and served requests
3. An optimized benchmark result exists with real data
4. The validation script passes with `optimization_comparison.json`

**FORBIDDEN:**
- Estimating speedup with Amdahl's law
- Copying baseline numbers and modifying them
- Reporting "estimated" or "conservative" speedup
- Skipping the patched server benchmark

## ⚠️ CRITICAL: Best Kernel Capture Verification Before Integration

**Phase 7's Step 3.5 should have already captured all best kernels. Before proceeding, verify:**

```bash
python3 -c "
import json, os, glob
results_path = '{{PROBLEMS_DIR}}/geak_results.json'
if not os.path.isfile(results_path):
    print('WARNING: geak_results.json not found — no kernels to integrate')
    exit(0)
results = json.load(open(results_path))
missing = []
for r in results:
    if r['speedup'] > 1.0:
        opt_file = os.path.join('{{OPTIMIZED_DIR}}', r['name'] + '_opt.py')
        if not os.path.isfile(opt_file):
            missing.append(r['name'])
if missing:
    print(f'ERROR: {len(missing)} winning kernels NOT captured in {{OPTIMIZED_DIR}}/: {missing}')
    print('Run Phase 7 Step 3.5 patch recovery before proceeding!')
    exit(1)
else:
    winners = sum(1 for r in results if r['speedup'] > 1.0)
    print(f'OK: All {winners} winning kernels captured in {{OPTIMIZED_DIR}}/')
"
```

**Only skip integration for a kernel if its speedup <= 1.0x** (performance degraded or no improvement). Do NOT proceed to benchmarking until all winning kernels are integrated into the plugin.

## Steps

### 0. GPU State Cleanup

```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f sglang 2>/dev/null || true
sleep 5
rocm-smi --showmeminfo vram 2>/dev/null || nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
```

### 1. Generate Framework-Specific Plugin

#### If FRAMEWORK is vllm:

```bash
python3 "{{SCRIPTS_DIR}}/generate_vllm_plugin.py" \
    --kernel-dir "{{OPTIMIZED_DIR}}"

echo "vLLM plugin files:"
ls -la "{{OPTIMIZED_DIR}}/vllm_plugin/" 2>/dev/null
cat "{{OPTIMIZED_DIR}}/vllm_plugin/manifest.json" 2>/dev/null
```

This generates:
- `{{OPTIMIZED_DIR}}/vllm_plugin/__init__.py` — registers CustomOps via `CustomOp.register_oot()`
- `{{OPTIMIZED_DIR}}/run_patched_vllm.py` — launcher script
- `{{OPTIMIZED_DIR}}/vllm_plugin/manifest.json` — registration summary

#### If FRAMEWORK is sglang:

```bash
python3 "{{SCRIPTS_DIR}}/generate_sglang_plugin.py" \
    --kernel-dir "{{OPTIMIZED_DIR}}"

echo "SGLang plugin files:"
ls -la "{{OPTIMIZED_DIR}}/sglang_plugin/" 2>/dev/null
cat "{{OPTIMIZED_DIR}}/sglang_plugin/manifest.json" 2>/dev/null
```

This generates:
- `{{OPTIMIZED_DIR}}/sglang_plugin/__init__.py` — monkey-patches SGLang modules
- `{{OPTIMIZED_DIR}}/run_patched_sglang.py` — launcher script
- `{{OPTIMIZED_DIR}}/sglang_plugin/manifest.json` — patch summary

### 1.5. Dispatch-Level Optimization Integration

**Trigger**: `{{PROBLEMS_DIR}}/geak_results.json` contains a vendor, CK, HIP, MoE, or attention kernel entry with `speedup > 1.0` where the optimization is dispatch-level rather than a drop-in kernel replacement (note mentions "dispatch", "backend", "BLAS", "config", "selection", or "NT layout"), OR the plugin generator in Step 1 found no mappable kernels but kernel optimizations exist in `geak_results.json`.

Dispatch-level optimizations change HOW the framework selects or configures a kernel (backend, algorithm, tile config, launch parameters) rather than replacing the kernel itself. Common patterns by kernel type:

| Kernel type | Dispatch mechanism | Patch target (SGLang) | Example |
|---|---|---|---|
| Dense GEMM (vendor) | BLAS backend selection per shape | `UnquantizedLinearMethod.apply` | CK for small-K, hipBLASLt for large-K |
| MoE GEMM (CK/aiter) | Expert routing config, stage selection | `FusedMoE.forward` or aiter dispatch | 1-stage vs 2-stage, tile size per expert count |
| Attention (CK/aiter/ASM) | Backend selection per seq length | `AttentionBackend` or MLA dispatch | Flash vs CK vs ASM by (seq_q, n_heads, d_qk) |
| Normalization (CK/aiter) | Fused vs unfused selection | `RMSNorm.forward` or aiter dispatch | Fused add+norm+quant vs separate kernels |

When triggered, this step generates a monkey-patch plugin for the dispatch decision. Kernel-level patches from Step 1 (Triton replacements, etc.) are still used if present.

#### 1.5a. Micro-benchmark validation

Generate `{{OPTIMIZED_DIR}}/bench_dispatch.py` and run it inside the container. The benchmark tests the dispatch alternatives identified by GEAK for the relevant kernel type.

**For Dense GEMM** — test `F.linear` with BLAS backends (cublaslt, hipblaslt, ck) across all shapes from the manifest:

```python
#!/usr/bin/env python3
"""Benchmark dispatch alternatives for kernel shapes."""
import torch
import torch.nn.functional as F
import time, statistics, json

SHAPES = {{KERNEL_SHAPES}}  # populated from manifest: list of (M, K, N) tuples
DISPATCH_ALTERNATIVES = {{DISPATCH_ALTERNATIVES}}  # e.g., ['cublaslt', 'hipblaslt', 'ck'] for GEMM

results = {}
for alt in DISPATCH_ALTERNATIVES:
    torch.backends.cuda.preferred_blas_library(alt)
    alt_results = []
    for shape in SHAPES:
        M, K, N = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        for _ in range(30):
            F.linear(x, w)
        torch.cuda.synchronize()
        times = []
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            F.linear(x, w)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        alt_results.append(statistics.median(times))
    results[alt] = alt_results

preferred = {}
for i, shape in enumerate(SHAPES):
    vals = {a: results[a][i] for a in DISPATCH_ALTERNATIVES}
    best = min(vals, key=vals.get)
    default = DISPATCH_ALTERNATIVES[0]
    if best != default:
        speedup = vals[default] / vals[best]
        preferred[str(shape)] = {"best": best, "speedup": round(speedup, 4)}
    print(f"  {shape}: best={best} " + " ".join(f"{a}={results[a][i]:.4f}" for a in DISPATCH_ALTERNATIVES))

with open('/workspace/optimized/micro_benchmark_validation.json', 'w') as f:
    json.dump({"shapes": SHAPES, "results": results, "preferred": preferred}, f, indent=2)
print(f"\nNon-default preferred: {len(preferred)} shapes")
```

For **MoE or attention** kernels, the agent should write an analogous benchmark that exercises the specific dispatch alternatives found by GEAK (e.g., different `fused_moe` stage configs, different attention backend calls). The structure is the same: iterate alternatives x shapes, measure, identify per-shape winners.

Run inside the container:
```bash
docker exec $GPU_ENV "$CONTAINER_NAME" python3 /workspace/optimized/bench_dispatch.py
```

#### 1.5b. Generate dispatch plugin

If micro-benchmarks confirm any shapes prefer a non-default dispatch, generate `{{OPTIMIZED_DIR}}/sglang_plugin/__init__.py` (for SGLang). The plugin monkey-patches the framework's dispatch point for the relevant kernel type.

The general pattern:
1. Identify the framework function that dispatches to the kernel (the "patch target" from the table above)
2. Replace it with a wrapper that checks input shape against a set of preferred configurations
3. For matching shapes, switch dispatch (e.g., set backend, change config), call original, restore default
4. For non-matching shapes, call original unchanged

**Dense GEMM example** (verified on dsr1-fp4-mi355x-sglang):

```python
"""SGLang Plugin — Dispatch-level kernel optimization.
Patches the framework's dispatch point to use per-shape optimal backend/config.
"""
import torch
import torch.nn.functional as F

# Per-shape dispatch overrides. Populated from micro_benchmark_validation.json.
# Format depends on kernel type:
#   GEMM: {(M, K, N): backend_name, ...}
#   MoE:  {(batch_tokens, num_experts, topk): config_dict, ...}
#   Attn: {(seq_q, n_heads, d_qk): backend_name, ...}
_DISPATCH_OVERRIDES = {
    # Example for GEMM: (705, 512, 4096): 'ck',
}
_DEFAULT_DISPATCH = 'hipblaslt'  # restored after each override

_patched = False

def _optimized_apply(self, layer, x, bias=None):
    M = x.shape[0] if x.dim() == 2 else x.shape[0]
    K = x.shape[-1]
    N = layer.weight.shape[0]
    key = (M, K, N)
    override = _DISPATCH_OVERRIDES.get(key)
    if override:
        torch.backends.cuda.preferred_blas_library(override)
        result = F.linear(x, layer.weight, bias)
        torch.backends.cuda.preferred_blas_library(_DEFAULT_DISPATCH)
        return result
    return F.linear(x, layer.weight, bias)

def patch_all():
    global _patched
    if _patched:
        return
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        UnquantizedLinearMethod.apply = _optimized_apply
        _patched = True
        print("[sglang_plugin] Patched UnquantizedLinearMethod.apply with per-shape dispatch selection")
        print(f"[sglang_plugin]   Overrides: {_DISPATCH_OVERRIDES}")
    except Exception as e:
        print(f"[sglang_plugin] Deferred: {e}")

try:
    patch_all()
except Exception:
    pass
```

For **MoE or attention** kernels, the agent should follow the same pattern but target the appropriate dispatch function. For example, MoE dispatch might patch `FusedMoE.forward` to select between 1-stage and 2-stage configs based on expert count and batch size. Attention dispatch might patch the attention backend selection based on sequence length. The monkey-patch structure (check shape -> override config -> call original -> restore default) is the same regardless of kernel type.

#### 1.5c. Generate standalone launcher

Generate `{{OPTIMIZED_DIR}}/launch_patched.py`:

```python
#!/usr/bin/env python3
"""Launch SGLang server with dispatch optimization plugin."""
import sys
import os

sys.path.insert(0, '/sgl-workspace/aiter')
sys.path.insert(0, '/workspace/optimized')

# Import the target module first so the plugin can find it
# For GEMM: UnquantizedLinearMethod; for MoE: FusedMoE; etc.
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
import sglang_plugin
sglang_plugin.patch_all()

from sglang.launch_server import prepare_server_args, run_server
if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
```

#### 1.5d. Generate standalone E2E benchmark script

Generate `{{OPTIMIZED_DIR}}/run_e2e_benchmark.sh`. This replaces the script-injection approach (Step 4) for dispatch plugins since the plugin requires a different server launch mechanism:

```bash
#!/usr/bin/env bash
set -e
export MODEL="$MODEL"
export HF_HUB_CACHE="${HF_CACHE}"
export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONPATH=/sgl-workspace/aiter:/workspace/optimized:$PYTHONPATH

PORT=8193
RESULT_DIR="/workspace/results"
mkdir -p "$RESULT_DIR"

python3 /workspace/optimized/launch_patched.py \
    --model-path=$MODEL --trust-remote-code \
    --host=0.0.0.0 --port=$PORT \
    --tensor-parallel-size=$TP \
    --chunked-prefill-size=196608 --mem-fraction-static=0.8 \
    --disable-radix-cache --num-continuous-decode-steps=4 \
    --max-prefill-tokens=196608 --cuda-graph-max-bs=128 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 > /workspace/server.log 2>&1 &
SERVER_PID=$!

MAX_WAIT=600; WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    curl -s http://localhost:$PORT/health > /dev/null 2>&1 && break
    kill -0 $SERVER_PID 2>/dev/null || { tail -80 /workspace/server.log; exit 1; }
    sleep 5; WAITED=$((WAITED + 5))
done
[ $WAITED -ge $MAX_WAIT ] && { tail -80 /workspace/server.log; kill $SERVER_PID; exit 1; }

grep -i 'sglang_plugin' /workspace/server.log || true

python3 /workspace/InferenceX/utils/bench_serving/benchmark_serving.py \
    --model "$MODEL" --backend vllm \
    --base-url "http://0.0.0.0:$PORT" \
    --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL \
    --random-range-ratio 0.5 \
    --num-prompts 40 --max-concurrency $CONC \
    --request-rate inf --ignore-eos --save-result --num-warmups 8 \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --result-dir "$RESULT_DIR/" \
    --result-filename "optimized_${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}.json"

kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
```

When Step 1.5 generates the standalone benchmark, skip Step 4 (script injection) and Step 5 (standard benchmark). Instead, run the standalone benchmark directly in Step 5 via:

```bash
docker run -d --name "$CONTAINER_NAME" \
    $GPU_FLAGS --shm-size 64g --ipc=host --network=host \
    -v {{HF_CACHE}}:/data/huggingface/hub \
    -v {{OPTIMIZED_DIR}}:/workspace/optimized \
    -v {{REPO_DIR}}:/workspace/InferenceX \
    -e SGLANG_USE_AITER=1 -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
    -e PYTHONPATH=/sgl-workspace/aiter:/workspace/optimized \
    $IMAGE bash /workspace/optimized/run_e2e_benchmark.sh
```

Then proceed to Step 6 (validation) as normal.

### 2. Retrieve Baseline Benchmark Data

Reuse Phase 2/3 data. InferenceX benchmark results are named like:
`<exp>_<seq>_<precision>_<framework>_tp<N>-ep<N>_conc<N>.json`

```bash
echo "Looking for baseline benchmark data..."
python3 -c "
import json, glob, os

results_dir = '{{RESULTS_DIR}}'
# Find the benchmark result matching the profiling config
candidates = sorted(glob.glob(os.path.join(results_dir, '*.json')))
# Exclude non-benchmark files
candidates = [c for c in candidates if not any(skip in os.path.basename(c)
    for skip in ['benchmark_summary', 'bottlenecks', 'sweep_configs', 'optimization', 'optimized_'])]

if candidates:
    # Prefer low-concurrency result (typically used for profiling)
    selected = candidates[0]
    for c in candidates:
        if '_conc4' in c or '_conc1' in c:
            selected = c
            break
    print(f'BASELINE_FILE={selected}')
    data = json.load(open(selected))
    print(f'Baseline throughput: {data.get(\"total_token_throughput\", data.get(\"output_throughput\", \"N/A\"))} tok/s')
else:
    print('ERROR: No baseline benchmark results found in {{RESULTS_DIR}}/')
    print('Phase 2 (benchmark) must be run first.')
"
```

### 3. Start Optimized Benchmark Container

```bash
if [[ "$RUNNER" == mi* ]]; then
    GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"
else
    GPU_FLAGS="--gpus all"
fi

CONTAINER_NAME="inferencex-optimized-{{CONFIG_KEY}}"

# Clean up any stale container from a previous run
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Check if HIP/CK in-place optimization requires AITER_REBUILD
AITER_REBUILD_ENV=""
python3 -c "
import json, os
manifest_path = '{{PROBLEMS_DIR}}/optimization_manifest.json'
if os.path.isfile(manifest_path):
    manifest = json.load(open(manifest_path))
    for o in manifest.get('optimizations', []):
        if o.get('geak_mode') == 'kernel-url' and o.get('kernel_type') in ('hip', 'ck'):
            print('AITER_REBUILD=1')
            break
" && AITER_REBUILD_ENV="-e AITER_REBUILD=1"

docker run -d \
    --name "$CONTAINER_NAME" \
    --label inferencex-pipeline=true \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 64g \
    --ipc=host \
    --network=host \
    -v {{REPO_DIR}}:/workspace \
    -v {{HF_CACHE}}:/root/.cache/huggingface \
    -v {{OPTIMIZED_DIR}}:/workspace/optimized \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
    -e PYTHONPATH=/sgl-workspace/aiter:/workspace/optimized \
    -e SGLANG_USE_AITER=1 \
    -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
    $AITER_REBUILD_ENV \
    $IMAGE \
    -c "sleep infinity"

echo "Container started: $CONTAINER_NAME"
```

**Note**: SGLang Docker images require `/sgl-workspace/aiter` in PYTHONPATH for the aiter package. The `SGLANG_USE_AITER=1` and `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` env vars are required for MI355X SGLang.

### 4. Inject Plugin Into Benchmark Script

The InferenceX benchmark scripts start the inference server internally. The plugin must be imported BEFORE the server starts.

#### For vLLM:
```bash
# Restore benchmark script to clean state
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" 2>/dev/null || true

# Inject plugin import at the top of the benchmark script
docker exec "$CONTAINER_NAME" python3 -c "
import re
with open('/workspace/$BENCHMARK_SCRIPT') as f:
    content = f.read()

# Add plugin import before vllm serve command
plugin_import = '''
# Optimized kernel plugin (auto-injected by inferencex-optimize Phase 8)
export PYTHONPATH=/workspace/optimized:\$PYTHONPATH
python3 -c \"import sys; sys.path.insert(0, '/workspace/optimized'); import vllm_plugin\" 2>/dev/null || echo \"[WARNING] vllm_plugin import failed\"
'''
content = content.replace('vllm serve ', plugin_import + 'vllm serve ', 1)
with open('/workspace/$BENCHMARK_SCRIPT', 'w') as f:
    f.write(content)
print('Patched benchmark script with vllm_plugin import')
"
```

#### For SGLang:
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" 2>/dev/null || true

docker exec "$CONTAINER_NAME" python3 -c "
with open('/workspace/$BENCHMARK_SCRIPT') as f:
    content = f.read()

# IMPORTANT: Match 'python3 -m sglang.launch_server' as a whole so we
# insert the plugin block BEFORE the full command instead of splitting
# 'python3 -m ' from 'sglang.launch_server'.
plugin_import = '''# Optimized kernel plugin (auto-injected by inferencex-optimize Phase 8)
export PYTHONPATH=/workspace/optimized:\$PYTHONPATH
python3 -c \"import sys; sys.path.insert(0, '/workspace/optimized'); import sglang_plugin\" || echo \"[WARNING] sglang_plugin import failed\"

'''
content = content.replace('python3 -m sglang.launch_server', plugin_import + 'python3 -m sglang.launch_server', 1)
with open('/workspace/$BENCHMARK_SCRIPT', 'w') as f:
    f.write(content)
print('Patched benchmark script with sglang_plugin import')
"
```

### 4b. torch.mm Override for GEMM Kernels (optional)

If any optimized kernels target `aten::mm` (GEMM) shapes without a CustomOp mapping, add a `torch.mm` override to the plugin:

```python
# Added to vllm_plugin/__init__.py or sglang_plugin/__init__.py when GEMM optimizations exist
_original_mm = torch.mm
def _patched_mm(a, b, **kw):
    if (a.shape, b.shape[1]) in OPTIMIZED_SHAPES:
        return optimized_gemm(a, b)
    return _original_mm(a, b, **kw)
torch.mm = _patched_mm
```

This is auto-generated by `generate_vllm_plugin.py` / `generate_sglang_plugin.py` when GEMM problem files with speedup > 1.0x are detected.

### 5. Run Optimized Benchmark

Use the **same benchmark mode as Phase 2** (compiled by default). Do NOT add a separate eager-mode benchmark — the comparison must be apples-to-apples with the baseline. Stream stdout/stderr live to the terminal.

```bash
MANUAL_GPUS="{{GPUS}}"
if [ -n "$MANUAL_GPUS" ]; then
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=$MANUAL_GPUS"
else
    GPU_ENV=""
fi

OPTIMIZED_RESULT="optimized_${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}"
DOCKER_RUN_LOG="{{RESULTS_DIR}}/${CONTAINER_NAME}_docker_run.log"

echo "DOCKER_LOG: $DOCKER_RUN_LOG"
echo "Starting optimized benchmark run: RESULT_FILENAME=$OPTIMIZED_RESULT"

(
    set -o pipefail
    docker exec \
        $GPU_ENV \
        -e MODEL=$MODEL \
        -e TP=$TP \
        -e EP_SIZE=$EP \
        -e CONC=$CONC \
        -e ISL=$ISL \
        -e OSL=$OSL \
        -e MAX_MODEL_LEN=$MAX_MODEL_LEN \
        -e RANDOM_RANGE_RATIO=0.5 \
        -e RESULT_FILENAME=$OPTIMIZED_RESULT \
        -e PRECISION=$PRECISION \
        -e FRAMEWORK=$FRAMEWORK \
        -e EXP_NAME=$EXP_NAME \
        "$CONTAINER_NAME" \
        /bin/bash /workspace/$BENCHMARK_SCRIPT \
        2>&1 | tee -a "$DOCKER_RUN_LOG"
) &
EXEC_PID=$!

while kill -0 "$EXEC_PID" 2>/dev/null; do
    sleep 30
    if kill -0 "$EXEC_PID" 2>/dev/null; then
        echo "[heartbeat] Optimized benchmark still running"
    fi
done

wait "$EXEC_PID"
EXIT_CODE=$?
echo "Optimized benchmark exit code: $EXIT_CODE"
```

Collect results (InferenceX scripts use `--result-dir /workspace/` which writes to repo root):
```bash
# Collect from repo root (where --result-dir /workspace/ writes)
cp {{REPO_DIR}}/${OPTIMIZED_RESULT}*.json "{{RESULTS_DIR}}/" 2>/dev/null || true
rm -f {{REPO_DIR}}/${OPTIMIZED_RESULT}*.json 2>/dev/null || true
# Fallback: also check results/ subdirectory
cp {{REPO_DIR}}/results/${OPTIMIZED_RESULT}*.json "{{RESULTS_DIR}}/" 2>/dev/null || true
rm -f {{REPO_DIR}}/results/${OPTIMIZED_RESULT}*.json 2>/dev/null || true
```

### 6. Validate Results

```bash
python3 << 'VALIDATE'
import json, os, sys, glob

results_dir = "{{RESULTS_DIR}}"

# Find baseline and optimized results
baseline_files = [f for f in glob.glob(os.path.join(results_dir, "*.json"))
                  if not any(skip in os.path.basename(f)
                  for skip in ["benchmark_summary", "bottlenecks", "sweep_configs", "optimization", "optimized_"])]
optimized_files = glob.glob(os.path.join(results_dir, "optimized_*.json"))

errors = []
if not baseline_files:
    errors.append("No baseline benchmark result found")
if not optimized_files:
    errors.append("No optimized benchmark result found — the patched server may have failed")

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

# Load and compare
baseline = json.load(open(baseline_files[0]))
optimized = json.load(open(optimized_files[0]))

bl_tps = baseline.get("total_token_throughput", baseline.get("output_throughput", 0))
opt_tps = optimized.get("total_token_throughput", optimized.get("output_throughput", 0))
speedup = opt_tps / bl_tps if bl_tps > 0 else 1.0

if speedup < 1.0:
    print("WARNING: REGRESSION DETECTED — optimized throughput is lower than baseline")
    print("Investigate: kernel compatibility issues, plugin import failures, or server startup errors")

print("VALIDATION PASSED" if speedup >= 1.0 else "VALIDATION COMPLETED (with regression)")
print(f"  Baseline:  {bl_tps:.1f} tok/s")
print(f"  Optimized: {opt_tps:.1f} tok/s")
print(f"  Speedup:   {speedup:.3f}x")

comparison = {
    "validated": True,
    "baseline_file": os.path.basename(baseline_files[0]),
    "optimized_file": os.path.basename(optimized_files[0]),
    "baseline": {
        "total_token_throughput": bl_tps,
        "mean_ttft_ms": baseline.get("mean_ttft_ms", 0),
        "mean_itl_ms": baseline.get("mean_itl_ms", 0),
        "mean_tpot_ms": baseline.get("mean_tpot_ms", 0),
        "duration_s": baseline.get("duration", 0),
    },
    "optimized": {
        "total_token_throughput": opt_tps,
        "mean_ttft_ms": optimized.get("mean_ttft_ms", 0),
        "mean_itl_ms": optimized.get("mean_itl_ms", 0),
        "mean_tpot_ms": optimized.get("mean_tpot_ms", 0),
        "duration_s": optimized.get("duration", 0),
    },
    "speedup": round(speedup, 4),
}

with open(os.path.join(results_dir, "optimization_comparison.json"), "w") as f:
    json.dump(comparison, f, indent=2)
print(f"  Saved optimization_comparison.json")
VALIDATE
```

### 7. Clean Up Container

```bash
# Restore benchmark script to original state
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true

docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm "$CONTAINER_NAME" 2>/dev/null
```

## Completion

Update progress.json:
```json
{
  "phase": "integration",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze", "problem-generate", "kernel-optimize", "integration"],
  "current_step": "integration complete",
  "details": {
    "baseline_throughput": "<tok/s>",
    "optimized_throughput": "<tok/s>",
    "speedup": "<X.XXx>",
    "plugin_type": "<vllm_plugin or sglang_plugin>"
  }
}
```
