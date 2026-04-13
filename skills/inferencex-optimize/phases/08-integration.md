> **ARCHIVE**: This file is a reference copy of the original phase runbook. The active
> agent docs are in `agents/phase-NN-*.md`. Script paths in this file reference the
> pre-reorganization flat layout (`scripts/*.py`); the actual scripts are now under
> `scripts/{env,container,profiling,optimize,plugin,report}/`.

# Phase 8: Integration & End-to-End Benchmarking {{SKIP_LABEL}}

## Objective
Integrate optimized kernels into the inference framework and measure ACTUAL end-to-end serving throughput using InferenceX Docker containers.

## Config Resolution

Before executing, read `{{OUTPUT_DIR}}/results/sweep_configs.json` (or `{{OUTPUT_DIR}}/config.json`) and set shell variables: `RUNNER`, `IMAGE`, `FRAMEWORK`, `MODEL`, `PRECISION`, `TP`, `EP`, `CONC`, `ISL`, `OSL`, `MAX_MODEL_LEN`, `EXP_NAME`, `BENCHMARK_SCRIPT`.

## MANDATORY: Real Measured Data Only

This phase is NOT complete until a baseline result, a patched server benchmark, and a passing validation exist. FORBIDDEN: estimating speedup, copying baseline numbers, or skipping the patched server benchmark.

## Steps

### 0. Verify Winning Kernels

Run: `python3 "{{SCRIPTS_DIR}}/verify_winning_kernels.py" --problems-dir "{{PROBLEMS_DIR}}" --optimized-dir "{{OPTIMIZED_DIR}}"`

Exit code 1 means missing kernels — run Phase 7 Step 3.5 patch recovery first. Only skip integration for kernels with speedup <= 1.0x.

### 0b. GPU State Cleanup
```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f sglang 2>/dev/null || true
sleep 5
```

### 1. Generate Framework-Specific Plugin

**For vLLM:**
```bash
python3 "{{SCRIPTS_DIR}}/generate_vllm_plugin.py" --kernel-dir "{{OPTIMIZED_DIR}}"
```

**For SGLang:**
```bash
python3 "{{SCRIPTS_DIR}}/generate_sglang_plugin.py" --kernel-dir "{{OPTIMIZED_DIR}}"
```

Produces `{{OPTIMIZED_DIR}}/<framework>_plugin/` with `__init__.py`, launcher script, and `manifest.json`.

### 1.5. Dispatch-Level Optimization (if applicable)

**Trigger**: `geak_results.json` contains vendor/CK/HIP/MoE/attention kernel with `speedup > 1.0` where the optimization is dispatch-level (mentions "dispatch", "backend", "BLAS", "config", "selection", or "NT layout"), OR no mappable kernels found but optimizations exist.

Dispatch-level optimizations change HOW the framework selects/configures a kernel rather than replacing it.

| Kernel Type | Dispatch Mechanism | Patch Target (SGLang) |
|---|---|---|
| Dense GEMM (vendor) | BLAS backend selection per shape | `UnquantizedLinearMethod.apply` |
| MoE GEMM (CK/aiter) | Expert routing config | `FusedMoE.forward` or aiter dispatch |
| Attention (CK/aiter/ASM) | Backend selection per seq length | `AttentionBackend` or MLA dispatch |
| Normalization (CK/aiter) | Fused vs unfused selection | `RMSNorm.forward` or aiter dispatch |

**1.5a. Micro-benchmark**: Generate `{{OPTIMIZED_DIR}}/bench_dispatch.py` testing dispatch alternatives across kernel shapes from the manifest. Run inside the container to identify per-shape winners.

**1.5b. Generate dispatch plugin**: If micro-benchmarks confirm non-default dispatch for any shapes, generate monkey-patch plugin targeting the framework's dispatch point. Pattern: check shape -> override config -> call original -> restore default. See `{{TEMPLATES_DIR}}/dispatch_plugin_example.py` for the SGLang GEMM reference implementation.

**1.5c. Generate standalone launcher**: `{{OPTIMIZED_DIR}}/launch_patched.py` — imports plugin before server startup.

**1.5d. Generate standalone E2E benchmark**: `{{OPTIMIZED_DIR}}/run_e2e_benchmark.sh` — replaces script injection for dispatch plugins. When this exists, skip Steps 4-5 and run the standalone benchmark directly, then proceed to Step 6.

### 2. Retrieve Baseline Benchmark Data

Reuse Phase 2/3 data. Find benchmark result JSONs in `{{RESULTS_DIR}}/`, excluding summary/config files. Prefer low-concurrency results matching the profiling config.

### 3. Start Optimized Benchmark Container

Set framework-specific env vars before starting the container:

**For SGLang:**
```bash
bash "{{SCRIPTS_DIR}}/start_profile_container.sh" \
    --name "inferencex-optimized-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{RESULTS_DIR}}" \
    --mode optimize \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --env "PYTHONPATH=/sgl-workspace/aiter:/workspace/optimized" \
    --env "SGLANG_USE_AITER=1" \
    --env "ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
```

**For vLLM:**
```bash
bash "{{SCRIPTS_DIR}}/start_profile_container.sh" \
    --name "inferencex-optimized-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{RESULTS_DIR}}" \
    --mode optimize \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --env "PYTHONPATH=/workspace/optimized" \
    --env "ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
```

Use the block matching `$FRAMEWORK`. `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` applies to both frameworks — it controls RCCL all-reduce quantization for communication bandwidth, independent of model weight precision.

### 4. Inject Plugin Into Benchmark Script

Restore script first: `cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" 2>/dev/null || true`

```bash
docker cp "{{SCRIPTS_DIR}}/inject_plugin.py" "$CONTAINER_NAME:/tmp/"
docker exec "$CONTAINER_NAME" python3 /tmp/inject_plugin.py \
    --framework "$FRAMEWORK" --target "/workspace/$BENCHMARK_SCRIPT"
```

### 5. Run Optimized Benchmark

Use the SAME benchmark mode as Phase 2 (compiled by default). Stream output live with heartbeats.

```bash
bash "{{SCRIPTS_DIR}}/run_profile_exec.sh" \
    --container "$CONTAINER_NAME" \
    --benchmark-script "$BENCHMARK_SCRIPT" \
    --model "$MODEL" --tp "$TP" --ep "$EP" --conc "$CONC" \
    --isl "$ISL" --osl "$OSL" --max-model-len "$MAX_MODEL_LEN" \
    --precision "$PRECISION" --framework "$FRAMEWORK" --exp-name "$EXP_NAME" \
    --result-filename "optimized_${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}" \
    --repo-dir "{{REPO_DIR}}" --gpus "{{GPUS}}"
```

Collect results from repo root and results/ subdirectory into `{{RESULTS_DIR}}/`.

### 6. Validate Results

Run: `python3 "{{SCRIPTS_DIR}}/validate_optimization.py" --results-dir "{{RESULTS_DIR}}"`

Produces `optimization_comparison.json` with baseline/optimized throughput and speedup.

### 7. Clean Up
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
docker stop "$CONTAINER_NAME" 2>/dev/null; docker rm "$CONTAINER_NAME" 2>/dev/null
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
