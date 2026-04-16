# Phase 8: Integration & E2E Verification {{SKIP_LABEL}}

## Goal
Integrate the optimized kernel(s) into vLLM inference and **prove** they are used, with correctness and performance data.

## Success criteria
1. vLLM server starts with the optimized kernel loaded — **no system library files modified**
2. During inference, the optimized kernel is **verifiably called** (call count > 0)
3. Inference outputs match the unpatched baseline (correctness preserved)
4. E2E throughput is **not degraded** compared to baseline

## Constraints
- **DO NOT modify any file under `/opt/`, `/usr/`, or any pip-installed package.**
- Use `{{SCRIPTS_DIR}}/select_gpus.py` to pick GPU(s).
- Server startup must use fail-fast detection (same pattern as Phase 1 Step 7).
- If the patched server is slower or incorrect, **restore the original server**.

## Verified integration path (from experiments)

### How vLLM dispatches GEMM on ROCm
```
UnquantizedLinearMethod.apply()
  → dispatch_unquantized_gemm()()   # called on EVERY forward pass
    → rocm_unquantized_gemm(layer, x, weight, bias)
      → torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)
        → rocm_unquantized_gemm_impl(x, weight, bias)
          → torch.nn.functional.linear(x, weight, bias)  [on gfx1100]
```

### Where to patch
**Patch `linear_mod.dispatch_unquantized_gemm`** in `vllm.model_executor.layers.linear`. This is the name binding that `apply()` calls each time. Patching `utils.rocm_unquantized_gemm_impl` does NOT work because the torch custom op captures the function reference at registration time.

### How to inject without modifying system files
Use a Python `meta_path` import hook via `sitecustomize.py`:
1. Write `auto_patch.py` in `{{OPTIMIZED_DIR}}` with a `_PatchFinder` meta_path hook
2. Write `sitecustomize.py` in `{{OPTIMIZED_DIR}}` that does `import auto_patch`
3. Set `PYTHONPATH={{OPTIMIZED_DIR}}:$PYTHONPATH`
4. Start vLLM normally — `sitecustomize.py` runs in ALL processes (including spawn'd children)

### CRITICAL: Triton autotune in serving

**WARNING**: Triton `@triton.autotune` causes severe performance regression in serving mode.

In micro-benchmarks, autotune finds optimal configs for fixed shapes. But in vLLM serving with dynamic batching, M (batch size) changes every iteration. Each new (M, N, K) combination triggers autotune's benchmark sweep, adding massive overhead.

**Measured impact**: 1.1x micro-benchmark speedup → 0.43x E2E regression (2.3x SLOWER).

**Mitigations** (the agent should try one of these):
1. **Pre-warm autotune**: Before accepting requests, run the Triton kernel through all expected M values (1..max_batch_size) to populate the autotune cache
2. **Fixed config**: Replace `@triton.autotune` with a single best config (determined during Phase 7) to eliminate runtime benchmarking
3. **Shape-bucketed dispatch**: Only route to Triton for a small set of pre-warmed M values; fallback to original for others

## Steps

### Step 1: Baseline E2E benchmark
Run a short benchmark on the current **unpatched** server. Save to `{{OUTPUT_DIR}}/results/baseline_e2e.json`.

### Step 2: Create the patch
Write the auto_patch.py and sitecustomize.py in `{{OPTIMIZED_DIR}}`.

The optimized kernel must handle the vLLM weight convention: weight is `(out_features, in_features)`, so call `triton_fn(x_2d, weight.t())`.

The patch must:
- Track call counts (`triton`, `fallback`, `error`) and persist to a JSON file
- Catch all exceptions and fall back to original

### Step 3: Address the autotune problem
Before deploying, fix the autotune overhead. The recommended approach: run a pre-warm loop that calls the Triton kernel with M values [1, 2, 4, 8, 16, 32, 64] to populate the autotune cache, then start serving.

### Step 4: Kill current server, start patched server
- Select GPU with `select_gpus.py`
- Set `PYTHONPATH={{OPTIMIZED_DIR}}:$PYTHONPATH`
- Start vLLM, wait with fail-fast detection

### Step 5: Verify kernel is active
Send requests, then read the stats JSON file. `triton` count must be > 0.

### Step 6: Correctness check
Same prompt with temperature=0 → output must match baseline.

### Step 7: Optimized E2E benchmark
Same benchmark as Step 1. If throughput is LOWER than baseline, the integration is a failure — restore original.

### Step 8: Report
Save comparison to `{{REPORT_DIR}}/integration_report.md`. Include call counts, throughput, latency, correctness.

## Completion
Update progress.json with results.
