# Triton Kernel Optimization Knowledge Base

This file is a persistent reference for the kernel optimization agent. Read it at the START of each optimization session and re-read it when stuck.

## 1. Autotuning

Always use `@triton.autotune` with a range of tile sizes. Never hardcode a single config.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=2),
        # ... more configs
    ],
    key=['M', 'N', 'K'],
)
```

Use `@triton.heuristics` for compile-time decisions:
```python
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0})
```

When EVEN_K is True, skip boundary masks in the K loop — saves 5-10%.

## 2. Stride Assumptions

`tl.assume(stride > 0)` tells the compiler the stride is positive, enabling wider vector loads:
```python
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)
tl.assume(stride_bk > 0)
tl.assume(stride_bn > 0)
```
Typical impact: 5-15% on AMD GPUs.

## 3. GROUP_SIZE_M for L2 Cache Locality

GROUP_SIZE_M controls how thread blocks are grouped along the M dimension. A good value improves L2 cache tile reuse.

- Sweep GROUP_SIZE_M in [1, 2, 4, 8, 16]
- Smaller GROUP_SIZE_M = better L2 reuse for tall matrices (M >> N)
- Larger GROUP_SIZE_M = better for wide matrices (N >> M)
- For square matrices, GROUP_SIZE_M=8 is a good starting point.

## 4. Split-K for Large K Dimensions

When K > 4096 and the grid is small (M and N are small), splitting the K dimension across thread blocks improves parallelism.

```python
# Each block computes partial C[m,n] for a slice of K
pid_k = tl.program_id(1)  # K-split index
k_start = pid_k * (K // NUM_KSPLIT)
k_end = min(k_start + (K // NUM_KSPLIT), K)
# ... accumulate partial results ...
# Reduce: atomic add or separate reduction kernel
```

Warning: atomic operations may not work correctly for bf16 on some AMD GPUs. Use fp32 atomics or a two-pass approach (write partials, then reduce).

## 5. Cache Modifiers

Control L2 cache behavior:
```python
a = tl.load(a_ptrs, cache_modifier=".ca")  # Cache All levels — for reused data
b = tl.load(b_ptrs, cache_modifier=".cg")  # Cache Global only — for streaming data
```

## 6. Persistent Kernels

For operations called repeatedly with the same shape, keep the kernel alive and process multiple tiles:
```python
for tile_id in range(num_tiles_per_cta):
    # compute tile
    tl.debug_barrier()  # sync between tiles if needed
```

Warning: On RDNA3, persistent kernels may hurt because the hardware scheduler is already efficient at tile dispatch. Test before committing.

## 7. Fused Operations

Fusing adjacent operations eliminates intermediate memory traffic:
- **GEMM + bias + activation**: One kernel instead of three
- **Residual add + RMSNorm**: Fuse addition and normalization
- **SiLU(gate) * up**: Fuse activation with elementwise multiply

For fused RMSNorm, process one row per program:
```python
row = tl.program_id(0)
x = tl.load(X + row * stride + tl.arange(0, BLOCK_N))
# ... fuse residual add + variance + normalize ...
tl.store(Out + row * stride + tl.arange(0, BLOCK_N), result)
```

## 8. GPU Architecture Reference

### AMD CDNA3 (MI300X, MI300A) — gfx940, gfx942
- Wave64 (64-wide wavefronts)
- MI300X: 8 XCDs (compute chiplets), multi-die — `remap_xcd()` for PID remapping across XCDs
- Matrix cores: MFMA instructions
- HBM3 memory

### AMD RDNA3 (R7900, RX 7900 XTX) — gfx1100 / gfx1101
- Wave32 (32-wide wavefronts)
- Chiplet: 1x GCD (compute) + 6x MCD (memory/cache) on gfx1100; 4x MCD on gfx1101
- GCD: 96 CUs (gfx1100), 60 CUs (gfx1101), 6 Shader Engines
- MCD: each has 16MB Infinity Cache (L3) + 64-bit GDDR6 controller
- L2: 6MB shared; Infinity Cache: 96MB distributed across MCDs
- MCD is memory-only, not compute — L2 cache remapping is not needed
- WMMA instructions; BF16 supported natively

### AMD RDNA4 — gfx12xx
- Wave32
- Improved WMMA; refined cache hierarchy
- Details TBD as hardware matures

### How to use architecture info
- Read `rocminfo` or `gpu_arch.json` for the actual GPU
- Use `rocprofv3` kernel trace to get per-kernel HW counters (VGPR, SGPR, LDS, workgroup/grid sizes, dispatch duration)
- Profiling data should drive optimization decisions

## 9. Debugging Guide

### Correctness failure
- Check accumulator dtype (use fp32 for bf16 inputs)
- Check boundary masks (EVEN_K heuristic)
- Check output dtype cast (`.to(C.dtype.element_ty)`)
- For fused kernels: check intermediate precision (do fused computation in fp32)

### GPU memory access fault (SIGABRT)
- Tile size too large → reduce BLOCK_M, BLOCK_N
- Register spill → reduce num_warps, reduce tile size
- Out of LDS → reduce BLOCK_K or remove shared memory usage

### Performance regression
- Check if autotune selected a bad config (print autotune result)
- Measure launch overhead vs compute time
- Compare grid size to CU count (underfull GPU = bad)

## 10. Lessons from Experiments

### Triton autotune in LLM serving (measured on gfx1100 + Qwen3-8B)
- Micro-benchmark: autotune finds 1.1x avg speedup over rocBLAS for GEMM
- E2E serving: **0.43x throughput** (2.3x SLOWER) due to autotune overhead
- Root cause: dynamic batching causes M to change every iteration; each new (M,N,K) triggers autotune benchmark sweep
- Fix: pre-warm autotune cache for all expected M values before serving, OR use fixed configs

### Integration path for vLLM on ROCm
- Patch point: `vllm.model_executor.layers.linear.dispatch_unquantized_gemm` (NOT `utils.rocm_unquantized_gemm_impl` — the torch custom op captures the reference at registration time)
- Injection: `sitecustomize.py` via `PYTHONPATH` (works in spawn'd child processes)
- Weight convention: vLLM weight is (out_features, in_features), need `weight.t()` for Triton

## 11. Available Techniques Catalog

These are techniques you CAN try. Do NOT follow them in fixed order. Choose based on profiling data.

- `@triton.autotune` with varying tile sizes, num_warps, num_stages
- `@triton.heuristics` for compile-time decisions (e.g., skip masks)
- `tl.assume(stride > 0)` — compiler hint for wider loads
- GROUP_SIZE_M — controls L2 tile reuse pattern
- BLOCK_K sizing — affects K-loop trip count and register usage
- num_stages — software pipelining depth
- Cache modifiers — `.ca`, `.cg` for L2 control
- Split-K — decompose K dimension across CTAs
- Persistent kernels — amortize launch overhead
- Operation fusion — reduce memory traffic
- PID swizzling — remap thread block IDs for better cache behavior

## 11. Decision Process

1. Profile first (rocprofv3 or benchmark)
2. Identify the bottleneck from data (compute-bound? memory-bound? launch overhead? register spill?)
3. Choose ONE technique that addresses the observed bottleneck
4. Implement, test correctness, benchmark
5. Accept if better, reject with data-backed reason if not
6. Re-profile after acceptance to see if the bottleneck shifted
7. Repeat until the configured stopping criteria are met
