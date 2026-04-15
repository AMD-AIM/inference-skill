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

## 8. GPU Architecture Notes

### AMD CDNA (MI250X, MI300X) — gfx90a, gfx940, gfx942
- Wave64 (64-wide wavefronts)
- Multi-die: MI250X has 2 GCDs, MI300X has 8 XCDs
- XCD remapping improves die utilization — USE `remap_xcd(pid, GRID_MN)`
- Matrix cores: MFMA instructions
- Large tiles OK: 128x128, 128x256

### AMD RDNA3 (R7900, RX 7900 XTX) — gfx1100
- **Wave32** (32-wide wavefronts, NOT 64)
- **Single GCD** — NO XCD remapping needed
- 96 CUs, 2 SIMDs per CU
- 6MB L2 cache
- num_warps: 2 or 4 recommended; 8 causes register spill
- Max tile: 128x64 or 64x128 — avoid 128x128+
- `num_stages=0` is broken in Triton 3.5.1 on gfx1100 — use 1 or 2
- WMMA instructions (smaller matrix units than MFMA)

### AMD RDNA2 (RX 6000 series) — gfx1030
- Wave32
- Single die
- Fewer CUs, smaller L2
- Similar constraints to RDNA3 but slower

### NVIDIA (A100, H100)
- Wave32 (warp=32)
- Tensor cores (fp16, bf16, int8, fp8)
- Large tiles OK: 128x256, 256x128
- num_warps up to 8 is fine
- num_stages=3-5 for software pipelining

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

## 10. Optimization Priority Checklist

When starting optimization for a new kernel:

1. [ ] Write basic Triton kernel with `@triton.autotune` (3-5 safe configs)
2. [ ] Add `tl.assume(stride > 0)` for all strides
3. [ ] Add `EVEN_K` heuristic if applicable
4. [ ] Sweep GROUP_SIZE_M [1, 2, 4, 8, 16]
5. [ ] Try larger BLOCK_K (64, 128)
6. [ ] Try `num_stages=2` (software pipelining)
7. [ ] If kernel is memory-bound: try cache modifiers
8. [ ] If K is large and grid is small: try split-K
9. [ ] If multiple adjacent ops: try fusing them
10. [ ] Run rocprofv3 to check VGPR/SGPR/LDS usage
11. [ ] If VGPR > 128: reduce tile size or num_warps
12. [ ] If kernel is already near roofline: stop

## 11. When to Stop Optimizing

- 3+ consecutive rejected attempts with different approaches
- Kernel is within 10% of theoretical memory bandwidth limit
- rocprofv3 shows CU utilization > 80%
- Further improvements require algorithm changes (not kernel tuning)
