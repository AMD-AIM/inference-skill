# Triton Kernel Optimization Knowledge Base

Read this at the START of every optimization session and re-read when stuck.

---

## 1. The Micro-to-E2E Gap (most important lesson)

**Micro speedup does not mean E2E speedup.** Real measurements on AMD RDNA3:

| Kernel | Micro | E2E | Root cause |
|--------|-------|-----|------------|
| GEMM with autotune | 1.1x | 0.43x | Each new (M,N,K) triggers 100-iteration sweep at serving time |
| Shape-specific fast path | 1.8x | 0.94x | 60% coverage × 1.8x + 40% fallback overhead = net loss |
| Fused RMSNorm | 2.1x | 0.97x | Not on critical path at serving concurrency |

**Always run `serving-test` before accepting any kernel. If it fails, fix it.**

---

## 2. Autotune in Serving: The #1 Failure Mode

`@triton.autotune` runs 10+ configs × 100 iterations per unique (M,N,K) shape.
In vLLM serving with dynamic batching, M changes every iteration.
Each new batch size triggers the full sweep. On RDNA3: 50-200ms overhead per new M.

**Options (pick one):**
1. **Pre-warm**: Before serving, call the kernel with all expected M values (1..max_batch) to populate the autotune cache.
2. **Fixed config**: Replace `@triton.autotune` with a single best config discovered during Phase 4 micro-benchmarking.
3. **Shape-bucketed dispatch**: Pre-select config for M ranges (M=1: config_A, M<=16: config_B, M>16: config_C).

**Preferred for AMD serving**: option 2 or 3. `@triton.heuristics` for compile-time selection.

```python
# Good for serving — no runtime benchmarking
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
    'BLOCK_M': lambda args: 128 if args['M'] >= 64 else 32,
})
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    ...
```

---

## 3. `tl.assume(stride > 0)` — Always Apply on AMD

Tells the compiler the stride is positive → enables wider vector loads.
Impact: 5-15% on AMD GPUs. Zero cost if not beneficial.

```python
@triton.jit
def kernel(A, stride_am, stride_ak, ...):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    # ... rest of kernel
```

Apply to ALL stride arguments in EVERY kernel.

---

## 4. GROUP_SIZE_M for L2 Cache Locality

Controls how thread blocks are grouped along M for L2 tile reuse.

```python
# Grouped PID mapping (standard Triton GEMM pattern)
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

Sweep values: GROUP_SIZE_M ∈ {1, 2, 4, 8, 16}.
- Smaller → better for tall matrices (M >> N)
- Larger → better for wide matrices (N >> M)
- Start with 8 for square matrices.

---

## 5. Split-K for Large K, Small M×N

When K > 4096 and grid is small (M and N are small), split K across CTAs.
Increases parallelism by allowing multiple CTAs to collaborate on one output tile.

**AMD note**: Use fp32 for Split-K atomics. `tl.atomic_add` in bf16 may be inaccurate on some ROCm versions.

```python
# 2-pass: compute partials, then reduce
pid_k = tl.program_id(1)   # K-split index
k_per_split = tl.cdiv(K, NUM_KSPLIT)
k_start = pid_k * k_per_split
k_end   = min(k_start + k_per_split, K)
# ... accumulate in fp32, write partial to intermediate buffer
# ... second pass: sum partials
```

---

## 6. fp32 Accumulator (mandatory for bf16/fp16 inputs)

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# ... accumulate in fp32 ...
c = acc.to(C.dtype.element_ty)  # cast back to bf16/fp16 for output
tl.store(c_ptrs, c, mask=mask)
```

Never accumulate in bf16 directly — rounding errors will fail correctness tests.

---

## 7. Cache Modifiers

```python
a = tl.load(a_ptrs, cache_modifier=".ca")   # Cache All: good for reused data (matrix A in GEMM)
b = tl.load(b_ptrs, cache_modifier=".cg")   # Cache Global: good for streaming data (matrix B in GEMM)
```

On RDNA3 (Infinity Cache): `.ca` for the A matrix benefits most from cache reuse.

---

## 8. WMMA (RDNA3) / MFMA (CDNA3)

For matrix multiply, use Triton's `tl.dot` which compiles to WMMA/MFMA automatically.

```python
acc += tl.dot(a_tile, b_tile, allow_tf32=False)
```

- RDNA3 (Wave32): WMMA 16×16 tiles
- CDNA3 MI300X (Wave64): MFMA 16×16 and 32×32 tiles — prefer BLOCK_M=BLOCK_N=128

---

## 9. PID Swizzling for CDNA3 Multi-XCD

MI300X has 8 XCDs. Adjacent CTA IDs can map to different XCDs, causing cache misses.
Remap PIDs to keep adjacent CTAs on the same XCD:

```python
def remap_xcd(pid, num_xcd=8):
    return (pid % num_xcd) * (tl.num_programs(0) // num_xcd) + (pid // num_xcd)
```

Only needed for CDNA3 (gfx940/gfx942). Not needed for RDNA3.

---

## 10. GPU Architecture Quick Reference

| Arch | GPU | Wave | Matrix | Memory | Key Triton hint |
|------|-----|------|--------|--------|-----------------|
| gfx1100 (RDNA3) | R7900 XTX | 32 | WMMA | GDDR6 + 96MB Infinity Cache | `tl.assume`, GROUP_SIZE_M |
| gfx1101 (RDNA3) | RX7900 GRE | 32 | WMMA | GDDR6 + 64MB Infinity Cache | same |
| gfx940 (CDNA3) | MI300X | 64 | MFMA | HBM3 | XCD swizzle, Split-K |
| gfx942 (CDNA3) | MI355X | 64 | MFMA | HBM3e | same |

---

## 11. Debugging Guide

**Correctness failure (torch.allclose fails)**:
1. Check accumulator dtype — must be fp32 for bf16/fp16 inputs
2. Check boundary mask — add `mask = offs_k < K` for non-multiple-of-BLOCK_K
3. Check output cast — `c.to(C.dtype.element_ty)` before store
4. Try smaller tolerance first: `atol=1e-1` to isolate the bug

**GPU memory fault (SIGABRT / segfault)**:
1. Tile too large → reduce BLOCK_M, BLOCK_N
2. LDS overflow → reduce BLOCK_K (RDNA3 limit: 64KB LDS per CU)
3. Register spill → reduce num_warps or tile size

**Performance regression vs PyTorch reference**:
1. Check grid size — is it smaller than CU count? (RDNA3: 96 CUs for gfx1100)
2. Check autotune cache — might be selecting wrong config
3. Measure launch overhead: is it compute-bound or launch-overhead-bound?
4. Run `serving-test` — autotune regression may not show in fixed-shape benchmarks

**Triton JIT constraint**:
`@triton.jit` MUST be in a `.py` file. Never use `python3 -c "..."` inline.
The JIT compiler reads the source file; inline code has no file to read.

---

## 12. Optimization Priority Checklist

Use profiling data, not this list, to decide order. But as a starting framework:

1. `tl.assume(stride > 0)` — always, free gain
2. fp32 accumulator — correctness + numerical stability
3. GROUP_SIZE_M sweep — L2 cache reuse
4. BLOCK_M / BLOCK_N tuning — occupancy vs register pressure
5. `@triton.heuristics` instead of `@triton.autotune` — serving safety
6. Cache modifiers (.ca/.cg) — memory-bound kernels
7. Split-K — large K with small M,N
8. Operation fusion — eliminate memory round-trips
9. PID swizzling — CDNA3 only, multi-XCD
10. WMMA/MFMA intrinsics — handled by tl.dot automatically

**Stop when**: consecutive rejections ≥ max_consecutive_rejections, OR total attempts ≥ max_attempts, OR rocprof shows roofline efficiency > 85%.
