# Phase 4: Kernel Optimization

**Phase name**: `optimize`

## Objective
Optimize inference performance using two complementary methods, in order:

1. **Step 0: TunableOps rocBLAS Tuning** (always run first, fast, no code to write)
2. **Step 1+: Triton Kernel Optimization** (run if TunableOps gain < target, or for specific bottlenecks)

**You are the optimizer.** Use profiling data to decide what to try.

---

## Step 0: TunableOps rocBLAS Offline Tuning (Default First Step)

**Background from real measurement on gfx1100 + Qwen3-8B**:
- vLLM uses `BFloat16_TN` GEMM format (transposed weight matrix = `F.linear` pattern)
- Projections are **fused**: QKV → single GEMM (N = hidden + 2×kv_heads×head_dim), gate+up → single GEMM (N = 2×intermediate)
- rocBLAS algorithms beat hipBLASLt for these small-M shapes (M=batch_size, typically 1–128 during decode)
- Measured gain on Qwen3-8B, conc=64: **+48% output_tps** (+1.48x) from TunableOps alone

**Critical implementation notes**:
- `PYTORCH_TUNABLEOP_FILENAME` env var does **NOT** auto-load a CSV in PyTorch 2.9+. It only controls write destination.
- Loading requires `torch.cuda.tunable.read_file()` Python API called inside each subprocess.
- vLLM spawns EngineCore as a subprocess; use `sitecustomize.py` + `PYTHONPATH` to inject `read_file()`.
- The untuned shapes CSV was already collected in Phase 2 via `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`.

### Step 0a: Offline GEMM Tuning

```bash
CUDA_VISIBLE_DEVICES={{SELECTED_GPU}} python3 {{SCRIPTS_DIR}}/tune_gemm_shapes.py \
    --untuned  "{{RESULTS_DIR}}/untuned_shapes_final.csv" \
    --output   "{{OPTIMIZED_DIR}}/tuned_gemm.csv" \
    --max-iter 50 \
    --max-duration 20
```

This runs ~3-5 minutes. Shows which rocBLAS/hipBLASLt algorithm was selected per shape.

### Step 0b: Create Injection Shim

```bash
python3 {{SCRIPTS_DIR}}/create_inject.py \
    --tuned-csv "{{OPTIMIZED_DIR}}/tuned_gemm.csv" \
    --output-dir "{{OPTIMIZED_DIR}}/pypath"
```

Creates `{{OPTIMIZED_DIR}}/pypath/sitecustomize.py` that calls `read_file()` in every spawned subprocess.

### Step 0c: Benchmark with TunableOps (reuse existing server or restart)

Start vLLM with the injection shim:
```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTHONPATH={{OPTIMIZED_DIR}}/pypath:${PYTHONPATH:-}
```

Then verify injection in logs:
```bash
grep "tunableops_inject" {{OUTPUT_DIR}}/vllm_tuned.log | head -3
# Expected: [tunableops_inject] Loaded N tuned entries from ...
```

Run benchmark (use same script as Phase 2). If output_tps ≥ baseline × 1.10 → **target achieved, skip Triton**.

---

## Step 1: Setup Triton Workspaces (only if TunableOps gain insufficient)

```bash
CUDA_VISIBLE_DEVICES={{SELECTED_GPU}} python3 {{SCRIPTS_DIR}}/kernel_agent.py setup \
    --targets      "{{PROBLEMS_DIR}}/targets.json" \
    --real-shapes  "{{RESULTS_DIR}}/real_shapes.json" \
    --model-config "{{MODEL}}/config.json" \
    --gpu-arch     "{{RESULTS_DIR}}/gpu_arch.json" \
    --output-dir   "{{OPTIMIZED_DIR}}" \
    --max-attempts {{MAX_OPTIMIZATION_ATTEMPTS}} \
    --max-rejections {{MAX_CONSECUTIVE_REJECTIONS}} \
    --knowledge-base "{{SKILL_DIR}}/references/TRITON_KNOWLEDGE.md"
```

After setup, read `{{OPTIMIZED_DIR}}/manifest.json` for:
- Target list with baseline performance
- `dtype` — use this for ALL tensor allocations (do NOT assume bf16)
- `knowledge_base` path — **read this file before starting optimization**
- `max_attempts`, `max_consecutive_rejections`

---

## Step 2: For Each Target — Autonomous Optimization Loop

### 2a. Read current status

```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py status \
    --log-path "{{OPTIMIZED_DIR}}/<ktype>/optimization_log.json"
```

Check:
- `dtype` (actual model dtype — use it for all tensors)
- `gpu_arch` (architecture — select AMD or NVIDIA techniques)
- `baseline` performance per shape
- `baseline_rocprof` HW counters (VGPR, SGPR, LDS, workgroup size)
- `consecutive_rejections` vs `max_consecutive_rejections`
- `total_attempts` vs `max_attempts`
- `knowledge_base` — read it NOW

### 2b. Decide what to optimize

**Read `references/TRITON_KNOWLEDGE.md` before every first attempt and whenever stuck.**

**First attempt**: Write a basic Triton kernel for the operation. Start with:
- Correct functional behavior (correctness test must pass first)
- `tl.assume(stride > 0)` on all strides (5-15% AMD gain, free)
- fp32 accumulator for bf16/fp16 inputs
- `@triton.heuristics` preferred over `@triton.autotune` (see Autotune Warning below)

**After rejections**: Re-read `baseline_rocprof`. Ask: compute-bound or memory-bound?
- High VGPR → reduce tile size or num_warps
- Low occupancy → try persistent kernel or larger tiles
- Memory-bound → cache modifiers (.ca/.cg), GROUP_SIZE_M sweep
- Small grid → Split-K (for large K with small M,N)

**Do NOT follow a fixed checklist order.** Profile data drives the decision.

### 2c. Write the kernel file

**IMPORTANT**: Triton `@triton.jit` kernels MUST be in `.py` files. They CANNOT be in `python3 -c "..."` inline scripts — JIT needs the source file. Always write to a file first.

The file at `{{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py` MUST define:
- `reference(*inputs)` → PyTorch reference (used for correctness comparison)
- `optimized(*inputs)` → your Triton implementation

### 2d. Test correctness

```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py correctness \
    --kernel "{{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py" \
    --shapes '<shapes_json>'
```

**Do NOT proceed to benchmarking if correctness fails.** Fix the kernel. Common causes:
- Accumulator dtype wrong (use fp32 for bf16 inputs, cast output back)
- Boundary mask missing (add `mask = offs < limit`)
- Output dtype not cast (`c.to(C.dtype.element_ty)`)

### 2e. Benchmark

```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py benchmark \
    --kernel "{{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py" \
    --shapes '<shapes_json>'
```

### 2f. Accept or reject

**Accept** if average speedup across real shapes > current best:
```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py accept \
    --kernel   "{{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py" \
    --name     "<short_name>" \
    --desc     "<what was applied>" \
    --results  '<benchmark_json>' \
    --log-path "{{OPTIMIZED_DIR}}/<ktype>/optimization_log.json" \
    --kernel-dir "{{OPTIMIZED_DIR}}/<ktype>"
```

**Reject** if no improvement:
```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py reject \
    --name     "<short_name>" \
    --desc     "<what was tried>" \
    --reason   "<why rejected>" \
    --results  '<benchmark_json>' \
    --log-path "{{OPTIMIZED_DIR}}/<ktype>/optimization_log.json"
```

### 2g. Serving Readiness Test (MANDATORY after each accept)

**⚠ AUTOTUNE WARNING**: `@triton.autotune` causes E2E regression in serving. In vLLM with dynamic batching, each new (M,N,K) combination triggers the full benchmark sweep at serving time. Measured impact: 1.1x micro speedup → 0.43x E2E (2.3x SLOWER).

Before accepting any kernel for integration, test serving readiness:

```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py serving-test \
    --kernel   "{{OPTIMIZED_DIR}}/<ktype>/best_kernel.py" \
    --n <N_dim> --k <K_dim> \
    --m-values "1,2,3,5,8,13,21,34,55,64,128" \
    --iterations 200
```

If verdict is `FAIL_REGRESSION` or `FAIL_SEVERE_REGRESSION`:
- **Do NOT proceed to integration**
- Fix by: replacing `@triton.autotune` with fixed config, or pre-warming the cache, or shape-bucketed dispatch
- Re-run serving test until `PASS`

### 2h. Shape Coverage (required before marking a kernel ready)

Compute what fraction of real call volume the fast path covers:

```bash
python3 << 'PYEOF'
import json

with open("{{RESULTS_DIR}}/real_shapes.json") as f:
    rs = json.load(f)

all_shapes = rs.get("benchmark_shapes", [])
# Load accepted kernel's covered shapes from optimization log
with open("{{OPTIMIZED_DIR}}/<ktype>/optimization_log.json") as f:
    log = json.load(f)

covered_m = set(s[0][0] for s in log.get("benchmark_shapes", []))
total_calls = sum(rs.get("shape_call_counts", {}).get(str(s[0]), 1) for s in all_shapes)
covered_calls = sum(rs.get("shape_call_counts", {}).get(str(s[0]), 1)
                    for s in all_shapes if s[0][0] in covered_m)

coverage_pct = covered_calls / total_calls * 100 if total_calls > 0 else 0
print(f"Shape coverage: {coverage_pct:.1f}%  (covered M values: {sorted(covered_m)})")

if coverage_pct < 50:
    print("WARNING: coverage < 50% — E2E gain may be limited by uncovered shapes")
    print("Consider expanding benchmark_shapes to include more M values")
PYEOF
```

**A kernel with coverage < 50% must have a robust fallback to the original kernel for uncovered shapes.**

### 2i. Stopping Condition

**Stop when ANY of:**
- `consecutive_rejections >= max_consecutive_rejections` (default 3)
- `total_attempts >= max_attempts` (default 8)
- rocprof shows kernel is at roofline efficiency

**Do NOT stop just because a checklist item was tried.** If the last attempt was accepted, keep going.

---

## Step 3: HW Profiling with rocprofv3 (when stuck or every 3 accepts)

```bash
python3 {{SCRIPTS_DIR}}/kernel_agent.py rocprof \
    --kernel "{{OPTIMIZED_DIR}}/<ktype>/best_kernel.py" \
    --shapes '<shapes_json>' \
    --output-dir "{{OPTIMIZED_DIR}}/<ktype>"
```

Interpret the output:
- High VGPR (>128) → register pressure, reduce tile size
- Low `grid_size` relative to CU count → under-utilization, try Split-K
- High `lds_bytes` (>65536 on RDNA3) → LDS pressure, reduce BLOCK_K

---

## Completion

After all targets:

```bash
for dir in {{OPTIMIZED_DIR}}/*/; do
    log="$dir/optimization_log.json"
    [ -f "$log" ] && python3 {{SCRIPTS_DIR}}/kernel_agent.py status --log-path "$log" | \
        python3 -c "import json,sys; s=json.load(sys.stdin); print(f'[{s[\"kernel_type\"]}] best={s[\"best_avg_speedup\"]}x  attempts={s[\"total_attempts\"]}  serving_ready={s.get(\"serving_ready\",\"?\")}' )"
done
```

Update `{{PROGRESS_FILE}}`:
```json
{"phases_completed": ["env","server","bench-profile","analysis","optimize"],
 "details": {"kernels_accepted": "<N>", "kernels_rejected": "<M>"}}
```
