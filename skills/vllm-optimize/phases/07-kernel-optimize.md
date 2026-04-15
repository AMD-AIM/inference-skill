# Phase 7: Kernel Optimization {{SKIP_LABEL}}

## Objective
Optimize bottleneck GPU kernels via an autonomous auto-research loop. The agent observes hardware profiling data, writes Triton kernels, tests them, and iterates until no further improvement is found.

## Architecture

There are two components:
1. **Scaffold script** (`kernel_optimize_agent.py`): provides tool commands (`setup`, `benchmark`, `correctness`, `rocprof`, `accept`, `reject`, `status`).
2. **This phase document**: instructs the executing agent (you) how to use those tools in an autonomous optimization loop.

**You are the optimizer.** You decide what to try based on profiling data and Triton optimization knowledge. There are no pre-defined templates.

## Step 1: Setup workspace

Run the scaffold to create per-kernel workspaces, measure baselines, and collect initial HW counters:

```bash
CUDA_VISIBLE_DEVICES={{SELECTED_GPU}} python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py setup \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --model-config "{{MODEL}}/config.json" \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json" \
    --output-dir "{{OPTIMIZED_DIR}}" \
    --threshold {{OPTIMIZE_PRIORITY_THRESHOLD}} \
    --max-attempts {{MAX_OPTIMIZATION_ATTEMPTS}} \
    --max-rejections {{MAX_CONSECUTIVE_REJECTIONS}}
```

After setup, read `{{OPTIMIZED_DIR}}/manifest.json` to see:
- Targets and their baseline performance
- Model dtype (do NOT assume bf16 — use whatever the model config says)
- `knowledge_base` path — **read this file now** before starting optimization
- `max_attempts` and `max_consecutive_rejections` — the stopping criteria

## Step 2: For each kernel target, run the optimization loop

For each target in `manifest.json`, execute the auto-research loop described below.

### 2a. Read the current state

```bash
python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py status --log-path {{OPTIMIZED_DIR}}/<ktype>/optimization_log.json
```

This tells you:
- `dtype` — the model's actual dtype (bf16, fp16, or fp32). **Use this for all tensor allocations.**
- `gpu_arch` — the GPU architecture (e.g., gfx1100). Use this to select appropriate optimizations.
- Baseline performance (PyTorch reference, per shape)
- `baseline_rocprof` — HW counters: VGPR, SGPR, LDS, workgroup/grid sizes
- How many attempts so far, accepted/rejected counts
- `consecutive_rejections` and limits (`max_attempts`, `max_consecutive_rejections`)
- Current best speedup and the optimization path that achieved it
- `knowledge_base` — path to the optimization knowledge file. **Read it now if you haven't.**
- `steps_summary` — history of all previous attempts with descriptions and reasons
- `benchmark_shapes` — the exact shapes to use for testing

### 2b. Analyze and decide what to optimize

**First: read the knowledge base** at the path given in `status.knowledge_base`. This file contains AMD-specific Triton patterns, architecture notes, and an optimization checklist. Re-read it whenever you're stuck.

Based on the current state, reason about what the bottleneck is:

**If this is the first attempt** (no prior steps):
- Start simple: write a basic Triton kernel that does the same operation.
- Use `@triton.autotune` with a few safe configs suited to the GPU arch.
- Use the model's actual dtype (from `status.dtype`), NOT hardcoded bf16.
- For GEMM: use grouped PID mapping, fp32 accumulator.
- For fused ops: fuse the operations into one kernel.

**If previous attempts succeeded** (optimization path is growing):
- Read the current best kernel file (`status.best_kernel`) to see what's already been done.
- Check the `steps_summary` to see what was tried and what worked/failed.
- Consult the knowledge base for the next technique to try.
- Apply ONE optimization at a time so you can isolate its effect.

**If previous attempts failed** (consecutive rejections > 0):
- Re-read the rocprof HW data from `status.baseline_rocprof`. Analyze: Is it compute-bound or memory-bound? High VGPR pressure? Bad grid size?
- Run `rocprof` on the current best kernel to get fresh HW data.
- Try a fundamentally different approach — don't keep tweaking the same thing.
- Re-read the knowledge base's "Debugging Guide" section.

### 2c. Write the kernel

Create a new `.py` file at `{{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py`. The file MUST define:
- `reference(*inputs)` → the PyTorch reference implementation
- `optimized(*inputs)` → your Triton implementation

### 2d. Test correctness

```bash
python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py correctness \
    --kernel {{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py \
    --shapes '<shapes_json from manifest>'
```

If any shape fails correctness, fix the kernel and re-test. Do NOT proceed to benchmarking with an incorrect kernel.

### 2e. Benchmark

```bash
python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py benchmark \
    --kernel {{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py \
    --shapes '<shapes_json>'
```

This returns per-shape speedup vs the PyTorch reference.

### 2f. Accept or reject

Compare benchmark results against the current best (from `status`). 

**Accept** if the new kernel improves average speedup across shapes:
```bash
python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py accept \
    --kernel {{OPTIMIZED_DIR}}/<ktype>/attempt_<N>.py \
    --name "<short_name>" \
    --description "<what optimization was applied>" \
    --benchmark-results '<benchmark_json>' \
    --log-path {{OPTIMIZED_DIR}}/<ktype>/optimization_log.json \
    --kernel-dir {{OPTIMIZED_DIR}}/<ktype>
```

**Reject** if it regresses or doesn't improve:
```bash
python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py reject \
    --name "<short_name>" \
    --description "<what was tried>" \
    --reason "<why it was rejected>" \
    --benchmark-results '<benchmark_json>' \
    --log-path {{OPTIMIZED_DIR}}/<ktype>/optimization_log.json
```

### 2g. Profile with rocprofv3 (periodically)

After every 2-3 accepted optimizations, or when stuck, run rocprofv3 to get fresh HW data:

```bash
python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py rocprof \
    --kernel {{OPTIMIZED_DIR}}/<ktype>/best_kernel.py \
    --shapes '<shapes_json>' \
    --output-dir {{OPTIMIZED_DIR}}/<ktype>
```

Use the VGPR/SGPR/LDS/grid data to inform your next optimization decision.

### 2h. Stopping condition

Read `max_consecutive_rejections` and `max_attempts` from the status output. These are user-configurable (set during setup, defaults: 3 rejections, 8 attempts).

**Stop when:**
- `consecutive_rejections >= max_consecutive_rejections` — you've tried N different ideas with no improvement
- OR `total_attempts >= max_attempts` — hard cap reached
- OR you determine the kernel is at the roofline (from rocprof data)

**Do NOT stop just because all strategies in some checklist have been tried.** If the last attempt was accepted (improvement found), keep going — there may be more to gain.

## Optimization Knowledge Base

**Do NOT rely on inline knowledge here.** Read the external knowledge base file at the path given by `manifest.knowledge_base` or `status.knowledge_base`.

The knowledge base file (`references/TRITON_OPTIMIZATION_KNOWLEDGE.md`) contains:
- AMD-specific Triton patterns (autotune, heuristics, stride assumptions, cache modifiers, split-K, persistent kernels, fused ops)
- GPU architecture notes (RDNA3 vs CDNA vs NVIDIA — wave sizes, tile limits, XCD, L2 cache)
- Debugging guide (correctness, GPU faults, performance regressions)
- Optimization priority checklist
- When to stop optimizing

## Completion

After all targets are optimized, print a summary and update progress.json:

```bash
for ktype_dir in {{OPTIMIZED_DIR}}/*/; do
    log="$ktype_dir/optimization_log.json"
    [ -f "$log" ] && python3 {{SCRIPTS_DIR}}/kernel_optimize_agent.py status --log-path "$log"
done
```

Update progress.json:
```json
{
  "phase": "kernel-optimize",
  "phases_completed": [..., "kernel-optimize"],
  "current_step": "kernel optimization complete"
}
```
