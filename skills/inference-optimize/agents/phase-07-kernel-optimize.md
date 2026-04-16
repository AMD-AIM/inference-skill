# Phase 7: Kernel Optimization

## Instructions

You are a phase agent responsible for optimizing bottleneck GPU kernels using GEAK or manual methods. You read exactly 2 files: this document and your handoff at `handoff/to-phase-07.md`.

**Tools**: Shell commands, Docker, Python, file I/O.
**Outputs**: Write `agent-results/phase-07-result.md`. Write optimized kernels to `{{OPTIMIZED_DIR}}`, results to `{{PROBLEMS_DIR}}/geak_results.json`.
**Sub-agents**: Spawn coder subagents for kernel writing tasks per `agents/coding-agent.md`.
**Errors**: Track per-kernel attempts (max 5). Report partial results if some kernels fail.

## Runbook

### Progress Reporting
This phase can run up to 90 minutes. Print a one-line status update before each major step:
- Before GEAK mode resolution: `[phase-07] Resolving GEAK mode...`
- Before each kernel target: `[phase-07] Optimizing kernel N/M: <name> (<pct>% GPU time)...`
- After each kernel: `[phase-07] Kernel <name>: speedup=<X>x (attempt <i>/<max>)`
- Before collecting results: `[phase-07] Collecting winning kernels...`

### Prerequisites
- Problem files and metadata from Phase 6 under `{{PROBLEMS_DIR}}/`
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (with kernel types and profiling metadata)
- `{{ENV_INFO_FILE}}` from Phase 0 (GEAK / GPU / environment facts)
- Docker image with PyTorch + Triton — use the **same** `IMAGE` as Phases 2 and 4 unless the handoff explicitly changes it

### Config Resolution
Read `{{OUTPUT_DIR}}/results/sweep_configs.json` (fallback: `{{OUTPUT_DIR}}/config.json` if that is how this run stores globals) and export shell variables used later: `RUNNER`, `IMAGE`, `FRAMEWORK`, `MODEL`, `PRECISION`, `TP`, `EP` (and any other keys the container scripts expect from the handoff).

### 0. Resolve GEAK Mode
```bash
python3 "{{SCRIPTS_DIR}}/optimize/resolve_geak_mode.py" \
    --user-mode "{{GEAK_MODE}}" --env-info "{{ENV_INFO_FILE}}"
```
Capture `EFFECTIVE_GEAK_MODE`: `full`, `triton_only`, or `manual`.

Mode semantics:
- `full` — process both `simple` and `kernel-url` manifest entries.
- `triton_only` — process **only** `simple` entries (skip `kernel-url` vendor/C++ paths).
- `manual` — skip GEAK automation entirely and follow the **Manual Fallback** section.

### 1. Load Manifest
```bash
python3 "{{SCRIPTS_DIR}}/optimize/load_optimization_manifest.py" \
    --manifest "{{PROBLEMS_DIR}}/optimization_manifest.json" \
    --geak-mode "$EFFECTIVE_GEAK_MODE" --optimize-scope "{{OPTIMIZE_SCOPE}}"
```

The loader prints the prioritized kernel queue (highest `priority_score` first) grouped by GEAK mode—follow that ordering unless the handoff explicitly reprioritizes hot spots.

### 2. Start Optimization Container
```bash
bash "{{SCRIPTS_DIR}}/container/start_profile_container.sh" \
    --name "inference-kernel-opt-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{PROBLEMS_DIR}}" --mode optimize \
    --mount "{{PROBLEMS_DIR}}:/workspace/problems" \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --mount "{{SCRIPTS_DIR}}:/workspace/scripts" \
    --mount "{{GEAK_DIR}}:/workspace/geak" \
    --env "AMD_LLM_API_KEY=${AMD_LLM_API_KEY:-}" \
    --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
    --env "LLM_GATEWAY_KEY=${LLM_GATEWAY_KEY:-}"
```

**Container bind mounts (reference):**

| Host path | Container path | Purpose |
|-----------|------------------|---------|
| `{{PROBLEMS_DIR}}` | `/workspace/problems` | Inputs, GEAK workspaces, `optimization_logs/` |
| `{{OPTIMIZED_DIR}}` | `/workspace/optimized` | Winning kernels promoted for Phase 8 |
| `{{SCRIPTS_DIR}}` | `/workspace/scripts` | `kernel_finalize.py`, runners, utilities |
| `{{GEAK_DIR}}` | `/workspace/geak` | GEAK configs and tooling |

### 3. Detect Container Environment
Inside the running container, verify the toolchain before launching GEAK:

- PyTorch imports and can allocate on the expected GPU (`torch.cuda.is_available()`, device name matches profiling notes).
- Triton imports when any manifest entry still relies on Triton baselines.
- GEAK’s `mini` driver responds to `--help` (or the configured launcher exists on `PATH`) and API keys from the `docker run` env are visible to the process.
- Run `python3 "{{SCRIPTS_DIR}}/env/select_gpus.py"` (from the host **or** a one-off `docker exec` with the same GPU flags) so `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` maps to the intended devices—GEAK inherits those constraints.

Document any mismatch between detected arch and `{{OUTPUT_DIR}}/results/gpu_arch.json` before burning attempts.

### 3a. C++/Vendor Kernel Optimization (kernel-url mode)
For entries with `geak_mode: kernel-url`, process in `profiling_pct` descending order.

Source tracing required before GEAK — confirm the on-disk source matches the profiled kernel name (substring / symbol match):
- Vendor GEMM (`Cijk_*`): Tensile-generated, dispatched by hipBLASLt
- CK/aiter kernels: `aiter/csrc/` package

**Operator-specific guidance:**

| Type | Source location | Compute spec |
|------|-----------------|---------------|
| MoE GEMM | `aiter/csrc/ck_gemm_moe_2stages_codegen/` | Varies by precision |
| FP4 GEMM | `aiter/aiter/ops/triton/gemm_afp4wfp4.py` | `matrix_fp4` |
| Attention | `aiter/csrc/mla/` | `matrix_bf16` |
| Normalization | `aiter/csrc/` | Typically memory-bound |

For each kernel:
1. Prepare isolated workspace: copy source, `git init && git add -A && git commit -m init`
2. Build task description with roofline data: include `compute_spec`, `tflops_s`, `roofline_efficiency`, and `peak_tflops` when available from profiling artifacts
3. Launch GEAK: `mini -m claude-opus-4.6 --config mini_kernel.yaml --repo /workspace/${NAME}_opt --gpu-ids $GPU_IDS --yolo -t '${TASK_DESC}' -o traj_${NAME}.json` (where `GPU_IDS` is read from `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` env var)
4. **Vendor GEMM (no external source tree):** copy **only** the problem `.py` into the isolated workspace — avoid copying multi-megabyte generated trees that produce unusable mega-patches
5. If the winning build measurably speeds up the kernel (`speedup > 1.0x`), install it back into the environment the benchmark uses via `AITER_REBUILD=1` (rebuild aiter) or `pip install -e .` from the component root, matching how the stack is consumed in this image

### 3b. Triton Optimization (simple mode)
**Scope:** Simple mode applies **only** to kernels that were **originally authored in Triton** (`kernel_type: triton`). All other kernel types must flow through `kernel-url` / vendor workflows — do not route CK/HIP/vendor GEMMs through the Triton simple template.

For each kernel:
1. Build task description with shapes and roofline data
2. `git init && git add -A && git commit -m init` in problems dir
3. Launch GEAK: `mini -m claude-opus-4.6 --config geak.yaml --gpu-ids 0 --yolo -t '${TASK_DESC}' -o traj_${NAME}.json`

### 3c. Mandatory Attempt Rule

**A target MUST NOT be classified as `parity_or_blocked` unless at least one GEAK or manual optimization attempt has been made and a measured speedup recorded.** Specifically:

1. A target without a trajectory file (`traj_*.json`) or a manual attempt entry in `manual_attempts.md` has **zero attempts** and cannot be classified as `parity_or_blocked`.
2. Targets with zero attempts MUST be classified as `not_attempted`. A `not_attempted` classification triggers a monitor FAIL verdict — the phase will be retried with explicit instructions to attempt these targets.
3. Valid `parity_or_blocked` requires: at least 1 attempt + measured speedup <= 1.0 + a documented reason (`true_kernel_parity`, `framework_limit`, `vendor_binary_only`, etc.).
4. High-priority targets (top 3 by `profiling_pct`) require at least **two** attempts before `parity_or_blocked` is allowed — one GEAK and one manual/alternative approach.

This rule exists because skipping high-value targets (e.g., MoE GEMM at 83% of GPU time) without even attempting optimization is the single biggest source of value loss in the pipeline.

### 3.5. Patch Recovery (CRITICAL)
Run after EVERY GEAK attempt. GEAK's `[SelectPatch]` agent frequently fails.

GEAK typically writes trajectories plus `optimization_logs/<kernel_stem>_*/patch_<n>_test.txt` siblings. Treat every `patch_*_test.txt` as a candidate scoreboard even when the UI claims failure—many winning diffs never receive automatic promotion.

1. Check GEAK log for `[SelectPatch]` success
2. If selection failed: recursively scan `optimization_logs/${NAME}_*/patch_*_test.txt`. Each file usually records one candidate patch attempt.
3. Parse lines containing `RESULT_JSON:` (JSON payload with timing / speedup fields) **or** `GEAK_RESULT_LATENCY_MS=` (scalar latency). Prefer the candidate with the best measured speedup vs baseline; if only latency is present, rank by lowest latency consistent with the harness.
4. Map the winning text file back to its sibling `.diff` in the same patch directory; apply that diff to the workspace, then re-run `kernel_test_runner.py` to confirm accuracy + timing.
5. Copy the verified winning `*_opt.py` into `{{OPTIMIZED_DIR}}/` immediately so Phase 8 sees stable artifacts even if GEAK later overwrites the workspace.

### 4. Iterate and Finalize
Allow up to **five** GEAK/manual attempts per kernel. After **every** attempt—successful or not—run Step 3.5 so no winning patch is stranded in `optimization_logs/`. Once a candidate passes accuracy + performance gates, run the finalize script to normalize formatting, metadata, and file naming expected by downstream collectors:

```bash
docker exec $CONTAINER_NAME python3 /workspace/scripts/optimize/kernel_finalize.py \
    --target /workspace/problems/${NAME}_opt.py
```

Repeat finalize only on the file that actually ships to `{{OPTIMIZED_DIR}}/` to avoid clobbering exploratory drafts.

### 5. Collect Winning Kernels
```bash
docker exec $CONTAINER_NAME python3 /workspace/scripts/optimize/collect_winning_kernels.py \
    --problems-dir /workspace/problems --optimized-dir /workspace/optimized
```

This aggregates per-kernel outcomes into `{{PROBLEMS_DIR}}/geak_results.json` and ensures winning `*_opt.py` files land under `{{OPTIMIZED_DIR}}/` for integration.

### 6. Clean Up
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

### Manual Fallback (GEAK_MODE=manual)
When GEAK is disabled or unusable, optimize manually while preserving the same verification loop:

Always drive manual attempts through the same `kernel_test_runner.py` entrypoint GEAK uses so latency numbers remain comparable across kernels. Keep per-kernel notes under `{{PROBLEMS_DIR}}/manual_attempts.md` (create if missing) so Phase 9 can cite what changed when no `traj_*.json` exists.

- **HIP/CK kernels:** Log a clear warning that in-place vendor kernel optimization normally expects GEAK; if policy allows, attempt a **Triton replacement** that matches tensor shapes and numerics, then benchmark.
- **Vendor kernels:** Prefer small **compatibility / dispatch fixes** when source is available; otherwise fall back to a **Triton replacement** baseline competitive with `torch` ops.
- **Triton kernels:** Target AMD `wave_size=64`, explore `BLOCK_SIZE` 16–256, `num_warps` 4–8, and wrap launches in `@triton.autotune` with 10–20 configs covering memory vs compute trade-offs.
- For every attempt: test with `kernel_test_runner.py`, iterate up to **5** rounds, then run `kernel_finalize.py` on the surviving candidate just like the automated path.

### Completion
Write `agent-results/phase-07-result.md` with kernels_attempted, kernels_improved, geak_mode used, per-kernel speedups.

Include these scalar fields in `## Key Findings` for monitor consumption:
- `compiled_count`: integer count of compiled kernels
- `best_speedup`: float (best kernel-level speedup achieved)
- `winning_kernel_count`: integer count of kernels with speedup > 1.0
- `optimization_coverage_status`: complete | partial | none
- `expected_improvement_status`: improvable | parity_or_blocked | not_attempted (summarize across hot targets; `not_attempted` triggers monitor FAIL)

For targets that are inherently unimprovable (at parity with baseline, or blocked by framework/vendor limits), classify them as `parity_or_blocked` **ONLY after at least one GEAK or manual attempt has been made and the measured speedup is <= 1.0**. Targets with zero attempts MUST be classified as `not_attempted` — this triggers a monitor FAIL so the phase retries with those targets attempted. Document the reason per target in `geak_results.json` (e.g., `true_kernel_parity`, `framework_limit`). This distinction prevents the monitor from triggering endless retries on targets that genuinely cannot be improved, while ensuring high-value targets are never skipped without trying.

Report the measured `best_speedup` honestly even when it is `<= 1.0`. The monitor uses `expected_improvement_status` plus the structured blocker reasons to decide WARN versus FAIL; do not inflate or coerce the scalar just to satisfy a gate.

If the handoff contains a `## Root Cause Analysis` section (from a prior failed attempt), read the RCA artifact path and adjust your approach based on the retry recommendation. Use `next_attempt_mode` from the RCA to decide between GEAK retry, manual fallback, or coding-agent help. Targets classified as `true_kernel_parity` in the RCA's `blocker_classifications` should be skipped on retry.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
