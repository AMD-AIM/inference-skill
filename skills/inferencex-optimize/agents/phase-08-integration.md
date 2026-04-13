# Phase 8: Integration & End-to-End Benchmarking

## Instructions

You are a phase agent responsible for integrating optimized kernels and measuring actual E2E serving throughput. You read exactly 2 files: this document and your handoff at `handoff/to-phase-08.md`.

**Tools**: Shell commands, Docker, Python, file I/O.
**Outputs**: Write `agent-results/phase-08-result.md`. Write plugin to `{OPTIMIZED_DIR}`, comparison to `{RESULTS_DIR}`.
**Sub-agents**: May spawn coder subagents for plugin generation per `agents/coding-agent.md`.
**MANDATORY**: This phase MUST produce real measured data. FORBIDDEN: estimating speedup, copying baseline numbers, or skipping the patched server benchmark.
**SKIP_INTEGRATION**: If the handoff sets `SKIP_INTEGRATION=true`, this phase should NOT have been dispatched — the orchestrator removes it from the phase list. If you find yourself running with this flag set, generate the plugin only (Steps 0-1) and skip the E2E benchmark (Steps 2-7). Report in the result doc that the benchmark was skipped per user request.

## Runbook

### Config Resolution
Before any Docker or benchmark commands, read `{{OUTPUT_DIR}}/results/sweep_configs.json` (or `{{OUTPUT_DIR}}/config.json` when sweep metadata is folded there) and export: `RUNNER`, `IMAGE`, `FRAMEWORK`, `MODEL`, `PRECISION`, `TP`, `EP`, `CONC`, `ISL`, `OSL`, `MAX_MODEL_LEN`, `EXP_NAME`, `BENCHMARK_SCRIPT`, plus GPU selectors (`GPUS`, etc.) required by `run_profile_exec.sh`.

### Measured-data gate
Phase 8 is **not** complete until three artifacts exist: (1) a fresh baseline JSON from the same harness revision, (2) a patched-server benchmark log + JSON captured after plugin injection **or** the standalone `run_e2e_benchmark.sh` output, and (3) a passing `validate_optimization.py` run. Estimating speedups, copying historical throughput numbers without rerunning the server, or skipping the patched benchmark path is explicitly out of scope—abort and document blockers instead.

End-to-end success means **InferenceX Docker containers** exercised the real serving stack (vLLM or SGLang) with optimized kernels or dispatch patches loaded exactly as production would—not microbench-only speedups.

### 0. Verify Winning Kernels
```bash
python3 "{{SCRIPTS_DIR}}/optimize/verify_winning_kernels.py" \
    --problems-dir "{{PROBLEMS_DIR}}" --optimized-dir "{{OPTIMIZED_DIR}}"
```
Exit code **1** means one or more manifest entries lack corresponding optimized artifacts — return to Phase 7 Step 3.5 (patch recovery + `collect_winning_kernels.py`) before continuing. Only **skip** integrating a specific kernel when its measured `speedup <= 1.0x`; missing files are never silently ignored.

### 0b. GPU State Cleanup
```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f sglang 2>/dev/null || true
sleep 5
```

### 1. Generate Plugin
**For vLLM:**
```bash
python3 "{{SCRIPTS_DIR}}/plugin/generate_vllm_plugin.py" --kernel-dir "{{OPTIMIZED_DIR}}"
```

**For SGLang:**
```bash
python3 "{{SCRIPTS_DIR}}/plugin/generate_sglang_plugin.py" --kernel-dir "{{OPTIMIZED_DIR}}"
```

Each generator writes `{{OPTIMIZED_DIR}}/<framework>_plugin/` containing at minimum `__init__.py`, a launcher helper script, and `manifest.json` describing injected entry points.

After generation, open `manifest.json` and confirm every optimized kernel you expect appears with the correct module path, symbol name, and dispatch metadata. If a kernel is missing, rerun Phase 7 for that entry before editing the manifest by hand—manual edits should be a last resort documented in the handoff.

### 1.5. Dispatch-Level Optimization
**Trigger:** `geak_results.json` lists a vendor / CK / HIP / MoE / attention kernel with `speedup > 1.0` **and** the winning change is dispatch-level (description mentions dispatch, backend, BLAS, config, selection, NT layout, etc.), **or** no directly mappable kernels exist yet dispatch tuning would still help.

Dispatch work changes **how** the framework picks a kernel, not only the kernel source file.

| Kernel type | Dispatch mechanism | Patch target (SGLang examples) |
|-------------|--------------------|--------------------------------|
| Dense GEMM (vendor) | BLAS backend selection per shape | `UnquantizedLinearMethod.apply` |
| MoE GEMM (CK/aiter) | Expert routing / backend config | `FusedMoE.forward` or aiter dispatch helpers |
| Attention (CK/aiter/ASM) | Backend selection vs sequence length | `AttentionBackend` / MLA dispatch |
| Normalization (CK/aiter) | Fused vs unfused path selection | `RMSNorm.forward` or aiter dispatch |

- **1.5a. Micro-benchmark:** Author `{{OPTIMIZED_DIR}}/bench_dispatch.py` that exercises each candidate dispatch path across representative shapes from `optimization_manifest.json`. Run inside the same container image to rank per-shape winners.
- **1.5b. Dispatch plugin:** When benchmarks prove a non-default path wins for some shapes, emit a monkey-patch module that intercepts the framework hook (pattern: detect shape → temporarily override config → call original → restore defaults). Reference implementation: `{{TEMPLATES_DIR}}/dispatch_plugin_example.py` (SGLang GEMM).
- **1.5c. Standalone launcher:** Add `{{OPTIMIZED_DIR}}/launch_patched.py` that imports the plugin **before** importing the serving stack so patches register prior to module singletons initializing.
- **1.5d. Standalone E2E benchmark:** Add `{{OPTIMIZED_DIR}}/run_e2e_benchmark.sh` when script injection (Steps 4–5) is impractical. If this script exists, **skip** Steps 4–5, run the standalone benchmark, collect artifacts into `{{RESULTS_DIR}}/`, then jump to Step 6.

### 2. Retrieve Baseline Data
Reuse Phase 2/3 artifacts stored under `{{RESULTS_DIR}}/`. Locate raw benchmark JSON files while **excluding** summary-only or config-only JSONs. Prefer the **lowest concurrency** run whose ISL/OSL/TP/EP/precision matches the profiling configuration so apples-to-apples comparisons remain valid.

Record the exact baseline filename in `agent-results/phase-08-result.md` so validators can trace which run was used. When multiple baselines qualify, prefer the file referenced by Phase 3’s `benchmark_summary.json` if that index exists.

### 3. Start Optimized Container
**For SGLang:**
```bash
bash "{{SCRIPTS_DIR}}/container/start_profile_container.sh" \
    --name "inferencex-optimized-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{RESULTS_DIR}}" --mode optimize \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --env "PYTHONPATH=/sgl-workspace/aiter:/workspace/optimized" \
    --env "SGLANG_USE_AITER=1" --env "ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
```

**For vLLM:**
```bash
bash "{{SCRIPTS_DIR}}/container/start_profile_container.sh" \
    --name "inferencex-optimized-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{RESULTS_DIR}}" --mode optimize \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --env "PYTHONPATH=/workspace/optimized" --env "ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
```

Use the block that matches `$FRAMEWORK`. `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` applies to **both** stacks: it tunes RCCL all-reduce quantization for communication bandwidth and is independent of model weight precision.

SGLang’s `PYTHONPATH` prepends `/sgl-workspace/aiter` so packaged CK kernels resolve before falling back to upstream sources; vLLM only needs `/workspace/optimized` unless the handoff adds extra vendor paths—mirror whatever Phase 4 used for profiling parity.

Export `CONTAINER_NAME` immediately after `start_profile_container.sh` returns (match the `--name` argument, typically `inferencex-optimized-{{CONFIG_KEY}}`). All subsequent `docker exec`, `docker cp`, and `run_profile_exec.sh` calls assume this variable is set.

### 4. Inject Plugin
Restore a clean benchmark script before mutating it so repeated integrations do not stack edits:

```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" 2>/dev/null || true
docker cp "{{SCRIPTS_DIR}}/plugin/inject_plugin.py" "$CONTAINER_NAME:/tmp/"
docker exec "$CONTAINER_NAME" python3 /tmp/inject_plugin.py \
    --framework "$FRAMEWORK" --target "/workspace/$BENCHMARK_SCRIPT"
```

`inject_plugin.py` edits the repo-relative benchmark entrypoint as seen **inside** the container (`/workspace/...` mirrors `{{REPO_DIR}}`). Confirm the script path matches the checked-out tree before launching the server.

### 5. Run Optimized Benchmark
Double-check the following immediately before `run_profile_exec.sh` fires:

- Container still running (`docker ps --filter name=$CONTAINER_NAME`).
- Plugin files visible at `/workspace/optimized` inside the container (`docker exec ... ls /workspace/optimized`).
- Benchmark script inside the container contains the injection sentinel (quick `grep -n plugin` on `/workspace/$BENCHMARK_SCRIPT` via `docker exec`).
- Host-side `{{RESULTS_DIR}}/` has enough disk for full JSON + log captures.

```bash
bash "{{SCRIPTS_DIR}}/container/run_profile_exec.sh" \
    --container "$CONTAINER_NAME" \
    --benchmark-script "$BENCHMARK_SCRIPT" \
    --model "$MODEL" --tp "$TP" --ep "$EP" --conc "$CONC" \
    --isl "$ISL" --osl "$OSL" --max-model-len "$MAX_MODEL_LEN" \
    --precision "$PRECISION" --framework "$FRAMEWORK" --exp-name "$EXP_NAME" \
    --result-filename "optimized_${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}" \
    --repo-dir "{{REPO_DIR}}" --gpus "{{GPUS}}"
```

Mirror Phase 2’s benchmark mode (e.g., compiled-graph path) unless the handoff documents an intentional deviation — mismatched modes invalidate throughput comparisons. Stream container logs live and keep heartbeats so long runs remain attributable.

Collect JSON/text outputs from the repo root and any `results/` subfolder into `{{RESULTS_DIR}}/`.

The `--result-filename` value intentionally prefixes `optimized_` plus the TP/EP/concurrency dimensions—keep that convention so `validate_optimization.py` can glob deterministic pairs (`baseline_*.json` vs `optimized_*.json`) without extra flags.

### 6. Validate Results
```bash
python3 "{{SCRIPTS_DIR}}/report/validate_optimization.py" --results-dir "{{RESULTS_DIR}}"
```

`validate_optimization.py` writes `{{RESULTS_DIR}}/optimization_comparison.json` containing baseline vs optimized throughput, computed speedup, and validation flags — attach this file to the Phase 9 bundle.

Keep the baseline + optimized JSON filenames referenced in the validator logs; if validation fails, capture stderr and re-run the benchmark with smaller `CONC` before declaring the integration unsuccessful.

When reporting externally, quote **total token throughput** fields exactly as emitted by the benchmark JSON (do not normalize to per-GPU rates unless the baseline file already does). If multiple JSON files match the optimized filename pattern, prefer the newest mtime and archive the others under `{{RESULTS_DIR}}/archive/` to avoid validator ambiguity.

Spot-check `optimization_comparison.json` manually: `speedup` should be `optimized / baseline` within floating-point tolerance, and `validated` should be `true` before you tell downstream consumers the run succeeded.

### 7. Clean Up
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
docker stop "$CONTAINER_NAME" 2>/dev/null; docker rm "$CONTAINER_NAME" 2>/dev/null
```

### Completion
Write `agent-results/phase-08-result.md` with baseline_throughput, optimized_throughput, speedup, plugin_type (for quality check: include "validation passed" or "validation failed" string).

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
