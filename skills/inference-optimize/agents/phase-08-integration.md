# Phase 8: Integration & End-to-End Benchmarking

## Instructions

You are a phase agent responsible for integrating optimized kernels and measuring actual E2E serving throughput. You read exactly 2 files: this document and your handoff at `handoff/to-phase-08.md`.

**Tools**: Shell commands, Docker, Python, file I/O.
**Outputs**: Write `agent-results/phase-08-result.md`. Write plugin to `{{OPTIMIZED_DIR}}`, comparison to `{{RESULTS_DIR}}`.
**Sub-agents**: May spawn coder subagents for plugin generation per `agents/coding-agent.md`.
**MANDATORY**: This phase MUST produce real measured data. FORBIDDEN: estimating speedup, copying baseline numbers, or skipping the patched server benchmark.
**SKIP_INTEGRATION**: If the handoff sets `SKIP_INTEGRATION=true`, this phase should NOT have been dispatched — the orchestrator removes it from the phase list. If you find yourself running with this flag set, generate the plugin only (Steps 0-1) and skip the E2E benchmark (Steps 2-7). Report in the result doc that the benchmark was skipped per user request.

## Runbook

### Config Resolution
Before any Docker or benchmark commands, read `{{OUTPUT_DIR}}/results/sweep_configs.json` (or `{{OUTPUT_DIR}}/config.json` when sweep metadata is folded there) and export: `RUNNER`, `IMAGE`, `FRAMEWORK`, `MODEL`, `PRECISION`, `TP`, `EP`, `CONC`, `ISL`, `OSL`, `MAX_MODEL_LEN`, `EXP_NAME`, `BENCHMARK_SCRIPT`, plus GPU selectors (`GPUS`, etc.) required by `run_profile_exec.sh`.

### Measured-data gate
Phase 8 is **not** complete until three artifacts exist: (1) a fresh baseline JSON from the same harness revision, (2) a patched-server benchmark log + JSON captured after plugin injection **or** the standalone `run_e2e_benchmark.sh` output, and (3) a passing `validate_optimization.py` run. Estimating speedups, copying historical throughput numbers without rerunning the server, or skipping the patched benchmark path is explicitly out of scope—abort and document blockers instead.

End-to-end success means **Docker containers** exercised the real serving stack (vLLM or SGLang) with optimized kernels or dispatch patches loaded exactly as production would—not microbench-only speedups.

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
    --name "inference-optimized-{{CONFIG_KEY}}" \
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
    --name "inference-optimized-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{RESULTS_DIR}}" --mode optimize \
    --mount "{{OPTIMIZED_DIR}}:/workspace/optimized" \
    --env "PYTHONPATH=/workspace/optimized" --env "ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
```

Use the block that matches `$FRAMEWORK`. `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` applies to **both** stacks: it tunes RCCL all-reduce quantization for communication bandwidth and is independent of model weight precision.

SGLang’s `PYTHONPATH` prepends `/sgl-workspace/aiter` so packaged CK kernels resolve before falling back to upstream sources; vLLM only needs `/workspace/optimized` unless the handoff adds extra vendor paths—mirror whatever Phase 4 used for profiling parity.

Export `CONTAINER_NAME` immediately after `start_profile_container.sh` returns (match the `--name` argument, typically `inference-optimized-{{CONFIG_KEY}}`). All subsequent `docker exec`, `docker cp`, and `run_profile_exec.sh` calls assume this variable is set.

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

Keep the baseline + optimized JSON filenames referenced in the validator logs. Treat `artifacts_valid = false` or `performance_gate = fail` as validation failure. A `performance_gate = warn` result is still a usable measured outcome: record it honestly, preserve the comparison JSON, and let the monitor/reporting flow decide whether it should remain a WARN or trigger a retry because of other blockers.

When reporting externally, quote **total token throughput** fields exactly as emitted by the benchmark JSON (do not normalize to per-GPU rates unless the baseline file already does). If multiple JSON files match the optimized filename pattern, prefer the newest mtime and archive the others under `{{RESULTS_DIR}}/archive/` to avoid validator ambiguity.

Spot-check `optimization_comparison.json` manually: `speedup` should be `optimized / baseline` within floating-point tolerance, and `performance_gate` should match the measured band. `validated = true` is required only for a clean pass; a `warn` gate is acceptable, but it must never be reported as a passing E2E win.

### 6b. Independent Speedup Verification (V2 only)

When `V2_MONITOR=true` (indicated in the handoff), perform independent verification of GEAK-reported speedups against the measured E2E result:

1. Read per-kernel speedups from `{{PROBLEMS_DIR}}/geak_results.json`.
2. Compute the predicted system-level speedup by weighting each kernel's speedup by its profile contribution (from `{{PROBLEMS_DIR}}/optimization_manifest.json`, field `profile_pct`).
3. Compare `predicted_speedup` against the measured `e2e_speedup` from `{{RESULTS_DIR}}/optimization_comparison.json`.
4. Compute `geak_discrepancy_pct = abs(predicted_speedup - e2e_speedup) / e2e_speedup * 100`.
5. Write verification results to `## Key Findings`:
   - `predicted_speedup`: weighted sum from GEAK per-kernel results
   - `geak_discrepancy_pct`: percentage difference between predicted and measured
   - `geak_verification_status`: `pass` if discrepancy < `GEAK_DISCREPANCY_THRESHOLD_PCT` (default 10%), `warn` otherwise

If `geak_discrepancy_pct` exceeds the threshold, flag the kernels with the largest contribution to the discrepancy. This helps the monitor detect `geak_false_claim` category failures.

### 7. Clean Up
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
docker stop "$CONTAINER_NAME" 2>/dev/null; docker rm "$CONTAINER_NAME" 2>/dev/null
```

### 7b. Write Integration Manifest

After validation, write `{{RESULTS_DIR}}/integration_manifest.json` summarizing every integration target and its outcome:

```json
{
  "schema_version": "1.0",
  "plugin_type": "sglang_plugin",
  "comparison_file": "optimization_comparison.json",
  "targets": [
    {
      "name": "fused_moe",
      "kernel_file": "problem_fused_moe_opt.py",
      "strategy": "plugin",
      "status": "integrated",
      "kernel_speedup": 1.35,
      "blocker_classification": null
    },
    {
      "name": "rope_forward",
      "kernel_file": "problem_rope_forward_opt.py",
      "strategy": "skipped",
      "status": "blocked",
      "kernel_speedup": 1.0,
      "blocker_classification": "true_kernel_parity"
    }
  ],
  "summary": {
    "total_targets": 2,
    "integrated": 1,
    "blocked": 1,
    "skipped": 0,
    "coverage_pct": 0.5
  }
}
```

Populate from `geak_results.json` (kernel speedups) and the plugin `manifest.json` (which kernels were actually registered). For targets not integrated, record the `blocker_classification` from RCA context or Phase 07 metadata.

### Completion
Write `agent-results/phase-08-result.md` with baseline_throughput, optimized_throughput, speedup, plugin_type.

Include these scalar fields in `## Key Findings` for monitor consumption:
- `baseline_file`: filename of the baseline JSON used
- `optimized_file`: filename of the optimized JSON used
- `validation_status`: pass | warn | fail (mirrors `performance_gate` from `optimization_comparison.json`)
- `coverage_pct`: float — fraction of Phase 07 winners that were integrated
- `blocked_target_count`: integer — targets with a structured blocker classification
- `critical_blocker_count`: integer — subset of blocked targets where classification is not `true_kernel_parity`

Reference `results/optimization_comparison.json` and `results/integration_manifest.json` in `## Artifacts`. The monitor reads `artifacts_valid`, `performance_gate`, `e2e_speedup`, `ttft_regression_pct`, and `ttft_upgraded` from the comparison file via detection rules (pre-extracted into the monitor context JSON by the orchestrator).

If the handoff contains a `## Root Cause Analysis` section (from a prior failed attempt), read the RCA artifact and adjust your approach based on the retry recommendation and blocker classifications:
- Targets classified as `true_kernel_parity`: skip integration for these targets.
- Targets classified as `adapter_overhead`: rewrite the adapter to reduce overhead.
- Targets classified as `needs_source_patch` or `needs_model_adapter`: spawn the coding agent for those specific targets.
- Targets classified as `framework_limit`: document as a structured blocker rather than retrying.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
