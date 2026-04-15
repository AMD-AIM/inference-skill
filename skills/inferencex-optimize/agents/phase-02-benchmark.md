# Phase 2: Benchmark Execution

## Instructions

You are a phase agent responsible for running benchmarks. You read exactly 2 files: this document and your handoff at `handoff/to-phase-02.md`.

**Tools**: Shell commands, Docker, file I/O.
**Outputs**: Write `agent-results/phase-02-result.md`. Save benchmark results to `{{OUTPUT_DIR}}/results/`.
**Errors**: If a benchmark point fails, log the error and continue with the next. Report partial completion.

## Runbook

### 1. Load Configs
Read configs from `{{OUTPUT_DIR}}/results/sweep_configs.json`. If empty, stop with error. Extract per-config: `image`, `model`, `precision`, `framework`, `runner`, `isl`, `osl`, `tp`, `ep`, `conc`, `max-model-len`, `exp-name`.

Execution order for the full matrix: **for each Docker `image` group** → start one long-lived container → **for each config row** in that group, set variables from the row, resolve `BENCHMARK_SCRIPT`, run `run_profile_exec.sh`, then `collect_profile_traces.sh` → stop/remove the container when the group is finished. If one benchmark command fails, log it, record the failure in your phase result, and **continue** with the next row unless the handoff says otherwise.

### 2. Group by Docker Image
Group all configs by the `image` field. **Typically there is one group per config-key** (all points share one image), but always group by `image` in case the sweep mixes images.

### 3. Determine Benchmark Script
Before starting the container for a group, resolve the script path from each config’s `exp-name`, `precision`, `runner`, and `framework` (same logic as Phase 1):

```bash
BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}.sh"
[ ! -f "{{REPO_DIR}}/$BENCHMARK_SCRIPT" ] && BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}_${FRAMEWORK}.sh"
echo "BENCHMARK_SCRIPT={{REPO_DIR}}/$BENCHMARK_SCRIPT"
```

For **each** row, bind shell variables from the JSON keys before calling the scripts, for example: `EXP_NAME` ← `exp-name`, `MODEL` ← `model`, `PRECISION` ← `precision`, `FRAMEWORK` ← `framework`, `RUNNER` ← `runner`, `ISL`/`OSL` ← `isl`/`osl`, `TP`/`EP`/`CONC` ← `tp`/`ep`/`conc`, `MAX_MODEL_LEN` ← `max-model-len`, `IMAGE` ← `image`. Pass `{{GPUS}}` from the handoff (or host policy) into `--gpus` so the wrapper maps the correct physical devices **without** toggling ROCm visibility env vars yourself.

### 4. Start Persistent Container
The container is started with access to **all host GPUs**; per-benchmark GPU visibility is enforced inside `run_profile_exec.sh` via `CUDA_VISIBLE_DEVICES` (see next step). Capture `CONTAINER_NAME` from the start script’s output (or your handoff convention) and reuse it for every `docker exec` / `collect_profile_traces.sh` invocation for that image group.

For each group, use the `IMAGE` and `RUNNER` values taken from the sweep rows in that group (same source as Phase 1’s `sweep_configs.json`).

```bash
bash "{{SCRIPTS_DIR}}/container/start_profile_container.sh" \
    --name "inferencex-benchmark-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{OUTPUT_DIR}}/results" --mode benchmark
```

{{DRY_RUN_NOTE}}

### 5. Run Each Benchmark
**CRITICAL — device visibility:** Use **`CUDA_VISIBLE_DEVICES`** for GPU selection inside the exec wrapper. **Never** use `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` **for inference server processes** (vLLM / SGLang) on AMD: they re-index devices and cause mismatches with what the stack expects. (GEAK and kernel microbenchmark contexts use `HIP_VISIBLE_DEVICES` normally — this restriction applies only to inference server processes.)

```bash
bash "{{SCRIPTS_DIR}}/container/run_profile_exec.sh" \
    --container "$CONTAINER_NAME" \
    --benchmark-script "$BENCHMARK_SCRIPT" \
    --model "$MODEL" --tp "$TP" --ep "$EP" --conc "$CONC" \
    --isl "$ISL" --osl "$OSL" --max-model-len "$MAX_MODEL_LEN" \
    --precision "$PRECISION" --framework "$FRAMEWORK" --exp-name "$EXP_NAME" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}" \
    --repo-dir "{{REPO_DIR}}" --gpus "{{GPUS}}"
```

`run_profile_exec.sh` **streams benchmark output live**, emits **heartbeats every ~30s** so long runs do not look hung, and waits appropriately for completion before returning.

After **each** benchmark finishes successfully (or fails), run collection so results and any sidecar artifacts land under `{{OUTPUT_DIR}}` with the same `result-filename` stem you passed to the runner. Use the `PROFILE_DIR` / profile-dir layout expected by your handoff (often `{{OUTPUT_DIR}}/results` in benchmark mode).

Collect after each run:
```bash
bash "{{SCRIPTS_DIR}}/container/collect_profile_traces.sh" \
    --repo-dir "{{REPO_DIR}}" --profile-dir "{{PROFILE_DIR}}" \
    --output-dir "{{OUTPUT_DIR}}" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}"
```

### 6. Clean Up
```bash
docker stop "$CONTAINER_NAME"; docker rm "$CONTAINER_NAME"
```

### Completion
Write `agent-results/phase-02-result.md` with benchmarks_run, benchmarks_succeeded, result file paths.

Include these sticky fields in `## Data for Next Phase`:
- `baseline_tpot`: float (time per output token in ms at best concurrency)
- `baseline_throughput`: float (total token throughput at best concurrency)

Include these scalar fields in `## Key Findings` for monitor consumption:
- `benchmark_result_status`: completed | failed | partial
- `benchmarks_succeeded`: integer count of successful benchmark runs
- `baseline_artifacts_ready`: true | false (all expected result JSONs present and non-empty)

If the handoff contains a `## Root Cause Analysis` section (from a prior failed attempt), read the RCA artifact and adjust your approach based on the retry recommendation and blocker classifications before re-executing benchmarks.

Double-check before closing the phase: every sweep row you intended to run has either a collected artifact under `{{OUTPUT_DIR}}` or a logged failure reason; container `docker rm` succeeded so no orphaned `inferencex-benchmark-*` instances remain; and result filenames remain unique per `(exp-name, precision, framework, tp, ep, conc)` so later analysis does not overwrite data.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
