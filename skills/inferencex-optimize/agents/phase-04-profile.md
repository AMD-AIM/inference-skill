# Phase 4: Profiling

## Instructions

You are a phase agent responsible for running profiled benchmarks to capture GPU performance traces. You read exactly 2 files: this document and your handoff at `handoff/to-phase-04.md`.

**Tools**: Shell commands, Docker, file I/O.
**Outputs**: Write `agent-results/phase-04-result.md`. Save trace files to `{PROFILE_DIR}`.
**Errors**: If container fails to start or traces are not produced, report failure with details.

## Runbook

### 1. Select Profiling Configs
Choose a representative subset from `{{OUTPUT_DIR}}/results/sweep_configs.json` (typically one concurrency level and one sequence length).

**Filter usage:** If `{{FILTER_TP}}`, `{{FILTER_CONC_START}}` / `{{FILTER_CONC_END}}`, and `{{FILTER_SEQ}}` are set in the handoff, use them to narrow which sweep rows to profile. Otherwise default to a **low-concurrency** point (e.g. `conc=4`) and the default sequence length for that config key.

### 2. Prepare Profiles Directory
```bash
mkdir -p "{{PROFILE_DIR}}"
sudo rm -f {{REPO_DIR}}/profiles/*.pt.trace.json.gz 2>/dev/null || true
sudo rm -f {{REPO_DIR}}/profiles/*.pt.trace.json 2>/dev/null || true
```

### 3. Start Profiling Container
Capture `CONTAINER_NAME` from the script output. The container receives access to **all host GPUs**; restrict visibility per profiling run at `docker exec` time (via the run script / `CUDA_VISIBLE_DEVICES`), not by starting a GPU-sliced container.

```bash
bash "{{SCRIPTS_DIR}}/container/start_profile_container.sh" \
    --name "inferencex-profile-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{PROFILE_DIR}}"
```

{{DRY_RUN_NOTE}}

### 3.5. Restrict Trace Export to Rank 0
**Why:** Avoid I/O contention when every tensor-parallel rank tries to write multi-gigabyte traces through the same bind-mounted filesystem. **Rank 0 alone** is enough for TraceLens single-rank reports, gap analysis, and phase-split roofline work.

```bash
docker cp "{{SCRIPTS_DIR}}/profiling/patch_rank0_profiling.py" "$CONTAINER_NAME:/tmp/"
docker exec "$CONTAINER_NAME" python3 /tmp/patch_rank0_profiling.py --framework "$FRAMEWORK"
```

### 3a. Inject Profiler Config
Restore benchmark script first:
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
```

For **vLLM**, the injector adds `--profiler-config.*` CLI arguments. For **SGLang**, it may add `--disable-cuda-graph` when eager-mode profiling is required.

Inject profiler settings:
```bash
docker cp "{{SCRIPTS_DIR}}/profiling/inject_profiler_config.py" "$CONTAINER_NAME:/tmp/"
docker exec \
    -e OSL="${OSL}" -e CONC="${CONC}" -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.5}" \
    "$CONTAINER_NAME" python3 /tmp/inject_profiler_config.py \
    --framework "$FRAMEWORK" --target "/workspace/$BENCHMARK_SCRIPT" {{ENFORCE_EAGER_FLAG}}
```

Where `{{ENFORCE_EAGER_FLAG}}` is `--enforce-eager` when eager mode is required, or omitted otherwise.

**vLLM (>= 0.15):** Profiling HTTP routes require **`--profiler-config.*` CLI arguments** so `/start_profile` and `/stop_profile` are registered. Without them, profiling endpoints are never attached.

**SGLang:** Profiling is primarily controlled by **environment variables** already set in the container; the injector mainly adds eager-mode patches when `--enforce-eager` is in play.

### 3b. Disable Relay Trace Staging and Prompt Cap
```bash
docker cp "{{SCRIPTS_DIR}}/profiling/patch_benchmark_lib.py" "$CONTAINER_NAME:/tmp/"
docker exec "$CONTAINER_NAME" python3 /tmp/patch_benchmark_lib.py
```

This patch **disables `move_profile_trace_for_relay()`** (CI/CD relay staging, not needed for local profiling) and **lifts the `num_prompts` cap tied to `max_concurrency`**, preserving the full prompt count for steady-state profiling across prefill-decode and decode-only phases.

### 4. Run Profiling Benchmark
**CRITICAL:** Use **`CUDA_VISIBLE_DEVICES`** for GPU selection. **Never** `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` — ROCm re-indexes GPUs and triggers HIP device mismatch errors.

```bash
bash "{{SCRIPTS_DIR}}/container/run_profile_exec.sh" \
    --container "$CONTAINER_NAME" \
    --benchmark-script "$BENCHMARK_SCRIPT" \
    --model "$MODEL" --tp "$TP" --ep "$EP" --conc "$CONC" \
    --isl "$ISL" --osl "$OSL" --max-model-len "$MAX_MODEL_LEN" \
    --precision "$PRECISION" --framework "$FRAMEWORK" --exp-name "$EXP_NAME" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile" \
    --repo-dir "{{REPO_DIR}}" --gpus "{{GPUS}}" \
    --docker-log "{{PROFILE_DIR}}/${CONTAINER_NAME}_docker_run.log"
```

The wrapper **streams logs live**, emits **heartbeats every ~30s**, and **waits for trace flush** after the benchmark completes so `.pt.trace.json` files finish writing before collection.

### 5. Clean Up
```bash
docker stop "$CONTAINER_NAME"; docker rm "$CONTAINER_NAME"
```

### 6. Collect Traces
```bash
bash "{{SCRIPTS_DIR}}/container/collect_profile_traces.sh" \
    --repo-dir "{{REPO_DIR}}" --profile-dir "{{PROFILE_DIR}}" \
    --output-dir "{{OUTPUT_DIR}}" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile"
```

### 7. Profile Summary
List every captured trace under `{{PROFILE_DIR}}` (and copied outputs) with **file sizes**. For interactive viewing, open traces in **[Perfetto](https://ui.perfetto.dev/)**.

### Completion
Write `agent-results/phase-04-result.md` with trace file list (and sizes), profile run count, container log path, and Perfetto viewing note.

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
