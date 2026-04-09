# Phase 4: Profiling {{SKIP_LABEL}}

## Objective
Re-run selected benchmarks with profiling enabled to capture detailed performance traces.

{{PROFILE_SKIP_NOTE}}

## Steps

### 1. Select Profiling Configs
Choose a representative subset of configs to profile (typically one concurrency level, one sequence length).
If `{{FILTER_TP}}`, `{{FILTER_CONC_START}}`/`{{FILTER_CONC_END}}`, and `{{FILTER_SEQ}}` are set, use those to narrow configs. Otherwise, pick a low-concurrency config (e.g., conc=4) with the default sequence length.

### 2. Create Profiles Directory
```bash
mkdir -p "{{PROFILE_DIR}}"
sudo rm -f {{REPO_DIR}}/profiles/*.pt.trace.json.gz 2>/dev/null || true
sudo rm -f {{REPO_DIR}}/profiles/*.pt.trace.json 2>/dev/null || true
```

### 3. Start Persistent Profiling Container

Run:
```bash
bash "{{SCRIPTS_DIR}}/start_profile_container.sh" \
    --name "inferencex-profile-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{PROFILE_DIR}}"
```

Capture `CONTAINER_NAME` from output. The container gets access to all GPUs; visibility is restricted per-run at `docker exec` time.

{{DRY_RUN_NOTE}}

### 3½. Restrict Trace Export to Rank 0 Only
Prevents I/O contention when all TP workers write multi-GB traces simultaneously through the same bind-mounted filesystem. Rank-0 alone is sufficient for TraceLens single-rank reports, gap analysis, and phase-split roofline.

Copy the patch script into the container and run it:
```bash
docker cp "{{SCRIPTS_DIR}}/patch_rank0_profiling.py" "$CONTAINER_NAME:/tmp/"
docker exec "$CONTAINER_NAME" python3 /tmp/patch_rank0_profiling.py --framework "$FRAMEWORK"
```

### 3a. Inject Profiler Config
Restore the benchmark script to its original state first:
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
```

Copy the injection script and run it inside the container. For vLLM this injects `--profiler-config.*` CLI args; for SGLang it injects `--disable-cuda-graph` for eager-mode profiling.

```bash
docker cp "{{SCRIPTS_DIR}}/inject_profiler_config.py" "$CONTAINER_NAME:/tmp/"
docker exec \
    -e OSL="${OSL}" -e CONC="${CONC}" -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.5}" \
    "$CONTAINER_NAME" python3 /tmp/inject_profiler_config.py \
    --framework "$FRAMEWORK" --target "/workspace/$BENCHMARK_SCRIPT" {{ENFORCE_EAGER_FLAG}}
```

Where `{{ENFORCE_EAGER_FLAG}}` is `--enforce-eager` if eager mode is enabled, or omitted otherwise.

**vLLM note**: vLLM >= 0.15 requires `--profiler-config.*` CLI args to register `/start_profile` and `/stop_profile` endpoints. Without them, profiling routes are never attached.

**SGLang note**: Profiling is controlled via environment variables already set in the container. The script only patches for eager mode if needed.

### 3b. Disable Relay Trace Staging and Prompt Cap

```bash
docker cp "{{SCRIPTS_DIR}}/patch_benchmark_lib.py" "$CONTAINER_NAME:/tmp/"
docker exec "$CONTAINER_NAME" python3 /tmp/patch_benchmark_lib.py
```

This disables `move_profile_trace_for_relay()` (CI/CD only, not needed here) and the `num_prompts` cap to `max_concurrency` (keeps full prompt count for steady-state profiling with both prefill-decode and decode-only phases).

### 4. Run Profiling Benchmark

**CRITICAL**: Use `CUDA_VISIBLE_DEVICES` for GPU selection, NEVER `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES`. ROCR re-indexes GPUs causing HIP device mismatch errors.

```bash
bash "{{SCRIPTS_DIR}}/run_profile_exec.sh" \
    --container "$CONTAINER_NAME" \
    --benchmark-script "$BENCHMARK_SCRIPT" \
    --model "$MODEL" --tp "$TP" --ep "$EP" --conc "$CONC" \
    --isl "$ISL" --osl "$OSL" --max-model-len "$MAX_MODEL_LEN" \
    --precision "$PRECISION" --framework "$FRAMEWORK" --exp-name "$EXP_NAME" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile" \
    --repo-dir "{{REPO_DIR}}" \
    --gpus "{{GPUS}}" \
    --docker-log "{{PROFILE_DIR}}/${CONTAINER_NAME}_docker_run.log"
```

The script streams output live, emits heartbeats every 30s, and waits for trace flush after completion.

### 5. Clean Up Container
```bash
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
```

### 6. Collect Profile Traces

```bash
bash "{{SCRIPTS_DIR}}/collect_profile_traces.sh" \
    --repo-dir "{{REPO_DIR}}" \
    --profile-dir "{{PROFILE_DIR}}" \
    --output-dir "{{OUTPUT_DIR}}" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile"
```

### 7. Profile Summary
List captured trace files and their sizes.
Traces can be viewed at https://ui.perfetto.dev/

## Completion
Update progress.json:
```json
{
  "phase": "profile",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile"],
  "current_step": "profiling complete",
  "details": {
    "profile_runs": 1,
    "trace_files": ["<list of trace files>"]
  }
}
```
