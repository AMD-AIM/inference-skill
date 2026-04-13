> **ARCHIVE**: This file is a reference copy of the original phase runbook. The active
> agent docs are in `agents/phase-NN-*.md`. Script paths in this file reference the
> pre-reorganization flat layout (`scripts/*.py`); the actual scripts are now under
> `scripts/{env,container,profiling,optimize,plugin,report}/`.

# Phase 2: Benchmark Execution {{SKIP_LABEL}}

## Objective
Run benchmarks for each config point using a single persistent Docker container.

## Steps

### 1. Load Configs
Read configs from `{{OUTPUT_DIR}}/results/sweep_configs.json`. If empty, stop with error.

### 2. Extract Config Fields
For each config entry: `image`, `model`, `precision`, `framework`, `runner`, `isl`, `osl`, `tp`, `ep`, `conc`, `max-model-len`, `exp-name`.

### 3. Determine Benchmark Script
```bash
BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}.sh"
[ ! -f "{{REPO_DIR}}/$BENCHMARK_SCRIPT" ] && BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}_${FRAMEWORK}.sh"
echo "BENCHMARK_SCRIPT={{REPO_DIR}}/$BENCHMARK_SCRIPT"
```

### 4. Group Configs by Docker Image
Group all configs by `image` field. Typically one group per config-key.

### 5. Start Persistent Container

```bash
bash "{{SCRIPTS_DIR}}/start_profile_container.sh" \
    --name "inferencex-benchmark-{{CONFIG_KEY}}" \
    --image "$IMAGE" --runner "$RUNNER" \
    --repo-dir "{{REPO_DIR}}" --hf-cache "{{HF_CACHE}}" \
    --profile-dir "{{OUTPUT_DIR}}/results" \
    --mode benchmark
```

{{DRY_RUN_NOTE}}

### 6. Run Each Benchmark via `docker exec`

**CRITICAL**: Use `CUDA_VISIBLE_DEVICES` for GPU selection, NEVER `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` (causes re-indexing conflicts on AMD).

```bash
bash "{{SCRIPTS_DIR}}/run_profile_exec.sh" \
    --container "$CONTAINER_NAME" \
    --benchmark-script "$BENCHMARK_SCRIPT" \
    --model "$MODEL" --tp "$TP" --ep "$EP" --conc "$CONC" \
    --isl "$ISL" --osl "$OSL" --max-model-len "$MAX_MODEL_LEN" \
    --precision "$PRECISION" --framework "$FRAMEWORK" --exp-name "$EXP_NAME" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}" \
    --repo-dir "{{REPO_DIR}}" --gpus "{{GPUS}}"
```

After each run, collect results:
```bash
bash "{{SCRIPTS_DIR}}/collect_profile_traces.sh" \
    --repo-dir "{{REPO_DIR}}" --profile-dir "{{PROFILE_DIR}}" \
    --output-dir "{{OUTPUT_DIR}}" \
    --result-filename "${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}"
```

### 7. Clean Up Container
```bash
docker stop "$CONTAINER_NAME"; docker rm "$CONTAINER_NAME"
```

## Completion
Update progress.json:
```json
{
  "phase": "benchmark",
  "phases_completed": ["env", "config", "benchmark"],
  "current_step": "benchmarks complete",
  "details": {
    "benchmarks_run": 1,
    "benchmarks_succeeded": 1
  }
}
```
