# Phase 1: Config Parsing & Sweep Generation

## Instructions

You are a phase agent responsible for parsing the master YAML config and generating the benchmark sweep matrix. You read exactly 2 files: this document and your handoff at `handoff/to-phase-01.md`.

**Tools**: Shell commands, Python, file I/O.
**Outputs**: Write `agent-results/phase-01-result.md`. Write `{{OUTPUT_DIR}}/results/sweep_configs.json`.
**Errors**: If zero configs match filters, report failure with available filter values. Do NOT proceed.

## Runbook

### 1. Detect Config File and Validate
```bash
CONFIG_FILE=$([[ "{{CONFIG_KEY}}" == *mi3* ]] && echo ".github/configs/amd-master.yaml" || echo ".github/configs/nvidia-master.yaml")
echo "Config file: $CONFIG_FILE"
python3 "{{SCRIPTS_DIR}}/env/validate_config_key.py" \
    --config-file "{{REPO_DIR}}/$CONFIG_FILE" --config-key "{{CONFIG_KEY}}"
```

If the validator exits non-zero, **stop immediately** and use one of the suggested keys. Do **not** run sweep generation against a missing or invalid config key.

### 2. Generate Sweep Configs
The matrix script must run from the InferenceX repo root:

```bash
cd "{{REPO_DIR}}"
python3 utils/matrix_logic/generate_sweep_configs.py \
    test-config --config-files "$CONFIG_FILE" --config-keys "{{CONFIG_KEY}}" > /tmp/sweep_raw.json 2>&1
```

**Do NOT print the raw sweep output** — it is very large and noisy. Capture it silently (redirect to a temp file as above). Only print filtered or summary views later.

### 3. Apply Filters
**Field names in the sweep JSON output:** `generate_sweep_configs.py` emits configs with these exact keys (lowercase, hyphenated): `tp`, `conc`, `ep`, `isl`, `osl`, `model`, `framework`, `precision`, `runner`, `image`, `exp-name`, `max-model-len`. Use these exact keys when reading values — do **not** use aliases like `concurrency`, `CONC`, or `max_concurrency`.

Apply filters **before** saving. Only save and print configs that pass all active filters. Never print the full unfiltered sweep.

Sequence mapping: `1k1k` → ISL=1024, OSL=1024; `1k8k` → ISL=1024, OSL=8192; `8k1k` → ISL=8192, OSL=1024.

Filter semantics (mirror the master phase runbook):

- If `FILTER_TP` is set, keep only rows whose `tp` matches.
- If `FILTER_EP` is set, keep only rows whose `ep` matches; rows **without** `ep` behave as **EP=1**.
- If `FILTER_SEQ` is set, keep only rows whose `isl` and `osl` match the mapped lengths for that token (see sequence mapping above).
- If `FILTER_CONC_START` / `FILTER_CONC_END` are set, keep rows whose `conc` is **≥** start and **≤** end (inclusive when both are set).

Save the filtered result to `{{OUTPUT_DIR}}/results/sweep_configs.json`. **Only print** the filtered configs (or a short summary), never the full unfiltered matrix.

**Config resolution:** When executing later phases, read each entry in `sweep_configs.json` and bind `runner`, `image`, and `framework` (and other fields) from those keys into shell variables such as `RUNNER`, `IMAGE`, and `FRAMEWORK` for container startup and script resolution. The saved JSON is the source of truth for which image and runner each benchmark point uses.

### 4. Handle Zero Configs
If **zero** configs remain after filtering, **stop the pipeline immediately**:

1. Print a clear error: `ERROR: No configs match the applied filters.`
2. Print which filters were active and what led to the empty set.
3. Print **available values** for each filter dimension (TP list, concurrency levels, sequence lengths) so the user can correct CLI flags.
4. Report the error in `agent-results/phase-01-result.md`.

5. **Do not** proceed to Phase 2 or later phases.

### 5. Resolve Benchmark Script
For each filtered config, substitute the actual values of `EXP_NAME` (from `exp-name`), `PRECISION`, `RUNNER`, and `FRAMEWORK` from the config. Try the shorter filename first, then the fallback that appends `_$FRAMEWORK`:

```bash
BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}.sh"
if [ ! -f "{{REPO_DIR}}/$BENCHMARK_SCRIPT" ]; then
    BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_${RUNNER}_${FRAMEWORK}.sh"
fi
```

When reporting paths, print the **fully expanded** path with real values (e.g. `benchmarks/single_node/kimik2.5_fp4_mi355x.sh`), not shell parameter templates. Verify the file exists under `{{REPO_DIR}}`. If missing, warn. Optionally print the first few lines of the script so operators can confirm it matches the intended benchmark.

### 6. Report Config Summary
Print: total configs from the generator (before filtering), count after filtering, and per saved config: model, framework, precision, TP, EP, concurrency, ISL×OSL, and resolved benchmark script path.

### Completion
Write `agent-results/phase-01-result.md` with total/filtered config counts, benchmark script path, and saved sweep config path.

Include these sticky fields in `## Data for Next Phase`:
- `tp_size`: integer (tensor parallelism from sweep config)
- `ep_size`: integer (expert parallelism, default 1)
- `seq_lengths`: string (comma-separated ISL/OSL pairs)
- `concurrency_levels`: string (comma-separated concurrency values)

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
