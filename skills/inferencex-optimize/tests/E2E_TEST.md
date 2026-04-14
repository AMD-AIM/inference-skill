# E2E Test: inferencex-optimize

Automated end-to-end testing for the inferencex-optimize skill pipeline.

## Prerequisites

- 8x AMD MI350X/MI355X GPUs (or adjust TP for available GPU count)
- Docker images pulled for the target framework:
  - vLLM: `vllm/vllm-openai-rocm:v0.18.0`
  - SGLang: `lmsysorg/sglang:v0.5.9-rocm700-mi35x`
- HF cache with model weights (or network access to download)
- `inference-skill` repo cloned at `~/inference-skill`
- Skill installed: `cd ~/inference-skill && ./install.sh`
- Claude Code CLI available (`claude` command)

## Test Targets

| Target | Config Key | Model | Size | Framework |
|--------|-----------|-------|------|-----------|
| vLLM | `gptoss-fp4-mi355x-vllm` | amd/gpt-oss-120b-w-mxfp4-a-fp8 | 120B dense | vLLM |
| SGLang | `dsr1-fp4-mi355x-sglang` | amd/DeepSeek-R1-0528-MXFP4-Preview | 671B MoE | SGLang |

---

## Full E2E Test: New Run From Phase 0

This runs the complete optimize pipeline from scratch and validates every phase output.

### Step 1: Clean up any stale state

Kill leftover containers and check GPU availability before starting:

```bash
# Kill any leftover inferencex containers
docker ps -q --filter label=inferencex-pipeline | xargs -r docker stop
docker ps -aq --filter label=inferencex-pipeline | xargs -r docker rm

# Verify GPUs are free
rocm-smi --showuse 2>/dev/null || nvidia-smi
```

### Step 2: Launch the pipeline

Open a terminal and run the pipeline with Claude Code. Pick one target:

**vLLM (120B dense, ~30-60 min):**
```bash
claude "Use inferencex-optimize for gptoss-fp4-mi355x-vllm with optimize workflow, TP=8, 1k1k, conc=4"
```

**SGLang (671B MoE, ~45-90 min):**
```bash
claude "Use inferencex-optimize for dsr1-fp4-mi355x-sglang with optimize workflow, TP=8, 1k1k, conc=4"
```

Claude Code will execute all 10 phases in order:

| # | Phase | What happens |
|---|-------|-------------|
| 0 | env-setup | Checks Docker, GPUs, InferenceX repo, HF cache |
| 1 | config-parse | Generates `sweep_configs.json` from config key |
| 2 | benchmark | Runs baseline benchmark in Docker container |
| 3 | benchmark-analyze | Produces `benchmark_summary.json` |
| 4 | profile | Runs profiling benchmark with torch profiler |
| 5 | profile-analyze | Parses trace, generates `gap_analysis.json` and `trace_manifest.json` |
| 6 | problem-generate | Creates `problem_*.py` files for each bottleneck kernel |
| 7 | kernel-optimize | Writes optimized Triton kernels, tests accuracy + speedup |
| 8 | integration | Builds framework plugin, runs optimized E2E benchmark |
| 9 | report-generate | Produces `optimization_report.md` and summary JSON |

### Step 3: Note the output directory

When the pipeline completes, note the output directory path. It will be printed at the end and follows this pattern:

```
~/inferencex_<config-key>_<YYYYMMDD>_<HHMMSS>/
```

Example: `~/inferencex_gptoss-fp4-mi355x-vllm_20260330_140000/`

### Step 4: Validate the output

```bash
python3 ~/.claude/skills/inferencex-optimize/tests/e2e_optimize_test.py \
  --output-dir ~/inferencex_<config-key>_<timestamp>/
```

This runs all validation checks and generates `test_report.json` + `test_report.md` in the output directory.

### Step 5: Review the report

```bash
cat ~/inferencex_<config-key>_<timestamp>/test_report.md
```

Or check the JSON for programmatic use:
```bash
python3 -c "import json; r=json.load(open('test_report.json')); print(f'{r[\"passed\"]} pass, {r[\"failed\"]} fail, {r[\"warnings\"]} warn, {r[\"issues_count\"]} issues')"
```

A clean run should show **0 failures**. Any failure points to a phase doc that needs investigation.

---

## Interactive Mode

For a guided experience that combines Steps 2-4:

```bash
python3 ~/.claude/skills/inferencex-optimize/tests/e2e_optimize_test.py
```

The script prompts:
```
Which test case do you want to run?
  1. vLLM   (gptoss-fp4-mi355x-vllm, 120B dense)
  2. SGLang (dsr1-fp4-mi355x-sglang, 671B MoE)
  3. Both   (run sequentially)
  4. Validate existing output directory
>
```

Options 1-3 print the Claude Code command to run, then ask for the output directory after the pipeline completes. Option 4 validates an existing directory directly.

### Other CLI options

```bash
# Skip the menu, go straight to a target
python3 ~/.claude/skills/inferencex-optimize/tests/e2e_optimize_test.py --target vllm
python3 ~/.claude/skills/inferencex-optimize/tests/e2e_optimize_test.py --target sglang

# Validate only (no pipeline run)
python3 ~/.claude/skills/inferencex-optimize/tests/e2e_optimize_test.py --output-dir <path>
```

If you run from the repo checkout, the legacy path still works:

```bash
python3 ~/inference-skill/tests/e2e_optimize_test.py --target vllm
```

---

## What Gets Validated

### Phase Artifact Checks (27 checks)

| Phase | Check |
|-------|-------|
| 0 env | `progress.json` exists with `"env"` in completed phases |
| 1 config | `sweep_configs.json` is valid JSON array with required fields |
| 2 benchmark | Benchmark result JSON exists with throughput > 0 |
| 3 analyze | `benchmark_summary.json` exists with `configs` or `results` key |
| 4 profile | Trace files exist, size > 1MB, gzip integrity passes |
| 5 prof-analyze | `gap_analysis.json` has `top_kernels` and `category_breakdown` |
| 6 problem-gen | Problem files have `class Model`, `get_inputs`, `get_init_inputs`; manifest exists |
| 7 kernel-opt | `geak_results.json` with speedup data; winner/regression staging cross-check |
| 8 integration | Plugin manifest exists; registered kernels cross-checked against geak speedups |
| 8 integration | `optimization_comparison.json` has consistent tri-state fields; `performance_gate=pass` passes, `performance_gate=warn` warns, `performance_gate=fail` fails |
| 9 report | Report markdown (> 500 bytes) and summary JSON exist |
| progress | All expected phases completed in correct order |

### Kernel-Integration Cross-Check

The validator verifies the Phase 07 -> Phase 08 handoff:

| Kernel speedup | In `optimized/`? | Result |
|---|---|---|
| < 1.0 | No | pass -- regressed kernel correctly skipped |
| < 1.0 | Yes | **fail** -- regressed kernel leaked into integration |
| >= 1.0 | Yes | pass -- winning kernel correctly staged |
| >= 1.0 | No | warn -- winning kernel missing from integration |

Plugin manifest entries are also cross-checked: any registered kernel with speedup < 1.0 is a failure.

### Generic Issue Detection

Scans all logs and artifacts for any failures (not pattern-matched to known issues):

- Error patterns in log files (ERROR, FAIL, Traceback, RuntimeError, OOM, HIP/CUDA errors, TimeoutError, 404, SIGKILL)
- False positives filtered: `error_rate`, `error_count`, `FAILED (accuracy)` in optimization history logs
- Truncated gzip trace files
- E2E warn-band results (`0.97 <= speedup < 1.0`) vs fail-band regressions (`speedup < 0.97`)
- Regressed kernel leakage into `optimized/` or plugin manifest

Each issue includes: source file/line, severity, matched pattern, context, suggested phase doc, analysis.

---

## Output

The validator writes two report files to the output directory:

- `test_report.json` -- machine-readable results with all checks and issues
- `test_report.md` -- human-readable summary with per-phase status and recommendations

Exit code: `0` if all checks pass (warnings allowed), `1` if any check fails.

---

## Troubleshooting

### Pipeline hangs at Phase 4 (profiling)

vLLM v0.18 may return 404 on `/start_profile`. Phase 04 has a force-mount patch for this. If the pipeline still hangs, kill the container and check the profile logs:

```bash
docker logs inferencex-profile-<config-key> 2>&1 | tail -50
```

### GPU memory not freed between phases

Stale server processes may hold GPU memory. The phase docs include pre-run cleanup (`pkill -f "vllm.entrypoints"`), but if a container crashed:

```bash
docker ps -q --filter label=inferencex-pipeline | xargs -r docker stop
docker ps -aq --filter label=inferencex-pipeline | xargs -r docker rm
```

### Trace file truncated (gzip integrity fail)

Large torch traces (100MB-1GB) need time to finish writing. Phase 04 includes a flush-wait loop (polls file size until stable, max 60s). If traces still truncate, the model's trace is too large for the timeout. The profile-analyze phase can still work with a partial trace.

### Kernel regression (speedup < 1.0)

This is expected -- not all kernels benefit from Triton optimization. The validator checks that regressed kernels were correctly excluded from integration. A regression is only a failure if the kernel leaked into the plugin.

## Known Limitations

- No smaller SGLang model exists in InferenceX for MI355X; DSR1-FP4 (671B MoE) is the test target
- vLLM v0.18 profiler route registration may require force-mount patch
- Large model traces (>200MB) may truncate if flush-wait times out
- SGLang profiling uses env vars (`SGLANG_TORCH_PROFILER_DIR`), not CLI args

## Multi-Agent Workspace Validation

When running in multi-agent mode, the E2E validator additionally checks:

### Workspace directories

- `handoff/` — Contains `to-phase-NN.md` for each dispatched phase
- `agent-results/` — Contains `phase-NN-result.md` per `protocols/phase-result.schema.md`
- `monitor/` — Contains `running-summary.md` and `phase-NN-review.md` per `protocols/monitor-feedback.schema.md`

### Skill layout

- `orchestrator/ORCHESTRATOR.md`, `phase-registry.json`, `monitor.md` exist
- `agents/` contains 12 agent files (10 phase agents + coding-agent + analysis-agent)
- `protocols/` contains 5 schema files

### Troubleshooting multi-agent issues

- **Phase agent failed**: Check `agent-results/phase-NN-result.md` for error details
- **Monitor FAIL verdict**: Check `monitor/phase-NN-review.md` for failure_type and rerun guidance
- **Missing handoff**: Orchestrator dispatch loop may have stopped early — check `progress.json`
- **Stale running-summary**: Monitor may not have been spawned — verify orchestrator completed the phase
