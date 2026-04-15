# Runtime Notes

Use this file after guided intake and before phase execution.

## Bundled assets

Use only the files bundled next to this skill:

- `phases/00-env-setup.md`
- `phases/01-vllm-setup.md`
- `phases/02-benchmark.md`
- `phases/03-benchmark-analyze.md`
- `phases/04-profiling.md`
- `phases/05-profile-analyze.md`
- `phases/06-problem-generate.md`
- `phases/07-kernel-optimize.md`
- `phases/08-integration.md`
- `phases/09-report-generate.md`
- `scripts/trace_analyzer.py`
- `scripts/select_gpus.py`
- `scripts/classify_kernel.py`
- `scripts/analyze_fusion_vllm.py`
- `scripts/generate_problems_vllm.py`
- `scripts/kernel_test_runner.py`
- `scripts/kernel_finalize.py`
- `scripts/generate_vllm_plugin.py`
- `resources/TraceLens-internal.tar.gz`

If required files are missing, stop and report incomplete installation.

## Required run inputs

Resolve these before loading any phase doc:

- `MODEL`
- `OUTPUT_DIR`
- `PROFILE_DIR`
- `REPORT_DIR`
- `SCRIPTS_DIR`
- `PROGRESS_FILE`
- `TP`
- `ISL`
- `OSL`
- `CONCURRENCY_LEVELS`
- `PRECISION`
- `GPUS`
- `PROFILE`
- `START_PHASE`
- `MODE`
- `PROBLEMS_DIR`
- `OPTIMIZED_DIR`
- `OPTIMIZE_PRIORITY_THRESHOLD`
- `GEAK_DIR`
- `GEAK_OE_DIR`
- `ENV_INFO_FILE`
- `GEAK_MODE` (auto, full, triton_only, manual; from INTAKE optimization extras)
- `OPTIMIZE_SCOPE` (all, fused_only; from INTAKE optimization extras)
- `MAX_OPTIMIZATION_ATTEMPTS` (default 8; from INTAKE optimization extras)
- `MAX_CONSECUTIVE_REJECTIONS` (default 3; from INTAKE optimization extras)

## Recommended defaults

- `OUTPUT_DIR`: `./vllm_optimize_<model-name>_<timestamp>`
- `PROFILE_DIR`: `<OUTPUT_DIR>/profiles`
- `REPORT_DIR`: `<OUTPUT_DIR>/report`
- `SCRIPTS_DIR`: `<OUTPUT_DIR>/scripts`
- `PROGRESS_FILE`: `<OUTPUT_DIR>/progress.json`
- `TP`: `1` (single GPU; use higher for multi-GPU)
- `ISL`: `1024`
- `OSL`: `1024`
- `CONCURRENCY_LEVELS`: `4,8,16,32,64,128`
- `PRECISION`: `half` (fp16; use `bfloat16` for AMD GPUs)
- `START_PHASE`: `env`
- `MODE`: `full`
- `PROBLEMS_DIR`: `<OUTPUT_DIR>/problems`
- `OPTIMIZED_DIR`: `<OUTPUT_DIR>/optimized`
- `OPTIMIZE_PRIORITY_THRESHOLD`: `5.0`
- `GEAK_DIR`: `~/GEAK`
- `GEAK_OE_DIR`: `~/geak-oe`
- `ENV_INFO_FILE`: `<OUTPUT_DIR>/env_info.json`
- `GEAK_MODE`: `auto`
- `OPTIMIZE_SCOPE`: `all`

## Placeholder rules

- Replace `{{VAR}}` placeholders in phase docs using the resolved variable map.
- Ignore `{{SKIP_LABEL}}` markers.
- If a phase doc references `{{PROFILE_SKIP_NOTE}}` or `{{PROFILE_ANALYSIS_NOTE}}`, resolve them from the selected mode before execution.

## Workspace bootstrap

Before executing any phase doc:

1. Create `OUTPUT_DIR`, `PROFILE_DIR`, `REPORT_DIR`, `SCRIPTS_DIR`, `PROBLEMS_DIR`, and `OPTIMIZED_DIR`.
2. Write `config.json` with the resolved run configuration.
3. Write initial `progress.json` if it does not already exist.
4. Copy bundled helper assets into `SCRIPTS_DIR` when needed:
   - `trace_analyzer.py`
   - `select_gpus.py`
   - `TraceLens-internal.tar.gz`
   - `analyze_fusion_vllm.py` (for optimize modes)
   - `generate_problems_vllm.py` (for optimize modes)
   - `kernel_test_runner.py` (for optimize modes)
   - `kernel_finalize.py` (for optimize modes)
   - `generate_vllm_plugin.py` (for optimize modes)

For `resume` or `from-phase`, read existing `progress.json` and artifacts first, then fully rerun the requested start phase.

## Phase map

- `full`: `env -> vllm-setup -> benchmark -> benchmark-analyze -> profiling -> profile-analyze`
- `benchmark`: `env -> vllm-setup -> benchmark -> benchmark-analyze`
- `profile`: `vllm-setup -> profiling -> profile-analyze`
- `optimize`: `env -> vllm-setup -> benchmark -> benchmark-analyze -> profiling -> profile-analyze -> problem-generate -> kernel-optimize -> integration -> report-generate`
- `optimize-only`: `env -> vllm-setup -> problem-generate -> kernel-optimize -> integration -> report-generate`

Read only the phase docs needed for the selected mode and start phase.

## Execution guardrails

- Execute phases in order.
- Update `progress.json` after every completed phase.
- Save all outputs under `OUTPUT_DIR`.
- Do not modify `/opt` or `/usr`.
- If filter application produces zero configs, stop immediately with a clear error.
- If profiling artifacts are root-owned, clean them up and continue.
- Prefer measured data over assumptions when writing reports.

## Execution status updates

Do not rely on tool output alone for user-visible progress.

Before execution begins, send a short status update like:

- `Status 5/5: starting execution. I'll report progress at each phase boundary.`

During execution:

- Before each phase, emit a short update. Adjust the total count to the selected mode:
  - For `full` mode (6 phases): `Running Phase 0/5: Environment Setup` through `5/5: Profile Analysis`
  - For `optimize` mode (10 phases): `Running Phase 0/9: Environment Setup` through `9/9: Final Report`
  - For `optimize-only` mode (6 phases): `Running Phase 0/5: Environment Setup` through `5/5: Final Report`
- After each phase, emit a short completion update.

## Long-running log surfacing

Long phases must not leave the terminal blank.

- If a command may run longer than ~30 seconds, state the running step and full log path.
- Stream stdout/stderr with `tee` while still saving the full log file.
- Before each long benchmark or profile run, print the config being run, the full log path, and what signal to expect next.
- Surface meaningful progress at least every 30-60 seconds.
- Prefer visible progress over silent log files.