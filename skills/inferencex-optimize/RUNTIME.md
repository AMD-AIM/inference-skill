# Runtime Notes

Use this file after guided intake and before phase execution.

## Bundled assets

Use only the files bundled next to this skill:

**Phase docs:**
- `phases/00-env-setup.md` through `phases/09-report-generate.md`

**Pre-existing scripts (domain logic):**
- `scripts/trace_analyzer.py`
- `scripts/select_gpus.py`
- `scripts/classify_kernel.py`
- `scripts/analyze_fusion_inferencex.py`
- `scripts/generate_problems_inferencex.py`
- `scripts/kernel_test_runner.py`
- `scripts/kernel_finalize.py`
- `scripts/generate_vllm_plugin.py`
- `scripts/generate_sglang_plugin.py`

**Extracted scripts (phase automation):**
- `scripts/generate_env_info.py` — GPU/GEAK/API key detection (Phase 0)
- `scripts/validate_config_key.py` — Master YAML config-key validation with close-match suggestions (Phases 0, 1)
- `scripts/start_profile_container.sh` — Docker container creation with `--mode benchmark|profile|optimize`, extra `--mount` and `--env` support (Phases 2, 4, 7, 8)
- `scripts/run_profile_exec.sh` — Docker exec with heartbeats + trace flush (Phases 2, 4, 8)
- `scripts/collect_profile_traces.sh` — Trace/result collection from repo (Phases 2, 4)
- `scripts/patch_rank0_profiling.py` — Rank-0-only trace export (Phase 4, runs inside container)
- `scripts/inject_profiler_config.py` — Profiler config injection (Phase 4, runs inside container)
- `scripts/patch_benchmark_lib.py` — Relay/cap disable (Phase 4, runs inside container)
- `scripts/validate_traces.py` — Trace discovery and validation (Phase 5)
- `scripts/detect_gpu_arch.py` — GPU arch detection for roofline (Phase 5)
- `scripts/install_tracelens.sh` — TraceLens installation (Phase 5)
- `scripts/run_tracelens.sh` — TraceLens analysis pipeline (Phase 5)
- `scripts/display_tracelens_results.sh` — Console result display (Phase 5)
- `scripts/extract_model_shapes.py` — Model shape extraction (Phase 6)
- `scripts/resolve_geak_mode.py` — GEAK mode resolution (Phase 7)
- `scripts/load_optimization_manifest.py` — Manifest loading/filtering (Phase 7)
- `scripts/collect_winning_kernels.py` — Kernel collection (Phase 7, runs inside container)
- `scripts/verify_winning_kernels.py` — Pre-integration verification (Phase 8)
- `scripts/inject_plugin.py` — Plugin injection (Phase 8, runs inside container)
- `scripts/validate_optimization.py` — Baseline vs optimized comparison (Phase 8)
- `scripts/generate_optimization_summary.py` — Summary JSON generation (Phase 9)

**Templates:**
- `templates/agent-config.md`
- `templates/benchmark_report.md` — Phase 3 report template
- `templates/dispatch_plugin_example.py` — Reference shape-aware dispatch plugin template (Phase 8 Step 1.5)
- `templates/profiling_report.md` — Phase 5 report template
- `templates/profile_analysis_schema.json` — Phase 5 JSON schema
- `templates/optimization_report.md` — Phase 9 report template

**Resources:**
- `resources/TraceLens-internal.tar.gz`

If required files are missing, stop and report incomplete installation.

## Required run inputs

Resolve these before loading any phase doc:

- `CONFIG_KEY`
- `OUTPUT_DIR`
- `REPO_DIR`
- `REPO_URL`
- `HF_CACHE`
- `RESULTS_DIR`
- `PROFILE_DIR`
- `REPORT_DIR`
- `SCRIPTS_DIR`
- `PROGRESS_FILE`
- `FILTER_TP`
- `FILTER_EP`
- `FILTER_CONC_START`
- `FILTER_CONC_END`
- `FILTER_SEQ`
- `GPUS`
- `DRY_RUN`
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
- `RESOURCES_DIR` (path to installed skill's `resources/` directory; contains `TraceLens-internal.tar.gz`)
- `TEMPLATES_DIR` (path where bundled templates are copied during bootstrap)
- `ENFORCE_EAGER_FLAG` (set to `--enforce-eager` when eager mode is required for the framework, or empty string otherwise; resolved by the agent based on framework and profiling requirements)

## Recommended defaults

- `OUTPUT_DIR`: `./inferencex_<config-key>_<timestamp>`
- `REPO_DIR`: `<OUTPUT_DIR>/InferenceX`
- `HF_CACHE`: `$HF_HUB_CACHE` or `~/.cache/huggingface`
- `RESULTS_DIR`: `<OUTPUT_DIR>/results`
- `PROFILE_DIR`: `<OUTPUT_DIR>/profiles`
- `REPORT_DIR`: `<OUTPUT_DIR>/report`
- `SCRIPTS_DIR`: `<OUTPUT_DIR>/scripts`
- `PROGRESS_FILE`: `<OUTPUT_DIR>/progress.json`
- `START_PHASE`: `env`
- `MODE`: `full`
- `PROBLEMS_DIR`: `<OUTPUT_DIR>/problems`
- `OPTIMIZED_DIR`: `<OUTPUT_DIR>/optimized`
- `OPTIMIZE_PRIORITY_THRESHOLD`: `5.0` (minimum percent of total kernel time to create a problem file)
- `GEAK_DIR`: `~/GEAK` (GEAK installation directory for HIP kernel optimization)
- `GEAK_OE_DIR`: `~/geak-oe` (geak-oe OpenEvolve directory, optional)
- `ENV_INFO_FILE`: `<OUTPUT_DIR>/env_info.json` (environment info written by Phase 0)
- `GEAK_MODE`: `auto` (auto-detect GEAK availability; `full` = Triton + HIP/CK; `triton_only` = simple mode only; `manual` = no GEAK)
- `OPTIMIZE_SCOPE`: `all` (optimize all bottleneck kernels; `fused_only` = only fused operator problems)
- `RESOURCES_DIR`: `<installed_skill_root>/resources` (the `resources/` directory alongside `SKILL.md`; not copied to OUTPUT_DIR due to tarball size -- must remain accessible during execution)
- `TEMPLATES_DIR`: `<OUTPUT_DIR>/templates`
- `ENFORCE_EAGER_FLAG`: `""` (empty; set to `--enforce-eager` by the agent when the framework requires eager mode for profiling)

## Placeholder rules

- Replace `{{VAR}}` placeholders in phase docs using the resolved variable map.
- Ignore `{{SKIP_LABEL}}` markers. They are compiler annotations from the built-in OpenCode command path.
- If a phase doc references `{{DRY_RUN_NOTE}}`, resolve it from `DRY_RUN`: empty for real runs, or a short note that benchmark/profile commands should be printed and validated without being executed.
- If a phase doc references `{{PROFILE_SKIP_NOTE}}` or `{{PROFILE_ANALYSIS_NOTE}}`, resolve them from the selected mode before execution.

## Workspace bootstrap

Before executing any phase doc:

1. Create `OUTPUT_DIR`, `RESULTS_DIR`, `PROFILE_DIR`, `REPORT_DIR`, `SCRIPTS_DIR`, `TEMPLATES_DIR`, `PROBLEMS_DIR`, and `OPTIMIZED_DIR`.
2. Write `config.json` with the resolved run configuration.
3. Write initial `progress.json` if it does not already exist.
4. Copy ALL bundled scripts into `SCRIPTS_DIR` and ALL bundled templates into `TEMPLATES_DIR`. Phase docs reference scripts via `{{SCRIPTS_DIR}}/` and templates via `{{TEMPLATES_DIR}}/`. `RESOURCES_DIR` points to the installed skill's `resources/` directory in place (not copied); it must remain accessible during Phase 5 TraceLens installation.

For `resume` or `from-phase`, read existing `progress.json` and artifacts first, then fully rerun the requested start phase.

## Phase map

- `full`: `env -> config -> benchmark -> benchmark-analyze -> profile -> profile-analyze`
- `benchmark`: `env -> config -> benchmark -> benchmark-analyze`
- `profile`: `env -> config -> profile -> profile-analyze`
- `benchmark+profile`: same phase set as `full`
- `optimize`: `env -> config -> benchmark -> benchmark-analyze -> profile -> profile-analyze -> problem-generate -> kernel-optimize -> integration -> report-generate`
- `optimize-only`: `env -> config -> problem-generate -> kernel-optimize -> integration -> report-generate` (requires existing `gap_analysis.json` from a prior profile run)

Read only the phase docs needed for the selected mode and start phase.

## Execution guardrails

- Execute phases in order.
- Update `progress.json` after every completed phase.
- Save all outputs under `OUTPUT_DIR`.
- Do not modify `/opt` or `/usr`.
- Only patch the cloned InferenceX repo temporarily when profiling requires it; restore patched files afterward.
- If filter application produces zero configs, stop immediately with a clear error.
- If profiling artifacts are root-owned, clean them up and continue instead of leaving the run half-complete.
- Prefer measured data over assumptions when writing reports.

## Execution status updates

Do not rely on tool output alone for user-visible progress.

Before execution begins, send a short status update like:

- `Status 5/5: starting execution. I’ll report progress at each phase boundary.`

During execution:

- Before each phase, emit a short update. Adjust the total count to the selected mode:
  - For `full` mode (6 phases): `Running Phase 0/5: Environment Setup` through `5/5: Profile Analysis`
  - For `optimize` mode (10 phases): `Running Phase 0/9: Environment Setup` through `9/9: Final Report`
  - For `optimize-only` mode (6 phases): `Running Phase 0/5: Environment Setup` through `5/5: Final Report`
- After each phase, emit a short completion update that also says what comes next.
- Before any long benchmark or profile step, say what is about to take time.
- If a step is unexpectedly slow, add another brief update instead of leaving the user in silence.

## Long-running log surfacing

Long phases must not leave the terminal blank.

Rules:

- If a command may run longer than ~30 seconds, state the running step and full log path.
- If a phase writes output to a log file, also surface live progress to the terminal:
  - prefer streaming stdout/stderr with `tee` while still saving the full log file
  - if streaming is impossible, print periodic heartbeat updates and show recent log lines
- Before each long benchmark or profile run, print:
  - the config being run
  - the full log path
  - what signal the user should expect next
- During long runs, surface meaningful progress at least every 30-60 seconds.
- After each long run, print:
  - exit code
  - success or failure
  - what comes next

When choosing between `silent log file only` and `visible progress`, prefer visible progress.
