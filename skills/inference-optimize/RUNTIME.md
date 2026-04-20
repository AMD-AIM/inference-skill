# Runtime Notes

Use this file after guided intake and before phase execution.

## Bundled assets

Use only the files bundled next to this skill:

**Phase docs:**
- `agents/phase-00-env-setup.md` through `agents/phase-09-report-generate.md`

**Scripts (organized by category):**

`scripts/env/`:
- `validate_config_key.py` — Master YAML config-key validation (Phases 0, 1)
- `detect_gpu_arch.py` — GPU arch detection for roofline (Phase 5)
- `generate_env_info.py` — GPU/GEAK/API key detection (Phase 0)
- `select_gpus.py` — Least-loaded GPU selection (Phases 2, 4, 7, 8)

`scripts/container/`:
- `start_profile_container.sh` — Docker container creation with `--mode benchmark|profile|optimize` (Phases 2, 4, 7, 8)
- `run_profile_exec.sh` — Docker exec with heartbeats + trace flush (Phases 2, 4, 8)
- `collect_profile_traces.sh` — Trace/result collection (Phases 2, 4)

`scripts/profiling/`:
- `inject_profiler_config.py` — Profiler config injection (Phase 4)
- `patch_rank0_profiling.py` — Rank-0-only trace export (Phase 4)
- `patch_benchmark_lib.py` — Relay/cap disable (Phase 4)
- `trace_analyzer.py` — Chrome-format trace parser for gap analysis (Phase 5)
- `validate_traces.py` — Trace discovery and validation (Phase 5)
- `install_tracelens.sh` — TraceLens installation (Phase 5)
- `run_tracelens.sh` — TraceLens analysis pipeline (Phase 5)
- `display_tracelens_results.sh` — Console result display (Phase 5)

`scripts/optimize/`:
- `classify_kernel.py` — Kernel type classification (Phase 6)
- `analyze_fusion.py` — Fusion opportunity detection (Phase 6)
- `extract_model_shapes.py` — Model shape extraction (Phase 6)
- `generate_problems.py` — GEAK problem file generation (Phase 6)
- `resolve_geak_mode.py` — GEAK mode resolution (Phase 7)
- `load_optimization_manifest.py` — Manifest loading/filtering (Phase 7)
- `kernel_test_runner.py` — Kernel correctness/benchmark testing (Phase 7)
- `kernel_finalize.py` — Best kernel finalization (Phase 7)
- `collect_winning_kernels.py` — Kernel collection (Phase 7)
- `verify_winning_kernels.py` — Pre-integration verification (Phase 8)

`scripts/plugin/`:
- `generate_vllm_plugin.py` — vLLM plugin generation (Phase 8)
- `generate_sglang_plugin.py` — SGLang plugin generation (Phase 8)
- `inject_plugin.py` — Plugin injection into benchmark (Phase 8)

`scripts/orchestrate/`:
- `predicate_engine.py` — Structured detection rule evaluation (monitor)
- `validate_handoff.py` — Handoff file validation (orchestrator)

`scripts/report/`:
- `validate_optimization.py` — Baseline vs optimized comparison (Phase 8)
- `integration_outcome.py` — Shared outcome derivation (imported by validate_optimization.py and generate_optimization_summary.py)
- `generate_optimization_summary.py` — Summary JSON generation (Phase 9)

**Templates:**
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
- `MODE`
- `PROBLEMS_DIR`
- `OPTIMIZED_DIR`
- `OPTIMIZE_PRIORITY_THRESHOLD`
- `GEAK_DIR`
- `GEAK_OE_DIR`
- `ENV_INFO_FILE`
- `GEAK_MODE` (auto, full, triton_only, manual; from INTAKE optimization extras)
- `OPTIMIZE_SCOPE` (all, fused_only; from INTAKE optimization extras)
- `MONITOR_LEVEL` (strict by default for skill-guided `optimize`, `optimize-only`, and `monitor` runs; optional manual override for advanced configs)
- `SKIP_INTEGRATION` (true/false; from INTAKE "Skip integration benchmark" option)
- `USE_RUNNER` (set to `true` for skill-guided runs; set `false` only for explicit rollback/debugging)
- `RESOURCES_DIR` (path to installed skill's `resources/` directory; contains `TraceLens-internal.tar.gz`)
- `TEMPLATES_DIR` (path where bundled templates are copied during bootstrap)
- `ENFORCE_EAGER_FLAG` (set to `--enforce-eager` when eager mode is required for the framework, or empty string otherwise; resolved by the agent based on framework and profiling requirements)

## Recommended defaults

- `OUTPUT_DIR`: `./inference_<config-key>_<timestamp>`
- `REPO_DIR`: `<OUTPUT_DIR>/benchmark_repo`
- `REPO_URL`: `https://github.com/SemiAnalysisAI/InferenceX.git`
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
- `MONITOR_LEVEL`: `strict` (default for skill-guided runs; quality checks on all phases, fail on any warning. Advanced/manual configs may still set `standard` or `minimal`)
- `SKIP_INTEGRATION`: `false` (when `true`, skip Phase 8 integration benchmark — only generate optimized kernels and plugins)
- `USE_RUNNER`: `true` (default for skill-guided runs; set `false` only for explicit rollback/debugging)
- `RESOURCES_DIR`: `<installed_skill_root>/resources` (the `resources/` directory alongside `SKILL.md`; not copied to OUTPUT_DIR due to tarball size -- must remain accessible during execution)
- `TEMPLATES_DIR`: `<OUTPUT_DIR>/templates`
- `ENFORCE_EAGER_FLAG`: `""` (empty; set to `--enforce-eager` by the agent when the framework requires eager mode for profiling)

## Placeholder rules

- Replace `{{VAR}}` placeholders in phase docs using the resolved variable map.
- Ignore `{{SKIP_LABEL}}` markers. They are compiler annotations from the built-in OpenCode command path.
- If a phase doc references `{{DRY_RUN_NOTE}}`, resolve it from `DRY_RUN`: empty for real runs, or a short note that benchmark/profile commands should be printed and validated without being executed.

## Workspace bootstrap

Before executing any phase doc:

1. Create `OUTPUT_DIR`, `RESULTS_DIR`, `PROFILE_DIR`, `REPORT_DIR`, `SCRIPTS_DIR`, `TEMPLATES_DIR`, `PROBLEMS_DIR`, `OPTIMIZED_DIR`, and the agent communication directories: `handoff/`, `agent-results/`, `monitor/`.
2. Write `config.json` with the resolved run configuration.
3. Write initial `progress.json` if it does not already exist.
4. Copy ALL bundled scripts (preserving subdirectory structure: `env/`, `container/`, `profiling/`, `optimize/`, `plugin/`, `report/`) into `SCRIPTS_DIR` and ALL bundled templates into `TEMPLATES_DIR`. Phase agents reference scripts via `{{SCRIPTS_DIR}}/env/`, `{{SCRIPTS_DIR}}/container/`, etc. `RESOURCES_DIR` points to the installed skill's `resources/` directory in place (not copied); it must remain accessible during Phase 5 TraceLens installation.

For `resume` or `from-phase`, read existing `progress.json` and artifacts first, then fully rerun the requested start phase. When scanning for existing output directories, check both the `inference_` prefix (current) and the legacy `inferencex_` prefix so that older runs can still be resumed.

## Phase map

- `full`: `env -> config -> benchmark -> benchmark-analyze -> profile -> profile-analyze`
- `benchmark`: `env -> config -> benchmark -> benchmark-analyze`
- `profile`: `env -> config -> profile -> profile-analyze`
- `optimize`: `env -> config -> benchmark -> benchmark-analyze -> profile -> profile-analyze -> problem-generate -> kernel-optimize -> integration -> report-generate`
- `optimize-only`: `env -> config -> problem-generate -> kernel-optimize -> integration -> report-generate` (requires existing `gap_analysis.json` from a prior profile run)

Read only the phase docs needed for the selected mode and start phase.

## Execution guardrails

- Execute phases in order.
- The orchestrator updates `progress.json` after each phase's monitor review — phase agents do not write to it.
- Save all outputs under `OUTPUT_DIR`.
- Do not modify `/opt` or `/usr`.
- Only patch the cloned benchmark repo temporarily when profiling requires it; restore patched files afterward.
- If filter application produces zero configs, stop immediately with a clear error.
- If profiling artifacts are root-owned, clean them up and continue instead of leaving the run half-complete.
- Prefer measured data over assumptions when writing reports.

## Canonical runtime path

The **deterministic runner** (`scripts/orchestrate/runner.py`) is the canonical orchestration path. It handles all mechanical work: mode resolution, dependency checks, artifact prerequisites, context-source resolution, context-budget enforcement, retry budgets, fallback invalidation, handoff generation, atomic `progress.json` writes, and parity artifact emission.

Set `USE_RUNNER=false` in the run configuration to revert to the legacy LLM orchestrator path. The legacy path remains fully supported.

Architecture details: `docs/ARCHITECTURE.md`. Parity verification: `docs/PARITY_CONTRACT.md`.

## Multi-agent orchestration

When running in multi-agent mode, the orchestrator manages the execution loop:

- Read `orchestrator/ORCHESTRATOR.md` for the dispatch loop protocol
- Read `orchestrator/phase-registry.json` for phase metadata, mode maps, and quality criteria
- Write `handoff/to-phase-NN.md` for each phase agent
- Spawn phase agents that read their own `agents/phase-NN-*.md`
- Spawn monitor agents after each phase per `orchestrator/monitor.md`
- Monitor writes reviews to `monitor/phase-NN-review.md` and updates `monitor/running-summary.md`
- Phase agents write results to `agent-results/phase-NN-result.md`
- Communication schemas are defined in `protocols/`

## Platform dispatch

The runner accepts platform-specific callbacks (`dispatch_fn`, `monitor_fn`, `rca_fn`). See `protocols/platform-dispatch.md` for the full adapter contract.

- **Cursor**: `Task` tool with `subagent_type` per dispatch table. Handoff content is inlined in the prompt. Use `AskQuestion` for guided setup.
- **Claude Code / OpenCode**: `Agent` tool with file-path handoffs. Use `question` tool for guided setup.

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
