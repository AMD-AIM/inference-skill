# Runtime Notes

Use this file after guided intake is complete and before phase execution starts.

## Bundled assets

Use only the files bundled next to this skill:

- `phases/00-env-setup.md`
- `phases/01-config-parse.md`
- `phases/02-benchmark.md`
- `phases/03-benchmark-analyze.md`
- `phases/04-profile.md`
- `phases/05-profile-analyze.md`
- `templates/agent-config.md`
- `scripts/trace_analyzer.py`
- `scripts/select_gpus.py`
- `resources/TraceLens-internal.tar.gz`

If required files are missing, stop and report that the skill installation is incomplete.

## Required run inputs

Resolve these before loading any phase file:

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

## Placeholder rules

- Replace `{{VAR}}` placeholders in phase docs using the resolved variable map.
- Ignore `{{SKIP_LABEL}}` markers. They are compiler annotations from the built-in OpenCode command path.
- If a phase doc references `{{PROFILE_SKIP_NOTE}}` or `{{PROFILE_ANALYSIS_NOTE}}`, resolve them from the selected mode before execution.

## Workspace bootstrap

Before executing any phase doc:

1. Create `OUTPUT_DIR`, `RESULTS_DIR`, `PROFILE_DIR`, `REPORT_DIR`, and `SCRIPTS_DIR`.
2. Write `config.json` with the resolved run configuration.
3. Write initial `progress.json` if it does not already exist.
4. Copy bundled helper assets into `SCRIPTS_DIR` when needed:
   - `trace_analyzer.py`
   - `select_gpus.py`
   - `TraceLens-internal.tar.gz`

For `resume` or `from-phase`, read the existing `progress.json` and prior artifacts first, but fully rerun the requested starting phase.

## Phase map

- `full`: `env -> config -> benchmark -> benchmark-analyze -> profile -> profile-analyze`
- `benchmark`: `env -> config -> benchmark -> benchmark-analyze`
- `profile`: `env -> config -> profile -> profile-analyze`
- `benchmark+profile`: same phase set as `full`

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

- Before each phase, emit a short update such as:
  - `Running Phase 0/5: Environment Setup`
  - `Running Phase 1/5: Config Parsing`
  - `Running Phase 2/5: Benchmark Execution`
  - `Running Phase 3/5: Benchmark Analysis`
  - `Running Phase 4/5: Profiling`
  - `Running Phase 5/5: Profile Analysis`
- After each phase, emit a short completion update that also says what comes next.
- Before any long benchmark or profile step, say what is about to take time.
- If a step is unexpectedly slow, add another brief update instead of leaving the user in silence.

## Long-running log surfacing

Long phases must not leave the terminal blank.

Rules:

- If a command may run for more than about 30 seconds, tell the user which step is running and where the full log is being written.
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
