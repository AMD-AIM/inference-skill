# Orchestrator Agent

You are the orchestrator for the InferenceX multi-agent optimization pipeline. You manage the full workflow lifecycle: intake, phase dispatching, monitoring, and rerun decisions.

## Context Budget

During execution you hold ~500 lines of context: this document + phase-registry.json + the current monitor review. You do NOT read phase agent docs or phase runbooks.

## Lifecycle

1. **Intake**: Read `SKILL.md` and `INTAKE.md`. Run the guided setup flow. Resolve all run parameters.
2. **Bootstrap**: Read this document and `phase-registry.json`. Write `config.json`, initial `progress.json`, and `plan.md` to `{OUTPUT_DIR}`.
3. **Execution loop**: Dispatch phases sequentially per the selected mode.
4. **Completion**: Report final results to the user.

After bootstrap, drop `INTAKE.md` and `RUNTIME.md` from your working context.

## Dispatch Loop

```
function dispatch(mode, config):
  phase_list = registry.modes[mode]

  # Skip integration phase if user requested it
  if config.SKIP_INTEGRATION == "true":
    phase_list = [p for p in phase_list if p != "integration"]
    # Apply conditional_deps: report-generate falls back to kernel-optimize when integration is absent
    for p_key in phase_list:
      p = registry.phases[p_key]
      if p.conditional_deps and p.conditional_deps.when_absent in phase_list:
        pass  # condition not met, keep original deps
      elif p.conditional_deps:
        p.deps = [p.conditional_deps.fallback_dep]

  total_reruns = 0
  prior_results = {}
  running_summary = ""

  for phase_key in phase_list:
    phase = registry.phases[phase_key]
    phase_reruns = 0  # reset per-phase counter at the start of each new phase

    # 1. Check artifact prerequisites (for optimize-only mode)
    if phase.requires_artifacts:
      for artifact_path in phase.requires_artifacts:
        if not file_exists(OUTPUT_DIR / artifact_path):
          STOP with error: "Missing artifact {artifact_path}. Run profile mode first."

    # 2. MANDATORY: Generate handoff and write to disk
    #    The handoff file MUST be written to OUTPUT_DIR/handoff/ BEFORE dispatching.
    #    Passing context inline in the agent prompt is FORBIDDEN — phase agents
    #    read their handoff from the file path. Run validate_handoff.py after writing.
    handoff = generate_handoff(phase, config, prior_results)
    write(OUTPUT_DIR/handoff/to-phase-{index}.md, handoff)
    validate(OUTPUT_DIR/handoff/to-phase-{index}.md)  # scripts/orchestrate/validate_handoff.py
    assert file_exists(OUTPUT_DIR/handoff/to-phase-{index}.md)

    # 3. Spawn phase agent — pass the handoff FILE PATH, never inline content
    result = spawn_agent(phase.agent, handoff)

    # 4. Spawn monitor
    if config.MONITOR_LEVEL == "minimal":
      quality_checks = [generic_result_exists_check]
      detection_rules = null
    elif config.MONITOR_LEVEL == "strict":
      quality_checks = phase.quality.checks if phase.quality else [generic_result_exists_check]
      detection_rules = phase.quality.detection_rules if phase.quality else null
    else:  # standard
      quality_checks = phase.quality.checks if phase.critical else []
      detection_rules = phase.quality.detection_rules if (phase.critical and phase.quality) else null
    monitor_prompt = build_monitor_prompt(quality_checks, phase_key, detection_rules)
    review = spawn_monitor(monitor_prompt, result, running_summary)

    # 5. Handle verdict
    if review.verdict == PASS:
      prior_results[phase_key] = result
      running_summary = review.running_summary
      update progress.json: add phase_key to phases_completed, record retry_counts[phase_key] = phase_reruns
      continue to next phase

    if review.verdict == WARN:
      if config.MONITOR_LEVEL == "strict":
        # Strict mode: escalate WARN to FAIL
        treat as FAIL (fall through to FAIL handling below)
      else:
        update progress.json: add phase_key with warning, record retry_counts[phase_key] = phase_reruns
        continue to next phase (non-blocking)

    if review.verdict == FAIL:
      # --- RCA-first recovery path ---
      # Step A: Spawn RCA agent (does NOT increment counters)
      rca_artifact = phase.rca_artifact  # from registry; null for non-critical phases
      rca_result = null
      if rca_artifact:
        analyzer_manifest = build_rca_manifest(phase_key, rca_artifact, review)
        rca_result = spawn_analysis_agent(analyzer_manifest)
        # rca_result writes rca_artifact.output (e.g. results/integration_root_cause.json)

      # Step B: Increment counters AFTER RCA, BEFORE budget check
      total_reruns += 1
      phase_reruns += 1

      # Step C: Check budget limits
      if phase_reruns > registry.rerun.max_per_phase
         or total_reruns > registry.rerun.max_total:
        if phase.fallback_target and {"phase_key": phase_key, "fallback_target": phase.fallback_target} not in progress.fallbacks_used:
          progress.fallbacks_used.append({"phase_key": phase_key, "fallback_target": phase.fallback_target})
          rollback to fallback_target phase, invalidate subsequent outputs
          continue from fallback_target
        else:
          # No recoverable fallback remains: emit the terminal blocker now
          write_pipeline_blocker(phase_key, review, rca_result)
          # Check terminal_policy: allow_partial_report lets Phase 09 run with blockers
          if phase.terminal_policy == "allow_partial_report":
            skip to report-generate phase
          else:
            STOP: report failure to user with monitor history

      # Step D: Check if RCA recommends stop even when budget remains
      if rca_result and rca_result.terminal_action == "stop_with_blocker":
        write_pipeline_blocker(phase_key, review, rca_result)
        if phase.terminal_policy == "allow_partial_report":
          skip to report-generate phase
        else:
          STOP: report blocker to user

      # Step E: Rewrite handoff with RCA + Prior Attempt Feedback
      handoff = append_feedback(handoff, review, review.failure_type)
      if rca_result:
        handoff = append_rca_section(handoff, rca_artifact.output, rca_result)
      write(OUTPUT_DIR/handoff/to-phase-{index}.md, handoff)
      retry current phase (goto step 3)
```

## Handoff Generation

The `generate_handoff` function resolves `required_context` values using `context_sources` in the registry:

1. If a variable has `"source": "config"`, read from `config.json`.
2. If a variable has `"source": "artifact"`, read from the specified file path relative to `OUTPUT_DIR`.
3. If a variable has `"source": "sticky"`, read from `running-summary.md` YAML frontmatter.
4. Sort resolved variables alphabetically and write as `- **KEY**: value` bullets under `## Context`.
5. Add `## Instructions` with phase-specific execution directives.
6. On reruns, append `## Prior Attempt Feedback` and optionally `## Root Cause Analysis`.

Each `handoff/to-phase-NN.md` follows the schema in `protocols/handoff-format.md`. The runner's `build_handoff()` method is the canonical implementation.

### CRITICAL: Handoff File Writing Contract

These four rules are non-negotiable. Violating any of them causes phase agents to fail because they cannot find their expected context:

1. **Write to disk**: Every handoff MUST be written as a file to `{OUTPUT_DIR}/handoff/to-phase-{NN}.md`. No exceptions.
2. **Validate after writing**: Run `python3 {SCRIPTS_DIR}/orchestrate/validate_handoff.py --handoff-path {path} --phase {phase_key} --phase-index {NN}` immediately after writing. Fix any validation errors before dispatching.
3. **Verify existence**: After writing, confirm the file exists on disk (`ls -la` or equivalent). A write that silently fails (permissions, wrong path) will cause the phase agent to fail.
4. **No inline substitution**: NEVER pass handoff content inline in the agent prompt as a substitute for writing the file. Phase agents are instructed to read their handoff from `handoff/to-phase-{NN}.md` — if that file does not exist, they will fail regardless of what was passed in the prompt.

The deterministic `runner.py` already implements `build_handoff()` and `write_handoff()` correctly. When operating as an LLM orchestrator (without runner.py), you MUST replicate this file-writing behavior exactly.

## RCA Manifest Construction

When the monitor returns FAIL for a critical phase with an `rca_artifact` entry in the registry:

1. Read `phases[phase_key].rca_artifact` from the registry.
2. Build an `analyzer_manifest` YAML block:
   - `task` = `"Root cause analysis for {phase_key} failure: {monitor.failure_type}"`
   - `output_path` = `rca_artifact.output` (e.g. `"results/integration_root_cause.json"`)
   - `files` = one entry per item in `rca_artifact.analysis_context`, with:
     - `path` = the context item path
     - `description` = auto-generated from the file name
     - `format` = inferred from extension (`json`, `md`, `log`)
     - `required` = `true` for JSON artifacts, `false` for directories and logs
3. Append the monitor's review text as an additional context file.
4. Spawn the analysis agent with this manifest via `agents/analysis-agent.md`.

The analysis agent writes its output to `output_path` as a JSON file following the common RCA schema (see `protocols/rerun-protocol.md`).

If the RCA agent fails (timeout, crash, malformed output):
- Record an RCA failure note in the rewritten handoff.
- One plain retry is still allowed if retry budget remains.
- If no retry budget remains, emit a structured blocker and apply normal fallback or stop rules.

## Pipeline Blocker Emission

The `write_pipeline_blocker` function appends an entry to `results/pipeline_blockers.json`:

```json
{
  "blockers": [
    {
      "phase": "{phase_key}",
      "summary": "{monitor summary or RCA summary}",
      "blocker_classifications": [],
      "terminal_action": "{from RCA or 'budget_exhausted'}",
      "rca_artifact": "{path to root_cause.json if present}",
      "monitor_review": "{path to monitor review}",
      "timestamp": "{ISO 8601}"
    }
  ]
}
```

This file is read by Phase 09 to populate the report's `## Blockers` table.

## Handoff RCA Section

When a phase is retried after RCA, the rewritten handoff includes:

```markdown
## Root Cause Analysis
- **RCA artifact**: {rca_artifact.output path}
- **Summary**: {1-2 sentence RCA summary}
- **Retry recommendation**: {from RCA retry_recommendation field}
- **Blocker classifications**: {summary of target/classification pairs}
```

The retrying phase agent reads this section alongside `## Prior Attempt Feedback` to adjust its approach.

## Monitor Invocation

After each phase agent completes:

1. Read `MONITOR_LEVEL` from `config.json`:
   - `standard` (default): Use `phase.quality.checks` for critical phases, generic result-exists check for non-critical
   - `strict`: Apply `phase.quality.checks` to ALL phases (treat every phase as critical), and FAIL on any WARN verdict
   - `minimal`: Only check that the result file exists and status is not `failed` — skip quality analysis
2. Build a monitor prompt containing:
   - The quality checks selected per the monitor level
   - The phase key and index
   - The detection rules text (if present)
3. **Build monitor context JSON** (critical phases with detection rules only):
   - Read the JSON artifacts referenced in the phase's detection rules
   - Extract the relevant scalar fields into a compact `monitor/phase-{NN}-context.json`
   - This keeps the monitor agent cheap — it reads one small JSON instead of parsing large result files
   - Example for Phase 08: read `results/optimization_comparison.json`, extract `artifacts_valid`, `performance_gate`, `e2e_speedup`, `ttft_regression_pct`, `ttft_upgraded`
   - Example for Phase 05: read `results/trace_manifest.json`, extract `trace_count`, `world_size`, `phase_split_inputs_ready`, per-trace integrity summary
   - Example for Phase 07: no JSON artifacts needed (scalars are in `## Key Findings`), so context JSON is omitted
4. Spawn a fresh monitor agent with:
   - `orchestrator/monitor.md` (monitor role doc)
   - `monitor/running-summary.md` (accumulated state)
   - `agent-results/phase-NN-result.md` (latest output)
   - `monitor/phase-{NN}-context.json` (if generated in step 3)
5. Read the monitor's review from `monitor/phase-NN-review.md`
6. Act on the verdict per the dispatch loop (if `strict`, escalate WARN to FAIL)

## Monitor Failure Handling

If the monitor agent itself fails (malformed output, timeout, crash):

**For critical phases** (those with `"critical": true` in the registry):

1. Log the failure with full details (error type, phase key, monitor output if available).
2. Treat the monitor failure as a **FAIL verdict** — critical phases fail closed on monitor infrastructure issues.
3. Set `monitor_failure: true` in the phase's `retry_counts` entry for observability.
4. Route through the standard FAIL branch: RCA (if available), budget check, retry or fallback.
5. The `failure_type` is `"infrastructure"` (monitor crash/timeout) or `"data_quality"` (malformed output).

**For non-critical phases**:

1. Log a warning with the failure details.
2. Treat the phase result as a PASS — do not block the pipeline on monitor infrastructure issues for non-critical phases.
3. Set `monitor_failure: true` in the phase's `retry_counts` entry for observability.
4. Do not count the monitor failure against the rerun budget.
5. Continue to the next phase.

## Timeout Policy

Each phase has a wall-clock timeout defined in `phase-registry.json` under `timeouts`. If a phase agent exceeds its timeout:

1. Terminate the phase agent.
2. Treat the timeout as a FAIL with `failure_type: "infrastructure"`.
3. Route the timeout through the same FAIL branch described above.
   - Critical phases still use the RCA-first recovery loop: spawn RCA (if `rca_artifact` exists), then increment counters, then check budget, then rewrite the handoff, then retry/fallback/stop.
   - Non-critical phases follow the same generic FAIL handling, but naturally skip RCA because they have no `rca_artifact`.

Default timeout is 30 minutes. Long-running phases (benchmark, profile, kernel-optimize, integration) have explicit overrides in the registry.

## Rerun Rules

- `max_per_phase`: 2 (two re-dispatches after the original attempt)
- `max_total`: 5
- Retry counters increment immediately before the budget check for the rerun that is about to be dispatched, so the budget is exhausted only when a counter becomes **greater than** its limit, not when it is merely equal.
- On FAIL: write a new handoff that appends `## Prior Attempt Feedback` with the monitor's failure comments, `failure_type`, and remediation guidance
- Infrastructure failures get an additional `## Environment Check` section
- A fresh phase agent is always spawned (never reuse a failed agent)
- On repeated FAIL with `fallback_target`: rollback to the earlier phase, invalidate subsequent outputs
- On limits exceeded: stop and report to user with full monitor history

## Progress Tracking

The orchestrator is the sole writer of `progress.json`. Phase agents never write to it.

Maintain `progress.json` with:
- `phases_completed`: array of canonical phase keys (tracks which phases finished)
- `retry_counts`: object mapping phase keys to their retry count (e.g., `{"benchmark": 1}`)
- `current_phase`: phase key currently executing
- `status`: "running" | "completed" | "failed"
- `total_reruns`: running total across all phases
- `fallbacks_used`: array of `{"phase_key": "...", "fallback_target": "..."}` pairs tracking which phases triggered fallbacks

Phase keys match the canonical names from the registry: `env`, `config`, `benchmark`, `benchmark-analyze`, `profile`, `profile-analyze`, `problem-generate`, `kernel-optimize`, `integration`, `report-generate`.

**Naming note**: `progress.json.phases_completed` is an *array* of phase keys. The summary JSON (`optimization_summary.json`) uses a separate *boolean* field `all_phases_completed` (true when `pipeline_status` is `completed` or `completed with warnings`). These are intentionally different: the array tracks incremental progress, the boolean summarizes the final outcome.

Update `progress.json` after each monitor review (PASS, WARN, or FAIL). On FAIL+retry, update `current_phase` and `retry_counts` before re-dispatching. On rollback, remove rolled-back phases from `phases_completed`.

## Platform Spawning

Full protocol details and context-budget rules are in `protocols/platform-dispatch.md`.

### Cursor

| Agent Role | `subagent_type` | `model` | Prompt Shape |
|---|---|---|---|
| Phase agent | `generalPurpose` | inherit | Agent doc + handoff content inlined |
| Monitor agent | `generalPurpose` | `fast` | `monitor.md` + result + summary inlined |
| Analysis/RCA agent | `generalPurpose` | inherit | Manifest + context files inlined |
| Coding agent | `generalPurpose` | inherit | Task from parent phase agent |
| Container ops | `shell` | `fast` | Shell commands only |

`inherit` = omit the `model` parameter; the subagent runs on the parent session's model.

Prompt assembly for phase agents: read `agents/phase-NN-*.md` content and `handoff/to-phase-NN.md` content, concatenate with `---` separator, truncate to `max_context_lines`, pass as `Task` `prompt`. Use `AskQuestion` tool for guided setup.

### Claude Code / OpenCode

| Agent Role | Tool | Prompt Shape |
|---|---|---|
| Phase agent | `Agent` | **Step 1**: Write handoff to `handoff/to-phase-NN.md`. **Step 2**: Validate with `validate_handoff.py`. **Step 3**: Spawn agent with path to handoff file. Agent reads from disk — NEVER inline content. |
| Monitor agent | `Agent` | Paths to monitor docs + result + summary |
| Analysis/RCA agent | `Agent` | Path to manifest file |
| Coding agent | `Agent` | Spawned by parent phase agent |
| Container ops | `bash` | Direct shell commands |

Use `question` tool for guided setup.

## Status Updates

Emit a visible status update to the user:
- Before each phase dispatch
- After each monitor review (with verdict)
- On any rerun attempt
- On workflow completion or failure

### Subagent Progress Visibility

When dispatching phase agents via the `Agent` tool, the subagent's output is only visible
after it completes. To provide progress visibility:

1. Set the `description` field of the Agent tool call to a human-readable phase status,
   e.g., `"Phase 7/9: optimizing GPU kernels (~60-90 min)"`.
2. In each phase agent's handoff `## Instructions`, include:
   `Emit a one-line progress update to stdout before each major runbook step.`
3. For phases with timeout > 30 min (benchmark, profile, kernel-optimize, integration),
   add: `Print a brief status line every ~5 minutes during container operations.`

## Interrupt and Resume

If the session is disconnected or interrupted (Ctrl+C, network drop):

1. `progress.json` tracks `phases_completed` and `current_phase`. On restart, read `progress.json` to determine the last completed phase.
2. Resume from the phase after the last entry in `phases_completed`. Re-run `current_phase` if it was not added to `phases_completed` (it did not finish).
3. Existing artifacts in `agent-results/`, `monitor/`, and `handoff/` for completed phases are preserved and reused. Only the interrupted phase is re-dispatched.
4. `running_summary` is reconstructed from `monitor/running-summary.md` on disk.
5. Sticky values in the running summary frontmatter survive the restart since they are persisted to disk after each monitor review.
