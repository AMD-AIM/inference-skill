# Guided Intake

Use this file to control the user interaction flow.

## Goal

Target UX:

1. User names a model or config key.
2. Agent asks 2-3 short setup rounds (batched when possible).
3. Agent summarizes the plan.
4. Agent starts workflow after confirmation.

The user should never need to manually construct an `opencode inference-optimize ...` command.

## User-facing language rules

- Treat a model name like `qwen3.5-bf16-mi355x-sglang` as enough to start.
- Prefer the user-facing terms `TP`, `sequence length`, `concurrency`, `output path`, and `profiling`.
- Do not expose internal variable names like `FILTER_CONC_START`, `PROFILE`, or `START_PHASE` in the first round.
- Prefer choice-based questions over open-ended prompts.
- Prefer the native `question` tool when available.
- If the runtime has no question tool, ask the same questions in compact numbered form.
- Keep setup tight: avoid more than 3 rounds unless the user explicitly asks to customize everything.
- Batch multi-question rounds into one form (or one numbered message when forms are unavailable).
- Do not split `Run plan`, `Output`, and `GPUs` across separate turns unless batching is impossible.

## Status output contract

Always show the user what stage you are in.

Use short updates like:

- `Status 1/5: target config found. I’ll ask one short setup form next.`
- `Status 2/5: setup choices received. I’m checking which TP, sequence-length, and concurrency options actually exist for this config.`
- `Status 3/5: discovery complete. I’ll show the filter choices now.`
- `Status 4/5: plan ready. One final confirmation and I’ll start the workflow.`
- `Status 5/5: starting execution. I’ll keep reporting progress at each phase boundary.`

Rules:

- Send a status update before any long-running discovery or execution step.
- Keep each update to 1-2 short sentences.
- Include: what finished, what is happening now, and what comes next.
- Do not rely on tool output as the only progress signal.

## Intake algorithm

1. Resolve the target config key from the user text.
2. Send a kickoff status update.
3. Ask the Round 1 high-level question set in one batched form.
4. If the answers require a concrete path or GPU list, ask one short follow-up for that value.
5. Send a status update before discovery starts.
6. Read [`RUNTIME.md`](RUNTIME.md), validate that the resolved config key exists in the selected master YAML, and do only the lightweight discovery needed to learn available TP / EP / sequence / concurrency values.
7. Send a status update when discovery is complete.
8. If this is a smoke-style run, offer a fast path with recommended smoke defaults.
9. Only if the user wants customization, ask the detailed filter question set in one batched form.
10. Summarize the plan and ask for final confirmation.
11. After confirmation, send an execution-start status update and begin work.

## Round 1: exact high-level question set

Unless the user already answered clearly, prefer one multi-question form with these three questions.

Before asking the form, say one natural sentence like:

`I found the target config. I’ll ask one short setup form so we can lock the run shape before I touch the workflow.`

### Question 1

- `header`: `Run plan`
- `question`: `What kind of run should I prepare for this model?`
- `options`:
  - `Smoke full workflow`: `Quick end-to-end run with narrow filters, including profile analysis`
  - `Smoke benchmark only`: `One small benchmark point without profiling`
  - `Benchmark only`: `Run benchmark and benchmark analysis only`
  - `Profile only`: `Run profile and profile analysis only`
  - `Full workflow`: `Run the broader benchmark plus profiling workflow`
  - `Optimize workflow`: `Full workflow including kernel optimization (benchmark, profile, optimize, and report)`
  - `Optimize from existing profile`: `Run kernel optimization using existing profiling data`
  - `Resume existing output`: `Continue or restart from an existing output directory`
  - `Monitor existing run`: `Analyze a completed run's artifacts with full transparency — see everything the quality monitor evaluates`

Map these to:

- `Smoke full workflow` -> `mode=full` with narrow filters
- `Smoke benchmark only` -> `mode=benchmark` with narrow filters
- `Benchmark only` -> `mode=benchmark`
- `Profile only` -> `mode=profile`
- `Full workflow` -> `mode=full`
- `Optimize workflow` -> `mode=optimize`
- `Optimize from existing profile` -> `mode=optimize-only`, ask for existing output path
- `Resume existing output` -> ask for an output path, then ask how to continue
- `Monitor existing run` -> `mode=monitor`, ask for run directory. Skip Questions 2-4 (Framework, Output, GPUs) only if analyzing existing run; if running fresh, keep them.

### Question 2

- `header`: `Framework`
- `question`: `Which serving framework is this config for?`
- `options`:
  - `vLLM`: `Use vLLM as the serving framework`
  - `SGLang`: `Use SGLang as the serving framework`
  - `Auto-detect from config`: `Derive the framework from the config key name or sweep output`

Map these to:
- `vLLM` -> `FRAMEWORK=vllm`
- `SGLang` -> `FRAMEWORK=sglang`
- `Auto-detect from config` -> leave `FRAMEWORK` unset (derived during Phase 1 config parse)

### Question 3

- `header`: `Output`
- `question`: `How should I handle the output directory?`
- `options`:
  - `New timestamped output`: `Create a fresh timestamped output directory`
  - `Resume previous output`: `Continue from an existing output directory`
  - `Custom output path`: `Use a specific output path I provide`

### Question 4

- `header`: `GPUs`
- `question`: `How should I choose GPUs for this run?`
- `options`:
  - `Auto-select GPUs`: `Let the workflow choose the most suitable GPUs`
  - `Use current CUDA/HIP env`: `Respect current CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES`
  - `Specify GPU IDs`: `I will provide explicit GPU device IDs`

## Follow-up questions only when needed

Ask a short follow-up only for missing concrete values:

- If the user chose `Resume previous output` or `Custom output path`, ask for the path.
- If the user chose `Specify GPU IDs`, ask for comma-separated GPU IDs.
- If the user chose `Resume existing output`, ask:
  - `Continue from last progress`
  - `Restart a specific phase`
- If the user chose `Restart a specific phase`, ask for one of:
  - `env`
  - `config`
  - `benchmark`
  - `benchmark-analyze`
  - `profile`
  - `profile-analyze`
  - `problem-generate`
  - `kernel-optimize`
  - `integration`
  - `report-generate`

If multiple concrete values are missing, batch follow-ups into one short message.

## Monitor mode follow-up

If the user chose `Monitor existing run`, skip Questions 2-4 entirely. Ask one follow-up:

- `header`: `Run directory`
- `question`: `Which run should I analyze?`
- `options`:
  - One option per discovered `~/inferencex_*` directory (show config key + date + status)
  - `Custom path`: `I will provide a specific output directory path`

If only one run exists, offer it as the default. Validate the chosen path contains
`progress.json` and `config.json`.

After receiving the run directory, skip discovery, filter questions, and confirmation.
Read `orchestrator/MONITOR-MODE.md` and `orchestrator/ORCHESTRATOR.md`, then begin the monitor dispatch loop.

## Lightweight discovery before filter questions

Before asking about TP / sequence / concurrency:

1. Read [`RUNTIME.md`](RUNTIME.md).
2. Prepare only the minimum bootstrap needed for config discovery.
3. Validate the resolved config key against the selected master YAML. If it is missing, stop and show close matches instead of continuing.
4. Discover available TP / EP / sequence-length / concurrency values for the selected config.
5. Do not start the expensive benchmark or profile run yet.

Discovery exists to turn abstract parameter prompts into concrete options.

## Smoke fast path

If the selected run plan is a smoke run, prefer a fast path before detailed filter questions.

Ask one question:

- `header`: `Filters`
- `question`: `How should I choose TP, sequence length, and concurrency for this smoke run?`
- `options`:
  - `Use recommended smoke defaults`: `Pick one narrow, representative configuration for me`
  - `Review each filter`: `Let me choose TP, sequence length, and concurrency`
  - `Use full discovered sweep`: `Do not narrow the discovered sweep`

If the user chooses `Use recommended smoke defaults`, skip detailed filter questions and move to the final summary.

## Recommended smoke defaults heuristic

When choosing smoke defaults:

- TP:
  - if the user already asked for a specific TP, use it
  - else if only one TP value is discovered, use it
  - else if `8` is available, prefer `8`
  - else prefer the smallest discovered TP greater than `1`
  - else use the smallest discovered TP
- Sequence length:
  - prefer the shortest discovered preset
  - if `1k1k` exists, prefer it
- Concurrency:
  - if `4` exists, prefer it
  - else prefer the smallest discovered positive value
- EP:
  - prefer `1` unless the user requested otherwise

Always show the chosen defaults in the final summary before starting execution.

## Detailed filter question set

If the user chooses `Review each filter`, ask only the questions that matter.

Ask the needed filter questions in one batched form, not as separate turns for TP, then sequence length, then concurrency.

Before asking the form, say one natural sentence like:

`I’ve checked the config and now I can offer concrete filter choices instead of generic parameter questions.`

### TP question

- `header`: `TP`
- `question`: `Which tensor parallelism setting should I use?`
- `options`:
  - `Recommended`: `Use the recommended TP for this run`
  - `All TP values`: `Keep all discovered TP values`
  - one option per discovered TP value
  - `Custom`: `I will describe a custom TP choice`

### Sequence question

- `header`: `Seq len`
- `question`: `Which sequence length preset should I use?`
- `options`:
  - `Recommended`: `Use the recommended smoke/default sequence length`
  - `All sequence lengths`: `Keep all discovered sequence presets`
  - one option per discovered preset such as `1k1k`, `1k8k`, `8k1k`
  - `Custom`: `I will describe a custom sequence choice`

### Concurrency question

- `header`: `Conc`
- `question`: `Which concurrency setting should I use?`
- `options`:
  - `Recommended`: `Use the recommended concurrency`
  - `All discovered values`: `Keep the full discovered concurrency sweep`
  - one option per discovered value when the list is short
  - `Custom range`: `I will provide a concurrency range`

### EP question

Ask only if more than one EP value exists:

- `header`: `EP`
- `question`: `Which expert parallelism setting should I use?`
- `options`:
  - `Recommended`
  - `All EP values`
  - one option per discovered EP value

### Profiling extras

Only if profiling is included, ask the extra choices that materially affect the run:

- `header`: `Profile`
- `question`: `Any profiling-specific options?`
- `options`:
  - `Default profiling settings`
  - `Enable eager mode`
  - `Reuse existing profile artifacts if present`
  - `Always start profile collection fresh`

### Optimization extras

Only if mode is `optimize` or `optimize-only`, ask the extra choices:

- `header`: `Optimize`
- `question`: `What optimization scope should I use?`
- `options`:
  - `Default optimization settings`: `Auto-detect GEAK availability; optimize all bottleneck kernels above threshold; fall back to manual if GEAK is unavailable`
  - `Focus on fused kernels only`: `Only generate fused operator problems (residual+norm, SwiGLU); uses GEAK auto mode`
  - `GEAK full mode`: `Use GEAK for both Triton (simple mode) and HIP/CK (kernel-url mode) kernels`
  - `GEAK Triton only`: `Use GEAK simple mode for Triton/ATen kernels only; skip HIP/CK optimization`
  - `Manual optimization only`: `Write optimized kernels manually without GEAK`
  - `Skip integration benchmark`: `Generate and optimize kernels only, skip E2E benchmark`

Map these to:
- `Default optimization settings` -> `GEAK_MODE=auto`
- `Focus on fused kernels only` -> `GEAK_MODE=auto, OPTIMIZE_SCOPE=fused_only`
- `GEAK full mode` -> `GEAK_MODE=full`
- `GEAK Triton only` -> `GEAK_MODE=triton_only`
- `Manual optimization only` -> `GEAK_MODE=manual`
- `Skip integration benchmark` -> `SKIP_INTEGRATION=true`

### Monitor level

Only if mode is `optimize` or `optimize-only`, optionally ask:

- `header`: `Monitor`
- `question`: `What level of quality monitoring should I use?`
- `options`:
  - `Standard monitoring`: `Review critical phases with quality checks, generic checks for others`
  - `Strict monitoring`: `Apply quality checks to all phases, fail on any warning`
  - `Minimal monitoring`: `Only check that phase results exist, skip quality analysis`

Map these to:
- `Standard monitoring` -> `MONITOR_LEVEL=standard` (default)
- `Strict monitoring` -> `MONITOR_LEVEL=strict`
- `Minimal monitoring` -> `MONITOR_LEVEL=minimal`

## Final confirmation

Before execution, summarize:

- config key
- run plan and mode
- output path behavior
- TP / EP / sequence / concurrency selection
- GPU selection
- profiling behavior

Then ask one final question:

- `header`: `Confirm`
- `question`: `Start the workflow with these settings?`
- `options`:
  - `Start now`
  - `Edit choices`
  - `Cancel`

Before the confirmation question, send a short status update that the plan is ready and this is the last decision before execution.

## Anti-patterns

- Do not begin by asking the user to type the full CLI command.
- Do not ask all internal variables one by one.
- Do not ask detailed profiling questions before the run mode is known.
- Do not ask abstract TP / sequence / concurrency questions before discovery tells you what values are actually available.
- Do not silently widen a smoke run into a full sweep.
- Do not make the user guess progress from silence.
