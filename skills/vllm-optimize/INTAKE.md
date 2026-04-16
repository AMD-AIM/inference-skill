# Guided Intake

Use this file to control the user interaction flow.

## Goal

Target UX:

1. User names a model (e.g., `Qwen/Qwen3.5-4B` or `/app/MyModel`).
2. Agent asks 2-3 short setup rounds (batched when possible).
3. Agent summarizes the plan.
4. Agent starts workflow after confirmation.

The model can be a HuggingFace ID or a local path. Phase 1 will handle downloading if needed — the user does not need to pre-download.

The user should never need to manually construct a full command.

## User-facing language rules

- Treat a model name like `Qwen/Qwen3.5-4B` as enough to start.
- Prefer the user-facing terms `TP`, `sequence length`, `concurrency`, `output path`.
- Do not expose internal variable names in the first round.
- Prefer choice-based questions over open-ended prompts.
- Prefer the native `question` tool when available.
- Keep setup tight: avoid more than 3 rounds unless the user explicitly asks to customize everything.
- Batch multi-question rounds into one form.

## Status output contract

Always show the user what stage you are in:

- `Status 1/5: target model found. I'll ask one short setup form next.`
- `Status 2/5: setup choices received. Checking available configuration options.`
- `Status 3/5: discovery complete. Showing filter choices now.`
- `Status 4/5: plan ready. One final confirmation and I'll start.`
- `Status 5/5: starting execution. I'll report progress at each phase boundary.`

## Intake algorithm

1. Resolve the target model name from the user text.
2. Send a kickoff status update.
3. Ask the Round 1 high-level question set in one batched form.
4. If the answers require a concrete path or GPU list, ask one short follow-up.
5. Send a status update before discovery starts.
6. Read [`RUNTIME.md`](RUNTIME.md) and do only the lightweight discovery needed.
7. Send a status update when discovery is complete.
8. If this is a smoke-style run, offer a fast path with recommended smoke defaults.
9. Only if the user wants customization, ask the detailed filter question set.
10. Summarize the plan and ask for final confirmation.
11. After confirmation, send an execution-start status update and begin work.

## Round 1: exact high-level question set

### Question 1

- `header`: `Run plan`
- `question`: `What kind of run should I prepare for this model?`
- `options`:
  - `Smoke full workflow`: `Quick end-to-end run with narrow filters, including profile analysis`
  - `Smoke benchmark only`: `One small benchmark point without profiling`
  - `Benchmark only`: `Run benchmark and benchmark analysis only`
  - `Profile only`: `Run profiling and profile analysis only (requires server running)`
  - `Full workflow`: `Run benchmark plus profiling workflow`
  - `Optimize workflow`: `Full workflow including kernel optimization`
  - `Optimize from existing profile`: `Run kernel optimization using existing profiling data`

Map these to:

- `Smoke full workflow` -> `mode=full` with narrow filters
- `Smoke benchmark only` -> `mode=benchmark` with narrow filters
- `Benchmark only` -> `mode=benchmark`
- `Profile only` -> `mode=profile`
- `Full workflow` -> `mode=full`
- `Optimize workflow` -> `mode=optimize`
- `Optimize from existing profile` -> `mode=optimize-only`, ask for existing output path

### Question 2

- `header`: `Output`
- `question`: `How should I handle the output directory?`
- `options`:
  - `New timestamped output`: `Create a fresh timestamped output directory`
  - `Custom output path`: `Use a specific output path I provide`

### Question 3

- `header`: `GPUs`
- `question`: `How should I choose GPUs for this run?`
- `options`:
  - `Auto-select GPUs`: `Let the workflow choose the most suitable GPUs`
  - `Use current CUDA/HIP env`: `Respect current CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES`
  - `Specify GPU IDs`: `I will provide explicit GPU device IDs`

## Follow-up questions only when needed

Ask a short follow-up only for missing concrete values:

- If the user chose `Custom output path`, ask for the path.
- If the user chose `Specify GPU IDs`, ask for comma-separated GPU IDs.

## Network/environment detection (auto, no user question needed)

Before any model download or HuggingFace access, the agent should auto-detect:

1. **HF_ENDPOINT**: Check if `HF_ENDPOINT` env var is already set. If not, test if `huggingface.co` is directly reachable. If not reachable, ask the user for a proxy URL.
2. **HF_HUB_DISABLE_XET**: Check if set. If the HF hub version has XET issues, set it.

If the user provided a local model path (e.g., `/app/MyModel`), no HuggingFace access is needed — skip this.

This detection happens automatically during Phase 1 Step 6 (model download). The variables are set in Phase 1 Step 4 with `${HF_ENDPOINT:-}` defaults. If the user's environment already has these set, nothing extra is needed.

**If model download fails**, the agent should ask the user:
- `header`: `Network`
- `question`: `Model download failed. Do you have a HuggingFace proxy or mirror?`
- `options`:
  - `Provide proxy URL`: `I have an HF_ENDPOINT URL`
  - `Model is already local`: `Point me to the local path`
  - `No proxy needed`: `Retry with direct access`

## Lightweight discovery before filter questions

Before asking about TP / sequence / concurrency:

1. Read [`RUNTIME.md`](RUNTIME.md).
2. Discover available GPU configuration.
3. Do not start the expensive benchmark or profile run yet.

## Smoke fast path

If the selected run plan is a smoke run, offer a fast path:

- `header`: `Filters`
- `question`: `How should I choose TP, sequence length, and concurrency for this smoke run?`
- `options`:
  - `Use recommended smoke defaults`: `Pick one narrow, representative configuration`
  - `Review each filter`: `Let me choose TP, sequence length, and concurrency`

## Recommended smoke defaults heuristic

- TP: if only one value available, use it; else prefer `1` for consumer GPUs, `8` for data center GPUs.
- Sequence length: prefer `1k1k` (ISL=1024, OSL=1024).
- Concurrency: prefer `4` for smoke runs.

## Detailed filter question set

### TP question

- `header`: `TP`
- `question`: `Which tensor parallelism setting should I use?`
- `options`:
  - `Recommended`
  - `1` (single GPU)
  - `2`
  - `4`
  - `8`

### Sequence question

- `header`: `Seq len`
- `question`: `Which sequence length preset should I use?`
- `options`:
  - `Recommended`
  - `1k1k`: `ISL=1024, OSL=1024`
  - `1k8k`: `ISL=1024, OSL=8192`
  - `8k1k`: `ISL=8192, OSL=1024`
  - `Custom`

### Concurrency question

- `header`: `Conc`
- `question`: `Which concurrency levels should I test?`
- `options`:
  - `Recommended`
  - `Smoke`: `4, 16, 64`
  - `Full sweep`: `4, 8, 16, 32, 64, 128`
  - `Custom range`

### Profiling extras (if profiling is included)

- `header`: `Profile`
- `question`: `Any profiling-specific options?`
- `options`:
  - `Default profiling settings`
  - `Enable eager mode`
  - `Reuse existing profile artifacts if present`
  - `Always start profile collection fresh`

### Optimization extras (if mode is optimize or optimize-only)

- `header`: `Optimize`
- `question`: `What optimization scope should I use?`
- `options`:
  - `Default optimization settings (Recommended)`: `Auto-research with Triton; GEAK if available; 8 attempts, stop after 3 consecutive failures`
  - `Deep optimization`: `More aggressive: 15 attempts, stop after 5 consecutive failures`
  - `Quick optimization`: `Fast: 4 attempts, stop after 2 consecutive failures`
  - `GEAK full mode`: `Use GEAK for Triton and HIP/CK kernels (requires GEAK installed)`
  - `Skip integration benchmark`: `Optimize kernels only, skip E2E benchmark`

Map these to:
- `Default` → `MAX_OPTIMIZATION_ATTEMPTS=8, MAX_CONSECUTIVE_REJECTIONS=3`
- `Deep` → `MAX_OPTIMIZATION_ATTEMPTS=15, MAX_CONSECUTIVE_REJECTIONS=5`
- `Quick` → `MAX_OPTIMIZATION_ATTEMPTS=4, MAX_CONSECUTIVE_REJECTIONS=2`
- `GEAK full` → `GEAK_MODE=full`
- `Skip integration` → `SKIP_INTEGRATION=true`

## Final confirmation

Before execution, summarize:

- model name
- run plan and mode
- output path
- TP / sequence / concurrency selection
- GPU selection
- profiling behavior

Then ask one final question:

- `header`: `Confirm`
- `question`: `Start the workflow with these settings?`
- `options`:
  - `Start now`
  - `Edit choices`
  - `Cancel`