# Intake: Guided Setup

Controls the user interaction before execution starts.

## Goal

1. User names a model (`Qwen/Qwen3-8B` or `/local/path`).
2. Agent asks 2 short setup rounds (batched).
3. Agent summarizes the plan.
4. Execution begins after confirmation.

---

## Status Messages (use exactly these labels)

```
Status 1/5: model identified — sending setup form now.
Status 2/5: setup received — running GPU discovery.
Status 3/5: discovery complete — sending filter form.
Status 4/5: plan ready — awaiting confirmation.
Status 5/5: confirmed — starting execution.
```

---

## Round 1 — Send on First Turn (before any other file reads)

Ask all three in one batched form:

### Q1: Run plan
- header: `Run plan`
- question: `What should I run for this model?`
- options:
  - `Full optimize (Recommended)` — benchmark + profile + kernel opt + integration + report
  - `Profile only` — benchmark + profile + analysis, no kernel changes
  - `Optimize from existing data` — kernel opt using existing Phase 3 artifacts

Map to:
- Full optimize → `MODE=optimize`
- Profile only → `MODE=profile-only`
- Optimize from existing data → `MODE=optimize-only`, ask follow-up for existing output dir

### Q2: Output directory
- header: `Output`
- question: `Where should I save results?`
- options:
  - `New timestamped dir (Recommended)` — `./vllm_opt_<model>_<timestamp>`
  - `Specify a path` — ask follow-up for the path

### Q3: GPU selection
- header: `GPUs`
- question: `How should I pick GPUs?`
- options:
  - `Auto-select least busy (Recommended)` — run select_gpus.py
  - `Use current CUDA/HIP env` — respect CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES
  - `Specify GPU IDs` — ask follow-up for comma-separated IDs

---

## Follow-ups (only when needed)

- If "Specify a path": ask for the output path.
- If "Specify GPU IDs": ask for comma-separated IDs.
- If "Optimize from existing data": ask for existing output directory.
- If model is a HuggingFace ID and download fails: ask for HF_ENDPOINT or local path.

---

## After Round 1 — Discovery

1. Read [`RUNTIME.md`](RUNTIME.md).
2. Detect GPU count and type via `rocm-smi` or `nvidia-smi`.
3. Do NOT start benchmark yet.

---

## Round 2 — Filter Form

Send all at once after discovery:

### Q4: Tensor parallelism
- header: `TP`
- question: `Which tensor parallelism setting?`
- options: `1 (Recommended for single GPU)`, `2`, `4`, `8`

### Q5: Sequence length
- header: `Seq len`
- question: `Which sequence length preset?`
- options:
  - `1k × 1k (Recommended)` — ISL=1024, OSL=1024
  - `1k × 8k` — ISL=1024, OSL=8192
  - `8k × 1k` — ISL=8192, OSL=1024
  - `Custom` — ask follow-up

### Q6: Concurrency levels
- header: `Concurrency`
- question: `Which concurrency sweep?`
- options:
  - `Full sweep (Recommended)` — 4,8,16,32,64,128
  - `Smoke` — 4,16,64
  - `Custom` — ask follow-up

### Q7 (only if MODE=optimize): Optimization budget
- header: `Opt budget`
- question: `How many optimization attempts per kernel?`
- options:
  - `Default (Recommended)` — 8 attempts, stop after 3 consecutive rejections
  - `Deep` — 15 attempts, stop after 5 rejections
  - `Quick` — 4 attempts, stop after 2 rejections

---

## Confirmation

Before execution, print exactly this summary block (fill in values):

```
Plan summary:
  Model:       {{MODEL}}
  Mode:        {{MODE}}
  Output:      {{OUTPUT_DIR}}
  GPU(s):      {{GPUS}} ({{GPU_VENDOR}} {{GPU_ARCH}})
  TP:          {{TP}}
  ISL × OSL:   {{ISL}} × {{OSL}}
  Concurrency: {{CONCURRENCY_LEVELS}}
  Dtype:       {{DTYPE}}
  Opt budget:  {{MAX_OPTIMIZATION_ATTEMPTS}} attempts / {{MAX_CONSECUTIVE_REJECTIONS}} max rejections
```

Then ask:
- header: `Confirm`
- question: `Start with these settings?`
- options: `Start now`, `Edit choices`, `Cancel`

---

## Network Detection (automatic, no user question)

Before model download, auto-check:
1. Is `HF_ENDPOINT` already set? If yes, use it.
2. Is `huggingface.co` reachable? If no, ask user for proxy URL.
3. Is `HF_HUB_DISABLE_XET` needed? Check hub version.

If model is a local path → skip all HF checks.
