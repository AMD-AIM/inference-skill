# Examples

These examples show the intended interaction style.

## Example 1: smoke workflow from a model name

User:

```text
use vllm-optimize skill for Qwen/Qwen3.5-4B
```

Assistant:

```text
Status 1/5: target model found. I'll ask one short setup form next.
```

Assistant then asks a short choice-based form:

- `Run plan`: `Smoke full workflow` / `Benchmark only` / `Profile only` / `Full workflow` / `Optimize workflow`
- `Output`: `New timestamped output` / `Custom output path`
- `GPUs`: `Auto-select GPUs` / `Use current CUDA/HIP env` / `Specify GPU IDs`

User picks:

- `Smoke full workflow`
- `New timestamped output`
- `Auto-select GPUs`

Assistant:

```text
Status 2/5: setup choices received. Checking available configuration options.
```

Assistant then asks:

- `Filters`: `Use recommended smoke defaults` / `Review each filter`

User picks:

- `Use recommended smoke defaults`

Assistant:

```text
Status 3/5: discovery complete. Smoke defaults selected.

Status 4/5: plan ready. One final confirmation.

Plan summary:
- model: Qwen/Qwen3.5-4B
- mode: full
- output: new timestamped directory
- filters: recommended smoke defaults (TP=1, ISL=1024, OSL=1024, conc=4)
- GPUs: auto-select

Start now?
```

## Example 2: optimize workflow

User:

```text
use vllm-optimize skill for Qwen/Qwen3.5-4B, optimize mode
```

Assistant asks the Round 1 form plus optimization extras after Round 2.

User selects:

- `Optimize workflow`
- `New timestamped output`
- `Auto-select GPUs`
- `Default optimization settings`

Assistant confirms plan and starts the full 10-phase optimization pipeline.

## Example 3: what not to do

Avoid opening with this style:

```text
Please provide MODEL, OUTPUT_DIR, TP, ISL, OSL, CONCURRENCY, PRECISION, GPUS, PROFILE, START_PHASE, and MODE.
```

That is an internal bootstrap interface, not the intended user experience.

Also avoid asking one question at a time:

```text
What mode do you want?
```

```text
Okay. What output path?
```

```text
Okay. Which GPUs?
```

Those should be asked together in one setup form.