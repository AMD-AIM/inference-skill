# Examples

These examples show the intended interaction style.

## Example 1: smoke workflow from only a model/config name

User:

```text
Use inference-optimize skill for qwen3.5-bf16-mi355x-sglang.
```

Assistant:

```text
Status 1/5: target config found. I’ll ask one short setup form next so we can lock the run shape before starting any work.
```

Assistant then asks a short choice-based form:

- `Run plan`: `Smoke full workflow` / `Smoke benchmark only` / `Benchmark only` / `Profile only` / `Full workflow` / `Resume existing output`
- `Output`: `New timestamped output` / `Resume previous output` / `Custom output path`
- `GPUs`: `Auto-select GPUs` / `Use current CUDA/HIP env` / `Specify GPU IDs`

User picks:

- `Smoke full workflow`
- `New timestamped output`
- `Use current CUDA/HIP env`

Assistant:

```text
Status 2/5: setup choices received. I’m checking the available TP, sequence-length, and concurrency options for this config so the next choices are concrete.
```

Assistant then asks:

- `Filters`: `Use recommended smoke defaults` / `Review each filter` / `Use full discovered sweep`

User picks:

- `Use recommended smoke defaults`

Assistant:

```text
Status 3/5: discovery complete. I can now offer a smoke fast path instead of generic filter questions.

Status 4/5: plan ready. One final confirmation and I’ll start the workflow.

Plan summary:
- config: qwen3.5-bf16-mi355x-sglang
- mode: full
- output: new timestamped directory
- filters: recommended smoke defaults
- GPUs: use current CUDA/HIP environment

Start now?
```

Assistant then asks:

- `Confirm`: `Start now` / `Edit choices` / `Cancel`

If the user confirms, Assistant continues with:

```text
Status 5/5: starting execution. I’ll report progress at each phase boundary.
Running Phase 0/5: Environment Setup
```

## Example 2: benchmark-only with manual filter review

User:

```text
Use inference-optimize skill for qwen3.5-bf16-mi355x-sglang, benchmark only.
```

Assistant:

```text
Status 1/5: target config found. I’ll ask one grouped setup form first, then I’ll discover the concrete filter options for benchmark-only mode.
```

Assistant asks the Round 1 form.

After lightweight discovery, Assistant says:

```text
Status 3/5: discovery complete. I can now ask the detailed filter choices in one batched form.
```

Assistant then asks the detailed filter form:

- `TP`: `Recommended` / `All TP values` / discovered TP values / `Custom`
- `Seq len`: `Recommended` / `All sequence lengths` / discovered presets / `Custom`
- `Conc`: `Recommended` / `All discovered values` / discovered values / `Custom range`

User selects:

- `TP=8`
- `Seq len=1k1k`
- `Conc=4`

Assistant:

```text
Status 4/5: plan ready. One final confirmation and I’ll start the workflow.

Plan summary:
- config: qwen3.5-bf16-mi355x-sglang
- mode: benchmark
- TP: 8
- sequence length: 1k1k
- concurrency: 4

Start now?
```

## Example 3: what not to do

Avoid opening with this style:

```text
Please provide CONFIG_KEY, OUTPUT_DIR, REPO_URL, HF_CACHE, FILTER_TP, FILTER_EP, FILTER_CONC_START, FILTER_CONC_END, FILTER_SEQ, GPUS, PROFILE, START_PHASE, and MODE.
```

That is an internal bootstrap interface, not the intended user experience.

Also avoid this drip-feed style:

```text
What mode do you want?
```

```text
Okay. What output path do you want?
```

```text
Okay. Which GPUs?
```

Those should be asked together in one setup form.

## Example 4: multi-agent orchestration (optimize workflow)

User:

```text
Use inference-optimize skill for dsr1-fp4-mi355x-sglang with optimize workflow.
```

Assistant:

The guided setup flow proceeds the same as before (Round 1 setup form, discovery, filter choices, strict monitoring is applied automatically, confirmation). After confirmation:

```text
Status 5/5: starting multi-agent execution.

Dispatching orchestrator → Phase 00 (env-setup)...
  ✓ Phase 00 completed — GPU: gfx950, GEAK: available
Dispatching orchestrator → Phase 01 (config-parse)...
  ✓ Phase 01 completed — 1 sweep config generated
Dispatching orchestrator → Phase 02 (benchmark) [critical]...
  ✓ Phase 02 completed — 1 benchmark run, throughput: 2847 tok/s
  Monitor: PASS — results_exist ✓, runs_completed ✓
Dispatching orchestrator → Phase 03 (benchmark-analyze)...
  ✓ Phase 03 completed — gap analysis: 12 bottleneck kernels
...
Dispatching orchestrator → Phase 07 (kernel-optimize) [critical]...
  ✓ Phase 07 completed — 5/12 kernels improved, best speedup: 1.42x
  Monitor: PASS — compiled_count=5 ✓, best_speedup=1.42 ✓
Dispatching orchestrator → Phase 08 (integration) [critical]...
  ✓ Phase 08 completed — E2E speedup: 1.08x
  Monitor: PASS — plugin_generated ✓, validation passed ✓
Dispatching orchestrator → Phase 09 (report-generate)...
  ✓ Phase 09 completed — report at report/optimization_report.md

All 10 phases completed. Final E2E speedup: 1.08x
Report: ~/inference_dsr1-fp4-mi355x-sglang_20260413_100000/report/optimization_report.md
```

> In multi-agent mode, each phase runs as an isolated agent with bounded context. The orchestrator dispatches phases sequentially, and the monitor validates critical phases before proceeding. If a critical phase fails, the orchestrator reruns it (up to 2 times) before escalating.

## Example 5: monitor workflow with fixed strict monitoring

User:

```text
Use inference-optimize skill for dsr1-fp4-mi355x-sglang with monitor workflow.
```

Assistant asks the Round 1 setup form (including `Run plan`, `Framework`, `Output`, and `GPUs`).

User selects:

- `Run plan`: `Monitor workflow`
- `Framework`: `SGLang`
- `Output`: `New timestamped output`
- `GPUs`: `Auto-select GPUs`

Assistant performs lightweight discovery, asks filter choices as needed, and applies strict monitoring automatically for monitor workflow.

Assistant:

```text
Status 4/5: plan ready. One final confirmation and I’ll start the workflow.

Plan summary:
- config: dsr1-fp4-mi355x-sglang
- mode: monitor
- framework: sglang
- output: new timestamped directory
- monitoring: strict (fixed)
- filters: TP=8, seq=1k1k, conc=4

Start now?
```

Assistant then asks:

- `Confirm`: `Start now` / `Edit choices` / `Cancel`
