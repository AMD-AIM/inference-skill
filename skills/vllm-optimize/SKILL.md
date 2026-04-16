---
name: vllm-optimize
description: "Run vLLM benchmark, profiling, and kernel optimization workflow for AMD and NVIDIA GPUs. Provides automated model loading, concurrency sweep, torch profiling, GPU kernel analysis, and kernel optimization with GEAK support."
compatibility: claude-code, opencode
metadata:
  workflow: vllm
  audience: performance-engineers
  distribution: standalone-skill-repo
---

# vLLM Optimize

Automated vLLM inference benchmark, profiling, and kernel optimization workflow. Works with AMD GPUs (MI355X, MI300X, R7900/RDNA3) and NVIDIA GPUs.

## Default user experience

- Treat a bare model name as enough to start. Do not require the user to spell out a full command.
- If the user says `use vllm-optimize skill for <model>`, start guided setup immediately.
- Prefer the native `question` tool for multiple-choice prompts when available.
- Ask questions in grouped batches, not as a drip-feed.
- Keep explicit progress updates so the user always knows current stage and next step.
- Inform user about GEAK availability during setup when running in optimize mode.

## First-turn latency rule

- Do not read any other file before the first visible reply unless the model name is ambiguous.
- Send one short kickoff status update.
- Ask the first grouped setup form.

## Guided setup flow

1. Resolve the target model name from the user's input.
2. Start with one short high-level question round:
   - `Run plan`
   - `Output`
   - `GPUs`
3. After Round 1 answers, read [`INTAKE.md`](INTAKE.md) for deeper config.
4. Read [`RUNTIME.md`](RUNTIME.md) only when about to do discovery or execution.
5. Ask high-level setup questions first, then do lightweight discovery, then ask filter-specific questions.
6. Summarize the final plan and get a clear go/no-go before executing.
7. Only after confirmation, read the needed phase docs and start execution.

## Status contract

- Before the first question round, send one short kickoff status message.
- After every major stage transition, send one short progress update.
- During long benchmark or profile runs, stream live output and emit heartbeat updates.
- Keep status updates human-readable.

## Modes

- `full`: `env -> vllm-setup -> benchmark-and-profile -> analysis`
- `optimize`: `env -> vllm-setup -> benchmark-and-profile -> analysis -> kernel-optimize -> integration -> report`
- `optimize-only`: `env -> vllm-setup -> kernel-optimize -> integration -> report` (requires existing analysis artifacts)

Choose the narrowest mode that matches the user's goal.

## Files to read

Read these in this order:

1. Before Round 1: no extra file reads required.
2. After Round 1 answers: [`INTAKE.md`](INTAKE.md)
3. Before discovery/bootstrap: [`RUNTIME.md`](RUNTIME.md)
4. Before execution: only the phase docs needed for the chosen mode
5. Read [`EXAMPLES.md`](EXAMPLES.md) only if interaction quality has drifted.

## References

- [`INTAKE.md`](INTAKE.md)
- [`RUNTIME.md`](RUNTIME.md)
- [`EXAMPLES.md`](EXAMPLES.md)
- [Phase 0: Environment Setup](phases/00-env-setup.md)
- [Phase 1: vLLM Server Setup](phases/01-vllm-setup.md)
- [Phase 2: Benchmark & Profile](phases/02-benchmark-and-profile.md)
- [Phase 3: Analysis](phases/03-analysis.md)
- [Phase 4: Kernel Optimization](phases/07-kernel-optimize.md)
- [Phase 5: Integration & E2E Benchmark](phases/08-integration.md)
- [Phase 6: Final Report](phases/09-report-generate.md)