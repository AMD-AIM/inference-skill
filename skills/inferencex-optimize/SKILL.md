---
name: inferencex-optimize
description: "Run the InferenceX benchmark and profiling workflow for a config key. When the user names a model or config key, immediately start a fast guided setup flow with batched choice questions, then execute the workflow."
compatibility: claude-code, opencode
metadata:
  workflow: inferencex
  audience: performance-engineers
  distribution: standalone-skill-repo
---

# InferenceX Optimize

## Default user experience

- Treat a bare model/config key as enough to start. Do not require the user to spell out a full command.
- If the user says `use inferencex-optimize skill for <model-or-config-key>`, start guided setup immediately.
- Do not dump raw parameter names at the user in the first reply. Translate them into a short setup conversation.
- Prefer the native `question` tool for multiple-choice prompts when the runtime provides it.
- If the runtime does not provide a question tool, ask concise numbered choices in normal chat.
- Ask questions in grouped batches, not as a drip-feed of one question at a time.
- Keep the user informed with explicit progress updates so they always know the current stage and next step.

## First-turn latency rule

- Do not read any other file before the first visible reply unless the model/config name is ambiguous.
- The first visible reply should happen immediately:
  - send one short kickoff status update
  - ask the first grouped setup form
- Do not read `INTAKE.md`, `RUNTIME.md`, or `EXAMPLES.md` before the first grouped form.
- Only after the user answers Round 1 should you read deeper reference files.

## Guided setup flow

1. Resolve the target config key from the user's model/config name.
2. Start with one short high-level question round. The first question groups should be exactly:
   - `Run plan`
   - `Output`
   - `GPUs`
3. After Round 1 answers, read [`INTAKE.md`](INTAKE.md) and follow its deeper intake flow.
4. Read [`RUNTIME.md`](RUNTIME.md) only when you are about to do discovery or execution bootstrap.
5. Ask high-level setup questions first, then do lightweight config discovery, then ask filter-specific questions.
6. Do not ask TP / sequence length / concurrency until discovery has produced concrete options.
7. For smoke runs, offer a fast-path `Filters` choice:
   - `Use recommended smoke defaults`
   - `Review each filter`
   - `Use full discovered sweep`
8. Summarize the final plan and get a clear go/no-go from the user with a `Confirm` question before executing.
9. Only after confirmation, read the needed phase docs and start execution.

## Status contract

- Before the first question round, send one short kickoff status message explaining that you found the target config and will ask one grouped setup form.
- After every major stage transition, send one short progress update:
  - config resolved
  - setup questions ready
  - discovery in progress
  - discovery complete
  - final plan ready
  - execution starting
- During execution, emit a visible status update at least at every phase boundary.
- Keep status updates human-readable. Prefer wording like `Status 2/5: checking available TP, sequence length, and concurrency options` over internal variable names.
- Never go silent for a long discovery or execution step without a status update first.

## Files to read

Read these in this order:

1. Before Round 1: no extra file reads required.
2. After Round 1 answers: [`INTAKE.md`](INTAKE.md)
3. Before discovery/bootstrap: [`RUNTIME.md`](RUNTIME.md)
4. Before execution: only the phase docs needed for the chosen mode and start phase
5. Read [`EXAMPLES.md`](EXAMPLES.md) only if interaction quality has drifted or you are editing/maintaining this skill.

## Modes

- `full`: `env -> config -> benchmark -> benchmark-analyze -> profile -> profile-analyze`
- `benchmark`: `env -> config -> benchmark -> benchmark-analyze`
- `profile`: `env -> config -> profile -> profile-analyze`
- `benchmark+profile`: same phase set as `full`

Choose the narrowest mode that matches the user's goal. For a smoke run, prefer a narrow configuration and confirm before widening to a full sweep.

## References

- [`INTAKE.md`](INTAKE.md)
- [`RUNTIME.md`](RUNTIME.md)
- [`EXAMPLES.md`](EXAMPLES.md)
- [Phase 0: Environment Setup](phases/00-env-setup.md)
- [Phase 1: Config Parsing & Sweep Generation](phases/01-config-parse.md)
- [Phase 2: Benchmark Execution](phases/02-benchmark.md)
- [Phase 3: Benchmark Analysis](phases/03-benchmark-analyze.md)
- [Phase 4: Profiling](phases/04-profile.md)
- [Phase 5: Profile Analysis](phases/05-profile-analyze.md)
