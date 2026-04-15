---
name: vllm-optimize
description: "Run vLLM benchmark and profiling workflow in containerized environments. Provides automated model loading, concurrency sweep, torch profiling, and GPU kernel analysis. Works with AMD MI355X/MI300X and NVIDIA GPUs."
compatibility: claude-code, opencode
metadata:
  workflow: vllm-benchmark
  audience: performance-engineers
  distribution: standalone-skill-repo
---

# vLLM Optimize

Automated vLLM inference benchmark and profiling workflow for containerized environments.

## Quick Start

Run the workflow with a model name:

```
use vllm-optimize skill for Qwen/Qwen3.5-35B-A3B
```

The skill will automatically:
1. Start vLLM server with the specified model
2. Run benchmark at various concurrency levels
3. Generate profiling traces
4. Analyze GPU kernel performance

## First-turn Latency Rule

- Do not read any other file before the first visible reply
- Send one short kickoff status update explaining the workflow
- Ask the first grouped setup form with options

## Guided Setup Flow

1. Start with one short high-level question round:
   - `Run plan` (smoke test vs full sweep)
   - `Output` (where to save results)
   - `GPUs` (which GPUs to use)

2. After Round 1 answers, read `INTAKE.md` for deeper config

3. Read `RUNTIME.md` for execution bootstrap

4. Summarize the final plan and get confirmation before executing

5. After confirmation, start execution following phase docs

## Modes

- `full`: benchmark + profiling + analysis
- `benchmark`: benchmark only (faster)
- `profile`: profiling only (requires server running)

## Files to Read

1. Before Round 1: no extra file reads required
2. After Round 1 answers: `INTAKE.md`
3. Before execution: `RUNTIME.md`
4. Phase docs: `phases/*.md`

## References

- [`INTAKE.md`](INTAKE.md) - Configuration options
- [`RUNTIME.md`](RUNTIME.md) - Execution details
- [Phase 1: vLLM Server Setup](phases/01-vllm-setup.md)
- [Phase 2: Benchmark Execution](phases/02-benchmark.md)
- [Phase 3: Profiling](phases/03-profiling.md)
- [Phase 4: Analysis](phases/04-analysis.md)