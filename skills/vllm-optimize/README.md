# vLLM Optimize

Automated vLLM benchmark, profiling, and kernel optimization workflow for AMD and NVIDIA GPUs.

## Overview

This skill provides automated vLLM inference benchmarking, GPU kernel profiling, and kernel optimization. It works with:
- **AMD GPUs**: MI355X, MI300X, R7900 (RDNA3), and other ROCm-supported GPUs
- **NVIDIA GPUs**: A100, H100, and other CUDA-supported GPUs

## Relationship to inferencex-optimize

- **vllm-optimize**: Standalone vLLM workflow (this skill) — runs directly on the host, no Docker needed
- **inferencex-optimize**: Full InferenceX pipeline — uses Docker containers and InferenceX benchmark scripts

This skill now includes all optimization phases from inferencex-optimize (problem generation, kernel optimization, integration, reporting), adapted for direct vLLM usage.

## Quick Usage

```
use vllm-optimize skill for <model-name>
```

The skill will:
1. Set up the environment and start vLLM server
2. Run concurrency sweep benchmark
3. Capture profiler traces
4. Analyze GPU kernels
5. Generate optimization problems
6. Optimize kernels (with GEAK or manual)
7. Integrate and re-benchmark
8. Generate final report

## Modes

| Mode | Phases | Description |
|------|--------|-------------|
| `full` | env → setup → benchmark → analyze → profiling → profile-analyze | Benchmark + profiling |
| `benchmark` | env → setup → benchmark → benchmark-analyze | Quick benchmark only |
| `profile` | setup → profiling → profile-analyze | Profiling only (needs running server) |
| `optimize` | All 10 phases | Full optimization pipeline |
| `optimize-only` | env → setup → problem-generate → kernel-optimize → integration → report | Skip re-profiling |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL | (required) | HuggingFace model ID |
| ISL | 1024 | Input sequence length |
| OSL | 1024 | Output sequence length |
| TP | 1 | Tensor parallelism |
| CONCURRENCY_LEVELS | 4,8,16,32,64,128 | Concurrency levels to test |
| PRECISION | half | Model precision (half, bfloat16, fp8) |
| GEAK_MODE | auto | GEAK optimization mode |

## Output

```
vllm_optimize_<model>_<timestamp>/
├── env_info.json
├── progress.json
├── config.json
├── results/
│   ├── benchmark_summary.json
│   ├── bottleneck.json
│   ├── gap_analysis/
│   │   ├── gap_analysis.json
│   │   └── category_breakdown.json
│   └── model_shapes/
├── profiles/
│   └── *.pt.trace.json.gz
├── problems/
│   ├── problem_list.json
│   └── problem_*.json
├── optimized/
│   └── finalized/
├── plugin/
├── report/
│   ├── benchmark_report.md
│   ├── profiling_report.md
│   └── final_report.md
└── scripts/
```

## Key Features

- **No Docker required**: Runs vLLM directly on the host
- **Works with consumer GPUs**: Supports R7900 (RDNA3), MI355X, MI300X, A100, H100
- **Automated**: Single command to run full workflow
- **Proper Kernel Analysis**: Filters out Python profiler annotations
- **GPU Kernel Categorization**: MoE, Attention, Memory, Activation, etc.
- **GEAK Integration**: Automated kernel optimization with GEAK when available
- **Manual Kernel Optimization**: Fallback when GEAK is not available

## Files

- `SKILL.md` - Skill definition and metadata
- `INTAKE.md` - Guided setup flow
- `RUNTIME.md` - Execution details and phase map
- `EXAMPLES.md` - Interaction examples
- `phases/00-env-setup.md` - Environment verification
- `phases/01-vllm-setup.md` - Server startup
- `phases/02-benchmark.md` - Benchmark execution
- `phases/03-benchmark-analyze.md` - Benchmark analysis
- `phases/04-profiling.md` - Profiler trace capture
- `phases/05-profile-analyze.md` - Kernel analysis and gap analysis
- `phases/06-problem-generate.md` - Optimization problem generation
- `phases/07-kernel-optimize.md` - Kernel optimization (GEAK or manual)
- `phases/08-integration.md` - Integration and E2E benchmark
- `phases/09-report-generate.md` - Final report generation
- `scripts/` - Helper scripts
- `templates/` - Agent configuration templates