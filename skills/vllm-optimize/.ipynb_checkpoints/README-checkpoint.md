# vLLM Optimize

Automated vLLM benchmark and profiling workflow for containerized GPU environments.

## Overview

This skill provides automated vLLM inference benchmarking and GPU kernel profiling. It is designed to work in containerized environments (AMD MI355X/MI300X, NVIDIA A100/H100).

## Relationship to inferencex-optimize

- **vllm-optimize**: Standalone vLLM benchmark and profiling (this skill)
- **inferencex-optimize**: Full InferenceX pipeline with benchmark, profiling, and analysis

These are parallel skills for different use cases.

## Quick Usage

```
use vllm-optimize skill for <model-name>
```

The skill will:
1. Start vLLM server
2. Run concurrency sweep benchmark
3. Capture profiler traces
4. Analyze GPU kernels
5. Generate performance report

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL | (required) | HuggingFace model ID |
| ISL | 1024 | Input sequence length |
| OSL | 1024 | Output sequence length |
| TP | 1 | Tensor parallelism |
| CONCURRENCY | 4,8,16,32,64,128 | Concurrency levels |

## Output

```
vllm_results/
├── benchmark_report.json
├── profiling_report.md
├── gap_analysis/
│   ├── gap_analysis.csv
│   └── gap_analysis.json
└── profiles/
    └── *.pt.trace.json.gz
```

## Key Features

- **Automated**: Single command to run full workflow
- **Proper Kernel Analysis**: Filters out Python profiler annotations
- **GPU Kernel Categorization**: MoE, Attention, Memory, Activation, etc.
- **Container Ready**: Works in Docker containers with GPU access

## Files

- `SKILL.md` - Skill definition and metadata
- `INTAKE.md` - Configuration options
- `RUNTIME.md` - Execution details
- `phases/01-vllm-setup.md` - Server startup
- `phases/02-benchmark.md` - Benchmark execution
- `phases/03-profiling.md` - Profiler trace capture
- `phases/04-analysis.md` - Kernel analysis and reporting