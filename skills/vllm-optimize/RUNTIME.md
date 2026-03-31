# vLLM Optimize - Runtime

## Execution Environment

### Container Requirements

- ROCm 6.0+ (AMD) or CUDA 12.0+ (NVIDIA)
- Python 3.10+
- vLLM installed
- PyTorch with profiler support
- HuggingFace transformers

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HF_HOME | HuggingFace cache | /root/.cache/huggingface |
| CUDA_VISIBLE_DEVICES | GPU selection | 0 |
| VLLM_WORKER_MULTIPROC_METHOD | Multiprocessing | spawn |
| VLLM_TORCH_PROFILER_DIR | Profiler output | /workspace/profiles |

## Execution Modes

### Full Mode

```
env -> vllm-setup -> benchmark -> profiling -> analysis
```

1. Environment validation
2. vLLM server startup
3. Concurrency sweep benchmark
4. Torch profiler trace capture
5. GPU kernel gap analysis
6. Generate markdown report

### Benchmark Mode

```
env -> vllm-setup -> benchmark
```

Faster execution without profiling:
- Server startup
- Concurrency sweep
- Results export

### Profile Mode

```
vllm-setup -> profiling -> analysis
```

Requires vLLM server already running:
- Profiler trigger
- Inference with trace
- Kernel analysis
- Report generation

## Phase Details

### Phase 1: Environment Setup

Validates:
- GPU availability
- vLLM installation
- Sufficient GPU memory
- Network access for model download

### Phase 2: vLLM Server

Starts vLLM with appropriate flags:
- `--enforce-eager` for profiling compatibility
- `--trust-remote-code` for custom models
- Profiler config flags when profiling enabled

### Phase 3: Benchmark

Runs concurrent requests at multiple levels:
- Uses OpenAI-compatible API
- Measures RPS, TPS, latency
- Captures P50/P90/P99 percentiles

### Phase 4: Profiling

Uses vLLM's built-in torch profiler:
- Trigger via `/start_profile` API
- Run inference requests
- Stop via `/stop_profile` API
- Capture trace to configured directory

### Phase 5: Analysis

Analyzes trace with proper kernel filtering:
- Excludes Python profiler annotations
- Identifies actual GPU hardware kernels
- Categorizes: MoE, Attention, Memory, Activation, etc.
- Generates performance report

## Common Issues

### Server Fails to Start

- Check GPU memory: `rocm-smi` or `nvidia-smi`
- Try `--enforce-eager` flag
- Verify model is supported

### Profiler Trace Not Generated

- Ensure `--profiler-config.profiler torch` is set
- Check server logs for profiler initialization
- Verify max_iterations > 0

### Analysis Shows Wrong Kernels

- Filter out `execute_context_*` (Python annotations)
- Only include `cat=kernel/cuda/gpu` events
- Use kernel pattern matching for actual GPU ops

## Exit Codes

- 0: Success
- 1: Environment error
- 2: Server startup failed
- 3: Benchmark failed
- 4: Profiling failed
- 5: Analysis failed