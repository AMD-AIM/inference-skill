# vLLM Optimize - Intake

## Configuration Options

### Model Selection

Any HuggingFace model that vLLM supports:
- Llama variants: `meta-llama/Llama-3.1-8B-Instruct`
- Qwen variants: `Qwen/Qwen3.5-35B-A3B`
- Mistral variants: `mistralai/Mistral-7B-Instruct-v0.3`
- Custom models: path to local model or HF model ID

### Benchmark Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| ISL | 1024 | 128-32K | Input sequence length |
| OSL | 1024 | 16-4K | Output sequence length |
| CONCURRENCY | 4,8,16,32,64,128 | 1-256 | Concurrency levels to test |
| TP | 1 | 1-8 | Tensor parallelism |
| PRECISION | fp16 | fp16, fp8, bf16 | Model precision |
| FRAMEWORK | vLLM | vLLM | Inference framework |

### Profiling Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| PROFILE_ITERATIONS | 128 | Number of profiler iterations |
| PROFILE_DELAY | 0 | Delay before profiling starts |
| ENABLE_TRACE | true | Generate torch profiler trace |
| GAP_ANALYSIS | true | Analyze GPU kernel breakdown |

### Output Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| OUTPUT_DIR | ./vllm_results | Results directory |
| SAVE_TRACE | true | Save raw profiler trace |
| GENERATE_REPORT | true | Generate markdown report |

## Filter Options

### Smoke Test Defaults
- Concurrency: 4, 16, 64
- Sequence length: 1024
- Profile iterations: 64

### Full Sweep
- Concurrency: 4, 8, 16, 32, 64, 128
- Sequence length: 512, 1024, 2048
- Profile iterations: 128

### Custom
- User specifies which parameters to sweep

## Discovery

The skill will auto-detect:
- GPU model (MI355X, MI300X, A100, H100, etc.)
- Available GPU memory
- vLLM installation and version
- ROCm/CUDA version

## Questions

### Round 1 (High Level)

1. **Run plan**: Smoke test / Full sweep / Custom
2. **Output**: Default ./vllm_results or custom path
3. **GPUs**: All available or specific GPU IDs

### Round 2 (After Discovery)

Model-specific options:
- Precision (if model supports fp8)
- Custom vLLM flags
- Specific sequence lengths to test