# Phase 1: vLLM Server Setup

## Objective
Start vLLM server with required configuration for benchmark and profiling.

## CRITICAL: Profiler Enablement
To use torch profiler, you MUST start vLLM with `--profiler-config.*` flags. Without them, `/start_profile` and `/stop_profile` return 404.

## Steps

### 1. Environment Validation

Run: `bash scripts/detect_gpu.sh`

Outputs `GPU_VENDOR`, validates vLLM + PyTorch installation, sets GPU visibility if needed.

### 2. Create Directories
```bash
OUTPUT_DIR=${OUTPUT_DIR:-./vllm_results}
PROFILE_DIR=${PROFILE_DIR:-$OUTPUT_DIR/profiles}
mkdir -p "$OUTPUT_DIR" "$PROFILE_DIR"
```

### 3. Validate Model Name (CRITICAL)

Use the EXACT model name provided by the user. Never guess or substitute. If ambiguous, ask the user for the exact HuggingFace model ID.

### 4. Ensure Model is Available
```bash
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True, use_fast=False)
print(f'Model {\"$MODEL\"} is available')
" 2>/dev/null || {
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
}
```

### 5. Start vLLM Server

**Benchmark-only mode (no profiling):**
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype ${DTYPE:-half} --tensor-parallel-size ${TP:-1} \
    --trust-remote-code --enforce-eager --api-key dummy \
    > /tmp/vllm.log 2>&1 &
```

**Profiling mode (REQUIRED for profiler API):**
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype ${DTYPE:-half} --tensor-parallel-size ${TP:-1} \
    --trust-remote-code --enforce-eager --api-key dummy \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "$PROFILE_DIR" \
    --profiler-config.torch_profiler_record_shapes True \
    --profiler-config.torch_profiler_with_stack True \
    --profiler-config.torch_profiler_use_gzip True \
    --profiler-config.ignore_frontend True \
    --profiler-config.max_iterations ${PROFILE_ITERATIONS:-128} \
    > /tmp/vllm.log 2>&1 &
```

### 6. Wait for Server Ready
```bash
for i in {1..50}; do
    curl -s -H "Authorization: Bearer dummy" http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL" && { echo "Server ready!"; break; }
    sleep 12
done
```

### 7. Verify Profiler API (if profiling enabled)
```bash
curl -s -X POST http://localhost:8000/start_profile -H "Authorization: Bearer dummy"
curl -s -X POST http://localhost:8000/stop_profile -H "Authorization: Bearer dummy"
```
If 404, restart with `--profiler-config.*` flags.

## Troubleshooting
- **Profiler 404**: Server not started with profiling flags. Restart with `--profiler-config.*`.
- **"execute_context" in traces**: Python annotations, not GPU kernels. Filter with EXCLUDE_PATTERNS in analysis.

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | HuggingFace model name |
| TP | 1 | Tensor parallelism |
| DTYPE | half | Model precision |
| PROFILE_DIR | $OUTPUT_DIR/profiles | Profiler output |

## Completion
Server running at http://localhost:8000. Next: Phase 2 (Benchmark).
