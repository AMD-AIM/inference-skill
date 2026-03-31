# Phase 1: vLLM Server Setup

## Objective
Start vLLM server with required configuration for benchmark and profiling.

## Steps

### 1. Environment Validation

```bash
# Check GPU availability
rocm-smi --showproductname 2>/dev/null || nvidia-smi --query-gpu=name --format=csv

# Check vLLM installation
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"

# Check GPU memory
rocm-smi --info mem 2>/dev/null | head -5 || nvidia-smi --query-gpu=memory.total --format=csv
```

### 2. Create Directories

```bash
OUTPUT_DIR=${OUTPUT_DIR:-./vllm_results}
PROFILE_DIR=${PROFILE_DIR:-$OUTPUT_DIR/profiles}

mkdir -p "$OUTPUT_DIR" "$PROFILE_DIR"
```

### 3. Start vLLM Server

**For benchmark-only mode:**
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype ${DTYPE:-half} \
    --tensor-parallel-size ${TP:-1} \
    --trust-remote-code \
    --enforce-eager \
    --api-key dummy \
    > /tmp/vllm.log 2>&1 &
```

**For profiling mode:**
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype ${DTYPE:-half} \
    --tensor-parallel-size ${TP:-1} \
    --trust-remote-code \
    --enforce-eager \
    --api-key dummy \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "$PROFILE_DIR" \
    --profiler-config.torch_profiler_record_shapes True \
    --profiler-config.torch_profiler_with_stack True \
    --profiler-config.torch_profiler_use_gzip True \
    --profiler-config.ignore_frontend True \
    --profiler-config.max_iterations ${PROFILE_ITERATIONS:-128} \
    > /tmp/vllm.log 2>&1 &
```

### 4. Wait for Server Ready

```bash
for i in {1..50}; do
    if curl -s -H "Authorization: Bearer dummy" \
        http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL"; then
        echo "Server ready!"
        break
    fi
    sleep 12
done
```

### 5. Verify Server

```bash
curl -s -H "Authorization: Bearer dummy" http://localhost:8000/v1/models | \
    python3 -c "import sys,json; d=json.load(sys.stdin)
print('Status: Ready' if d.get('data') else 'Status: Failed')
print('Model:', d.get('data',[{}])[0].get('id','unknown'))"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | (required) | HuggingFace model name |
| TP | 1 | Tensor parallelism |
| DTYPE | half | Model precision |
| OUTPUT_DIR | ./vllm_results | Results directory |
| PROFILE_DIR | $OUTPUT_DIR/profiles | Profiler output |
| PROFILE_ITERATIONS | 128 | Profiler iterations |

## Completion

Server running at http://localhost:8000 with OpenAI-compatible API.

Next: Proceed to Phase 2 (Benchmark)