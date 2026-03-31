# Phase 1: vLLM Server Setup

## Objective
Start vLLM server with required configuration for benchmark and profiling.

## Steps

### 1. Environment Validation

```bash
# Detect GPU vendor first
GPU_VENDOR=""
if command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
    echo "Detected AMD GPU"
elif command -v nvidia-smi &>/dev/null; then  
    GPU_VENDOR="nvidia"
    echo "Detected NVIDIA GPU"
else
    echo "Warning: No GPU detection tool found"
fi

# For AMD GPUs, get more info
if [[ "$GPU_VENDOR" == "amd" ]]; then
    echo "=== AMD GPU Detection ==="
    rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not available"
    rocminfo 2>/dev/null | grep -i "gfx" | head -3 || echo "No GFX info"
    
    # Check GPU visibility
    echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    
    # List available GPU devices
    if [[ -d /dev/dri ]]; then
        echo "Available GPU devices:"
        ls -la /dev/dri/card* 2>/dev/null || echo "No /dev/dri/cards found"
    fi
fi

# For NVIDIA GPUs
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# Check vLLM installation
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" || echo "vLLM not installed"

# Check PyTorch GPU detection
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "PyTorch GPU check failed"
```

### 2. Set Up GPU Visibility (if needed)

```bash
# If no GPUs visible, try to find and set GPU devices
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Warning: No GPUs visible to PyTorch, attempting to set..."
    
    # For AMD, check /dev/dri
    if [[ -d /dev/dri ]]; then
        GPU_IDS=$(ls -1 /dev/dri/card* 2>/dev/null | grep -v render | sed 's/.*card//' | tr '\n' ',' | sed 's/,$//')
        if [[ -n "$GPU_IDS" ]]; then
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            export HIP_VISIBLE_DEVICES="$GPU_IDS"
            echo "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"
        fi
    fi
    
    # Verify again
    python3 -c "import torch; print(f'After setting: {torch.cuda.device_count()} GPUs')"
fi

# Ensure HF_HOME is set for model caching
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}/hub
mkdir -p "$HF_HOME"
```

### 3. Create Directories

```bash
OUTPUT_DIR=${OUTPUT_DIR:-./vllm_results}
PROFILE_DIR=${PROFILE_DIR:-$OUTPUT_DIR/profiles}

mkdir -p "$OUTPUT_DIR" "$PROFILE_DIR"
```

### 4. Ensure Model is Available

```bash
# Download/cache model if not present (safe operation)
python3 << 'PYEOF'
import os
import sys

model = os.environ.get('MODEL', 'Qwen/Qwen3.5-35B-A3B')
hf_home = os.environ.get('HF_HOME', '/root/.cache/huggingface')
os.environ['HF_HOME'] = hf_home

print(f"Checking model: {model}")

try:
    # First try: Direct load (will download if needed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Attempting to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Model {model} is available and ready")
except Exception as e:
    print(f"First attempt failed: {e}")
    print("Trying alternative method...")
    
    # Second try: snapshot_download (more explicit)
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading model to {hf_home}...")
        model_dir = snapshot_download(model, cache_dir=hf_home)
        print(f"Model downloaded to: {model_dir}")
    except Exception as e2:
        print(f"Download also failed: {e2}")
        sys.exit(1)
PYEOF
```

### 5. Start vLLM Server

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

### 6. Wait for Server Ready

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

### 7. Verify Server

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
| HF_HOME | /root/.cache/huggingface | HuggingFace cache directory |
| OUTPUT_DIR | ./vllm_results | Results directory |
| PROFILE_DIR | $OUTPUT_DIR/profiles | Profiler output |
| PROFILE_ITERATIONS | 128 | Profiler iterations |

## Completion

Server running at http://localhost:8000 with OpenAI-compatible API.

Next: Proceed to Phase 2 (Benchmark)