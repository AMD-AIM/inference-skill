# Phase 0: Environment Setup {{SKIP_LABEL}}

## Objective
Verify that all prerequisites are installed and the GPU environment is ready for vLLM inference optimization.

## Steps

### 1. Check GPU Availability
Detect GPU vendor, architecture, and current utilization:
```bash
# Try AMD GPU detection via rocminfo
if command -v rocminfo &>/dev/null; then
    AMD_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 | tr '[:upper:]' '[:lower:]')
    if [ -n "$AMD_ARCH" ]; then
        echo "AMD GPU detected: $AMD_ARCH"
        # Show GPU count and utilization
        rocm-smi --showuse 2>/dev/null || true
    fi
fi

# Try NVIDIA GPU detection via nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
    if [ -n "$NVIDIA_ARCH" ]; then
        echo "NVIDIA GPU detected: $NVIDIA_ARCH"
        # Show GPU count and utilization
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    fi
fi

# Fail if no GPU was found
if [ -z "$AMD_ARCH" ] && [ -z "$NVIDIA_ARCH" ]; then
    echo "ERROR: No GPU detected" >&2
    exit 1
fi
```

### 2. Check vLLM and PyTorch
```bash
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/HIP available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### 3. Check Python3 and Dependencies
```bash
python3 --version
python3 -c "import yaml; import json; print('Dependencies OK')"
```

### 4. Verify HuggingFace Cache
```bash
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
echo "HF cache directory: $HF_CACHE"
```

### 5. Install or Detect GEAK
```bash
GEAK_DIR="${GEAK_DIR:-$HOME/GEAK}"
if [ -d "$GEAK_DIR" ] && python3 -c "
  import sys; sys.path.insert(0,'$GEAK_DIR/src')
  from minisweagent.run.mini import app; print('GEAK: OK')
" 2>/dev/null; then
    echo "GEAK found at $GEAK_DIR"
    GEAK_AVAILABLE=true
else
    echo "GEAK not found — cloning..."
    rm -rf "$GEAK_DIR"
    git clone https://github.com/amd/GEAK.git "$GEAK_DIR" 2>/dev/null || true
    if [ -d "$GEAK_DIR" ]; then
        cd "$GEAK_DIR" && pip install -e . 2>&1 | tail -5
        python3 -c "from minisweagent.run.mini import app; print('GEAK installed')" 2>/dev/null && GEAK_AVAILABLE=true || {
            echo "WARNING: GEAK install incomplete — manual fallback will be used"
            GEAK_AVAILABLE=false
        }
    else
        echo "WARNING: Failed to clone GEAK — manual fallback will be used"
        GEAK_AVAILABLE=false
    fi
fi
```

### 6. Verify LLM API Key
```bash
LLM_API_KEY_SET=false
if [ -n "${AMD_LLM_API_KEY:-}" ]; then
    echo "AMD_LLM_API_KEY: set"
    LLM_API_KEY_SET=true
elif [ -n "${LLM_GATEWAY_KEY:-}" ]; then
    echo "LLM_GATEWAY_KEY: set"
    LLM_API_KEY_SET=true
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY: set"
    LLM_API_KEY_SET=true
else
    echo "WARNING: No LLM API key found (AMD_LLM_API_KEY, LLM_GATEWAY_KEY, or ANTHROPIC_API_KEY)"
    echo "GEAK optimization will not work without an API key"
fi
```

### 7. Write env_info.json
```bash
python3 -c "
import json, os, subprocess

gpu_vendor = 'unknown'
gpu_arch = 'unknown'
gpu_count = 0

try:
    result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        import re
        arches = re.findall(r'gfx\w+', result.stdout)
        if arches:
            gpu_vendor = 'amd'
            gpu_arch = arches[0].lower()
            gpu_count = len(set(arches))
except Exception:
    pass

if gpu_vendor == 'unknown':
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_vendor = 'nvidia'
            gpu_count = int(result.stdout.strip().split('\n')[0])
        result2 = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        if result2.returncode == 0:
            cap = result2.stdout.strip().split('\n')[0].replace('.', '')
            gpu_arch = f'sm_{cap}'
    except Exception:
        pass

env_info = {
    'gpu_vendor': gpu_vendor,
    'gpu_arch': gpu_arch,
    'gpu_count': gpu_count,
    'geak_available': os.environ.get('GEAK_AVAILABLE', '${GEAK_AVAILABLE:-false}').lower() == 'true',
    'geak_dir': '${GEAK_DIR}',
    'llm_api_key_set': os.environ.get('LLM_API_KEY_SET', '${LLM_API_KEY_SET:-false}').lower() == 'true',
    'runtime_type': 'host',
}

with open('${OUTPUT_DIR}/env_info.json', 'w') as f:
    json.dump(env_info, f, indent=2)
print(json.dumps(env_info, indent=2))
"
```

## Completion
Update progress.json:
```json
{
  "phase": "env",
  "phases_completed": ["env"],
  "current_step": "environment verified",
  "details": {
    "geak_available": "<true/false>",
    "gpu_arch": "<detected arch>"
  }
}
```