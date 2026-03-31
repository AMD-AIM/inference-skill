# Phase 0: Environment Setup {{SKIP_LABEL}}

## Objective
Verify that all prerequisites are installed and the InferenceX repository is available.

## Steps

### 1. Check Docker
```bash
docker --version
```
If Docker is not available, report an error and stop.

### 2. Check GPU Availability
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

### 3. Clone or Update InferenceX Repository (CRITICAL — do NOT skip)
You MUST run this step. The repo directory may exist but be empty.
```bash
REPO_DIR="{{REPO_DIR}}"
REPO_URL="{{REPO_URL}}"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Using existing repo at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only || true
else
    echo "InferenceX repo not found at $REPO_DIR, cloning..."
    rm -rf "$REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi
```
If `git clone` fails, report the error and stop.

### 4. Verify Config File Exists
```bash
REPO_DIR="{{REPO_DIR}}"
CONFIG_KEY="{{CONFIG_KEY}}"
if [[ "$CONFIG_KEY" == *mi3* ]]; then
    CONFIG_FILE=".github/configs/amd-master.yaml"
else
    CONFIG_FILE=".github/configs/nvidia-master.yaml"
fi

if [ ! -f "$REPO_DIR/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $REPO_DIR/$CONFIG_FILE" >&2
    echo "This likely means the repo was not cloned. Go back to step 3." >&2
    exit 1
fi
echo "Config file found: $REPO_DIR/$CONFIG_FILE"
```

### 5. Check Python3 and Dependencies
```bash
python3 --version
python3 -c "import yaml; import json; print('Dependencies OK')"
```

### 6. Verify HuggingFace Cache
```bash
HF_CACHE="{{HF_CACHE}}"
mkdir -p "$HF_CACHE"
echo "HF cache directory: $HF_CACHE"
```

### 7. Install or Detect GEAK

```bash
GEAK_DIR="{{GEAK_DIR}}"
if [ -d "$GEAK_DIR" ] && python3 -c "
  import sys; sys.path.insert(0,'$GEAK_DIR/src')
  from minisweagent.run.mini import app; print('GEAK: OK')
" 2>/dev/null; then
    echo "GEAK found at $GEAK_DIR"
    GEAK_AVAILABLE=true
else
    echo "GEAK not found — cloning..."
    rm -rf "$GEAK_DIR"
    git clone https://github.com/amd/GEAK.git "$GEAK_DIR"
    if [ $? -eq 0 ]; then
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

# Also check geak-oe (optional)
GEAK_OE_AVAILABLE=false
[ -d "{{GEAK_OE_DIR}}" ] && { echo "geak-oe: found"; GEAK_OE_AVAILABLE=true; } || echo "geak-oe: not found (optional)"
```

### 8. Verify LLM API Key

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

### 9. Write env_info.json

Pass the detection results from Steps 7-8 as environment variables:

```bash
export GEAK_AVAILABLE="${GEAK_AVAILABLE:-false}"
export GEAK_OE_AVAILABLE="${GEAK_OE_AVAILABLE:-false}"
export LLM_API_KEY_SET="${LLM_API_KEY_SET:-false}"

python3 -c "
import json, os, subprocess

gpu_vendor = 'unknown'
gpu_arch = 'unknown'
gpu_count = 0

# Detect AMD GPUs
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

# Detect NVIDIA GPUs
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
    'geak_available': os.environ.get('GEAK_AVAILABLE', 'false').lower() == 'true',
    'geak_dir': '{{GEAK_DIR}}',
    'geak_oe_available': os.environ.get('GEAK_OE_AVAILABLE', 'false').lower() == 'true',
    'llm_api_key_set': os.environ.get('LLM_API_KEY_SET', 'false').lower() == 'true',
    'runtime_type': 'docker',
}

with open('{{ENV_INFO_FILE}}', 'w') as f:
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
