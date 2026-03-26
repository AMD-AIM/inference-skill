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

## Completion
Update progress.json:
```json
{
  "phase": "env",
  "phases_completed": ["env"],
  "current_step": "environment verified"
}
```
