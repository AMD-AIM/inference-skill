> **ARCHIVE**: This file is a reference copy of the original phase runbook. The active
> agent docs are in `agents/phase-NN-*.md`. Script paths in this file reference the
> pre-reorganization flat layout (`scripts/*.py`); the actual scripts are now under
> `scripts/{env,container,profiling,optimize,plugin,report}/`.

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
```bash
if command -v rocminfo &>/dev/null; then
    AMD_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 | tr '[:upper:]' '[:lower:]')
    [ -n "$AMD_ARCH" ] && { echo "AMD GPU detected: $AMD_ARCH"; rocm-smi --showuse 2>/dev/null || true; }
fi
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
    [ -n "$NVIDIA_ARCH" ] && { echo "NVIDIA GPU detected: $NVIDIA_ARCH"; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true; }
fi
[ -z "$AMD_ARCH" ] && [ -z "$NVIDIA_ARCH" ] && { echo "ERROR: No GPU detected" >&2; exit 1; }
```

### 3. Clone or Update InferenceX Repository (CRITICAL)
```bash
REPO_DIR="{{REPO_DIR}}"
REPO_URL="{{REPO_URL}}"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Using existing repo at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only || true
else
    rm -rf "$REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi
```

### 4. Verify Config File and Config Key
```bash
CONFIG_KEY="{{CONFIG_KEY}}"
CONFIG_FILE=$([[ "$CONFIG_KEY" == *mi3* ]] && echo ".github/configs/amd-master.yaml" || echo ".github/configs/nvidia-master.yaml")
[ -f "{{REPO_DIR}}/$CONFIG_FILE" ] || { echo "ERROR: Config file not found" >&2; exit 1; }
python3 "{{SCRIPTS_DIR}}/validate_config_key.py" \
    --config-file "{{REPO_DIR}}/$CONFIG_FILE" \
    --config-key "$CONFIG_KEY"
```
If validation fails, stop here and use one of the suggested keys before continuing.

### 5. Check Python3 and Dependencies
```bash
python3 --version
python3 -c "import yaml; import json; print('Dependencies OK')"
```

### 6. Verify HuggingFace Cache
```bash
mkdir -p "{{HF_CACHE}}"
```

### 7. Install or Detect GEAK
```bash
GEAK_DIR="{{GEAK_DIR}}"
GEAK_AVAILABLE=false
if [ -d "$GEAK_DIR" ] && python3 -c "import sys; sys.path.insert(0,'$GEAK_DIR/src'); from minisweagent.run.mini import app; print('GEAK: OK')" 2>/dev/null; then
    GEAK_AVAILABLE=true
else
    rm -rf "$GEAK_DIR"
    git clone https://github.com/amd/GEAK.git "$GEAK_DIR" && {
        cd "$GEAK_DIR" && pip install -e . 2>&1 | tail -5
        python3 -c "from minisweagent.run.mini import app" 2>/dev/null && GEAK_AVAILABLE=true || echo "WARNING: GEAK install incomplete"
    } || echo "WARNING: Failed to clone GEAK"
fi
GEAK_OE_AVAILABLE=false
[ -d "{{GEAK_OE_DIR}}" ] && GEAK_OE_AVAILABLE=true
```

### 8. Verify LLM API Key
```bash
LLM_API_KEY_SET=false
[ -n "${AMD_LLM_API_KEY:-}" ] || [ -n "${LLM_GATEWAY_KEY:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ] && LLM_API_KEY_SET=true
[ "$LLM_API_KEY_SET" = "false" ] && echo "WARNING: No LLM API key found — GEAK will not work"
```

### 9. Write env_info.json
```bash
export GEAK_AVAILABLE GEAK_OE_AVAILABLE LLM_API_KEY_SET
python3 "{{SCRIPTS_DIR}}/generate_env_info.py" --output "{{ENV_INFO_FILE}}" --geak-dir "{{GEAK_DIR}}"
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
