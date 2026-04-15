# Phase 0: Environment Setup

## Instructions

You are a phase agent responsible for verifying the execution environment. You read exactly 2 files: this document and your handoff at `handoff/to-phase-00.md`.

**Tools**: Shell commands, file I/O.
**Outputs**: Write `agent-results/phase-00-result.md` per `protocols/phase-result.schema.md`. Write `env_info.json` to `{{OUTPUT_DIR}}`.
**Errors**: If Docker or GPUs are unavailable, report failure immediately. Do not attempt workarounds for missing hardware.

## Runbook

### 1. Check Docker
```bash
docker --version
```
If unavailable, write a failed result and stop.

### 2. Check GPU Availability
```bash
if command -v rocminfo &>/dev/null; then
    AMD_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 | tr '[:upper:]' '[:lower:]')
    [ -n "$AMD_ARCH" ] && echo "AMD GPU: $AMD_ARCH"
fi
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
    [ -n "$NVIDIA_ARCH" ] && echo "NVIDIA GPU: $NVIDIA_ARCH"
fi
[ -z "$AMD_ARCH" ] && [ -z "$NVIDIA_ARCH" ] && { echo "ERROR: No GPU detected"; exit 1; }
```

### 3. Clone or Update InferenceX Repository
```bash
if [ -d "{{REPO_DIR}}/.git" ]; then
    cd "{{REPO_DIR}}" && git pull --ff-only || true
else
    rm -rf "{{REPO_DIR}}"
    git clone "{{REPO_URL}}" "{{REPO_DIR}}"
fi
```

### 4. Verify Config Key
```bash
CONFIG_FILE=$([[ "{{CONFIG_KEY}}" == *mi3* ]] && echo ".github/configs/amd-master.yaml" || echo ".github/configs/nvidia-master.yaml")
python3 "{{SCRIPTS_DIR}}/env/validate_config_key.py" \
    --config-file "{{REPO_DIR}}/$CONFIG_FILE" --config-key "{{CONFIG_KEY}}"
```
If validation fails, stop and report the error with suggested keys.

### 5. Check Python3 and Dependencies
```bash
python3 -c "import yaml; import json; print('Dependencies OK')"
```

### 6. Verify HuggingFace Cache
```bash
mkdir -p "{{HF_CACHE}}"
```

### 7. Install or Detect GEAK
Use `GEAK_DIR="{{GEAK_DIR}}"`. Prefer an existing install that imports cleanly; otherwise clone and install in editable mode. Also detect **geak-oe** via `{{GEAK_OE_DIR}}` (directory present ⇒ available).

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

Verification: after install, `python3 -c "from minisweagent.run.mini import app"` should succeed when run with `GEAK_DIR` on `PYTHONPATH` / from the cloned tree as above.

### 8. Verify LLM API Key
GEAK needs at least one of these environment variables set. If none are set, GEAK-dependent flows will not work.

```bash
LLM_API_KEY_SET=false
[ -n "${AMD_LLM_API_KEY:-}" ] || [ -n "${LLM_GATEWAY_KEY:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ] && LLM_API_KEY_SET=true
[ "$LLM_API_KEY_SET" = "false" ] && echo "WARNING: No LLM API key found — GEAK will not work"
```

### 9. Write env_info.json
Export `GEAK_AVAILABLE`, `GEAK_OE_AVAILABLE`, and `LLM_API_KEY_SET` for the generator if your shell session needs them persisted:

```bash
export GEAK_AVAILABLE GEAK_OE_AVAILABLE LLM_API_KEY_SET
python3 "{{SCRIPTS_DIR}}/env/generate_env_info.py" --output "{{ENV_INFO_FILE}}" --geak-dir "{{GEAK_DIR}}"
```

### Completion
Write `agent-results/phase-00-result.md` with status, artifacts (`env_info.json`), and key findings (GPU arch, GEAK availability, geak-oe detection, LLM key presence).

Include these sticky fields in `## Data for Next Phase`:
- `gpu_arch`: string (e.g. "gfx942", "sm_90")
- `gpu_count`: integer
- `container_image`: string (Docker image used for benchmarks)

Do NOT write to `progress.json` — the orchestrator manages progress tracking.
