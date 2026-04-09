#!/usr/bin/env bash
# Detect GPU vendor, validate vLLM installation, set GPU visibility.
# Outputs: GPU_VENDOR=<amd|nvidia>
set -euo pipefail

GPU_VENDOR=""
if command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
    echo "Detected AMD GPU"
    rocm-smi --showproductname 2>/dev/null || true
    rocminfo 2>/dev/null | grep -i "gfx" | head -3 || true
    echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    [ -d /dev/dri ] && ls -la /dev/dri/card* 2>/dev/null || true
elif command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    echo "Detected NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "Warning: No GPU detection tool found"
fi

python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || echo "vLLM not installed"
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "PyTorch GPU check failed"

# Set GPU visibility if needed
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Warning: No GPUs visible, attempting auto-detect..."
    if [ -d /dev/dri ]; then
        GPU_IDS=$(ls -1 /dev/dri/card* 2>/dev/null | grep -v render | sed 's/.*card//' | tr '\n' ',' | sed 's/,$//')
        if [ -n "$GPU_IDS" ]; then
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            export HIP_VISIBLE_DEVICES="$GPU_IDS"
            echo "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"
        fi
    fi
fi

export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}/hub
mkdir -p "$HF_HOME"

echo "GPU_VENDOR=$GPU_VENDOR"
