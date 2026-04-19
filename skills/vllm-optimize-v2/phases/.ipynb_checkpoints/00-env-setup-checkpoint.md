# Phase 0: Environment Setup

**Phase name**: `env`

## Objective
Detect GPU hardware, verify dependencies, write `env_info.json`. For AMD GPUs, identify architecture-specific optimization features.

---

## Step 1: Detect GPU

```bash
# AMD detection
if command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
    # Use rocm-smi for reliable GPU count
    GPU_COUNT=$(rocm-smi --showuse 2>/dev/null | grep -c 'GPU\[' || echo "0")
    # Use rocminfo for most-specific arch name (gfx1100 over gfx11)
    AMD_ARCHES=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | sort -u)
    GPU_ARCH=$(echo "$AMD_ARCHES" | awk '{ if(length($0)>maxlen) {maxlen=length($0); arch=$0} } END {print arch}')
    echo "AMD GPU: count=$GPU_COUNT arch=$GPU_ARCH"
fi

# NVIDIA detection (only if AMD not found)
if [ -z "$GPU_VENDOR" ] && command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    GPU_ARCH="sm_${CAP}"
    echo "NVIDIA GPU: count=$GPU_COUNT arch=$GPU_ARCH"
fi

# PyTorch fallback
if [ -z "$GPU_VENDOR" ]; then
    python3 -c "
import torch
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    arch = getattr(prop, 'gcnArchName', '').split(':')[0]
    vendor = 'amd' if 'gfx' in arch else 'nvidia'
    print(f'{vendor} {arch} x{torch.cuda.device_count()}')
else:
    print('ERROR: no GPU')
" || { echo "ERROR: No GPU detected"; exit 1; }
fi
```

## Step 2: Verify vLLM and PyTorch

```bash
python3 -c "
import vllm, torch
print(f'vLLM: {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" || { echo "ERROR: vLLM or PyTorch not installed"; exit 1; }
```

## Step 3: AMD Architecture Knowledge (AMD only)

If `GPU_VENDOR=amd`, actively identify architecture-specific features:

- **gfx1100/gfx1101 (RDNA3)**: Wave32, WMMA, 96MB Infinity Cache, BF16 native, `tl.assume` gives 5-15%
- **gfx940/gfx942 (CDNA3 MI300X/MI355X)**: Wave64, MFMA, HBM3, 8 XCDs → need PID swizzle across XCDs
- **gfx908/gfx90a (CDNA1/2)**: Wave64, MFMA, older HBM

Record this in env_info.json so Phase 4 can use it.

## Step 4: Write env_info.json

```bash
python3 << 'PYEOF'
import json, os, subprocess, re, sys

def detect_amd():
    vendor, arch, count = 'unknown', 'unknown', 0
    try:
        r = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            lines = [l for l in r.stdout.splitlines() if 'GPU[' in l]
            if lines:
                vendor = 'amd'
                count = len(lines)
    except Exception:
        pass
    if vendor == 'amd':
        try:
            r = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
            arches = re.findall(r'\bgfx(\d+)\b', r.stdout)
            if arches:
                arch = 'gfx' + max(arches, key=len)
        except Exception:
            pass
    return vendor, arch, count

def detect_nvidia():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            cap = r.stdout.strip().split('\n')[0].replace('.', '')
            r2 = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
            count = int(r2.stdout.strip().split('\n')[0]) if r2.returncode == 0 else 1
            return 'nvidia', f'sm_{cap}', count
    except Exception:
        pass
    return 'unknown', 'unknown', 0

vendor, arch, count = detect_amd()
if vendor == 'unknown':
    vendor, arch, count = detect_nvidia()

# Architecture-specific notes
amd_features = {}
if vendor == 'amd':
    if arch.startswith('gfx11'):
        amd_features = {'wave_size': 32, 'matrix_op': 'WMMA', 'cache': 'Infinity Cache 96MB',
                        'tl_assume_benefit': '5-15%', 'autotune_risk': 'HIGH', 'dtype_native': 'bfloat16'}
    elif arch.startswith('gfx94'):
        amd_features = {'wave_size': 64, 'matrix_op': 'MFMA', 'cache': 'HBM3',
                        'multi_xcd': True, 'xcd_count': 8, 'tl_assume_benefit': '5-15%',
                        'autotune_risk': 'HIGH', 'dtype_native': 'bfloat16'}

info = {
    'gpu_vendor': vendor, 'gpu_arch': arch, 'gpu_count': count,
    'amd_features': amd_features,
}

import vllm, torch
info['vllm_version'] = vllm.__version__
info['torch_version'] = torch.__version__

os.makedirs('{{OUTPUT_DIR}}', exist_ok=True)
with open('{{OUTPUT_DIR}}/env_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
PYEOF
```

## Completion

Update `{{PROGRESS_FILE}}`:
```json
{"phases_completed": ["env"], "current_phase": null, "details": {"gpu_vendor": "<detected>", "gpu_arch": "<detected>"}}
```
