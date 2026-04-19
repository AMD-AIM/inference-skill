# Phase 0: Environment Setup

**Phase name**: `env`  **Phase number**: 0

## Objective
Detect GPU hardware, verify vLLM/PyTorch, write `env_info.json`.

## Execution

```bash
mkdir -p "{{OUTPUT_DIR}}/logs"
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_0_env.log"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'env','status':'running','phase_0_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail

echo "[$(date +%T)] === Phase 0/6: env — STARTING ==="

# ── Step 1: Detect GPU ─────────────────────────────────────────────────────
echo "[$(date +%T)] [Step 1] Detecting GPU..."

python3 -u - << 'PYEOF'
import subprocess, re, sys

# AMD
r = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
if r.returncode == 0:
    # Take the LONGEST arch string to avoid prefix ambiguity (gfx1100 not gfx11)
    arches = re.findall(r'\bgfx\d+\b', r.stdout)
    arch = max(arches, key=len) if arches else 'unknown'  # e.g. gfx1100, NOT gfxgfx1100

    r2 = subprocess.run(['rocm-smi', '--showuse', '--csv'], capture_output=True, text=True, timeout=10)
    count = 0
    gpu_names = []
    for line in r2.stdout.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[0].isdigit():
            count += 1
    # Get GPU names from rocm-smi --showproductname
    r3 = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=10)
    for line in r3.stdout.splitlines():
        if 'GPU[' in line and 'Card series' in line:
            name = line.split(':', 1)[-1].strip() if ':' in line else 'AMD GPU'
            gpu_names.append(name)
    if not gpu_names:
        gpu_names = [f'AMD GPU (gfx{arch.replace("gfx","")})'] * max(count, 1)

    print(f"  GPU vendor:  AMD")
    print(f"  GPU arch:    {arch}")
    print(f"  GPU count:   {count}")
    for i, name in enumerate(gpu_names[:4]):
        print(f"  GPU {i}:      {name}")
    if count > 4:
        print(f"  ... (+{count-4} more)")

    # Arch-specific features
    if arch.startswith('gfx11'):
        note = 'RDNA3 — Wave32, WMMA, Infinity Cache. rocBLAS often beats hipBLASLt for small-M decode GEMMs.'
    elif arch.startswith('gfx94') or arch.startswith('gfx95'):
        note = 'CDNA3 — Wave64, MFMA, HBM3. Consider PID swizzle across XCDs.'
    elif arch.startswith('gfx90'):
        note = 'CDNA2 — Wave64, MFMA, HBM2e.'
    else:
        note = 'Unknown AMD arch — check TRITON_KNOWLEDGE.md for guidance.'
    print(f"  Arch note:   {note}")

else:
    # NVIDIA
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,compute_cap',
                        '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
        print(f"  GPU vendor:  NVIDIA")
        print(f"  GPU count:   {len(lines)}")
        for l in lines[:4]:
            print(f"  GPU:         {l}")
    else:
        print("  ERROR: No GPU tool found (no rocminfo, no nvidia-smi)")
        sys.exit(1)
PYEOF

# ── Step 2: Verify vLLM and PyTorch ────────────────────────────────────────
echo "[$(date +%T)] [Step 2] Verifying vLLM and PyTorch..."

python3 -u - << 'PYEOF'
import vllm, torch, sys

print(f"  vLLM:        {vllm.__version__}")
print(f"  PyTorch:     {torch.__version__}")

if not torch.cuda.is_available():
    print("  ERROR: torch.cuda not available"); sys.exit(1)

print(f"  CUDA/ROCm:   available  ({torch.cuda.device_count()} device(s))")

# Verify at least one GPU accessible
dev = torch.cuda.get_device_properties(0)
print(f"  GPU 0 props: total_memory={dev.total_memory//1024**3}GB")
PYEOF

# ── Step 3: Write env_info.json ─────────────────────────────────────────────
echo "[$(date +%T)] [Step 3] Writing env_info.json..."

python3 -u - << 'PYEOF'
import subprocess, re, json, datetime, torch, vllm

r = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
arches = re.findall(r'\bgfx\d+\b', r.stdout) if r.returncode == 0 else []
arch   = max(arches, key=len) if arches else 'unknown'
vendor = 'amd' if arches else 'nvidia'

r2 = subprocess.run(['rocm-smi','--showuse'], capture_output=True, text=True, timeout=10)
count = len([l for l in r2.stdout.splitlines() if 'GPU[' in l]) if r2.returncode == 0 else \
        torch.cuda.device_count()

amd_features = {}
if arch.startswith('gfx11'):
    amd_features = {'wave_size':32,'matrix_op':'WMMA',
                    'note':'RDNA3 — rocBLAS often beats hipBLASLt for small-M decode GEMMs'}
elif arch.startswith('gfx94') or arch.startswith('gfx95'):
    amd_features = {'wave_size':64,'matrix_op':'MFMA','note':'CDNA3 — HBM3, multi-XCD'}

info = {'gpu_vendor':vendor,'gpu_arch':arch,'gpu_count':count,
        'amd_features':amd_features,
        'vllm_version':vllm.__version__,'torch_version':torch.__version__,
        'timestamp':datetime.datetime.now().isoformat()}
with open('{{OUTPUT_DIR}}/env_info.json','w') as f:
    json.dump(info, f, indent=2)
print(f"  env_info.json: OK  (vendor={vendor} arch={arch} count={count})")
PYEOF

# ── Completion ─────────────────────────────────────────────────────────────
GPU_ARCH=$(python3 -c "import json; e=json.load(open('{{OUTPUT_DIR}}/env_info.json')); print(e['gpu_vendor']+'/'+e['gpu_arch'])" 2>/dev/null || echo "?")
echo "[$(date +%T)] === Phase 0/6: env — DONE  GPU=${GPU_ARCH} ==="

} 2>&1 | tee -a "$PHASE_LOG"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':list(dict.fromkeys(p.get('phases_completed',[])+['env'])),
          'current_phase':None,'status':'idle','phase_0_done':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```
