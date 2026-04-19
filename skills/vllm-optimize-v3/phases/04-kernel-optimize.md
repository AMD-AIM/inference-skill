# Phase 4: Kernel Optimization

**Phase name**: `optimize`  **Phase number**: 4

## Objective
Optimize bottleneck kernels. **Always run Step 0 (TunableOps) first** — it is fast (3-5 min), requires no code writing, and consistently yields 20-50%+ gains on AMD. Only proceed to Step 1+ (Triton) if TunableOps gain is insufficient.

## AMD vLLM GEMM shapes: what to expect
- vLLM uses `BFloat16_TN` format (F.linear: `x @ weight.T`)
- Projections are fused: QKV merged, gate+up merged → actual N values differ from model config
- rocBLAS typically beats hipBLASLt for small-M decode (M = active batch size)
- `PYTORCH_TUNABLEOP_FILENAME` env var does **NOT** auto-load CSV in PyTorch 2.9+; use `sitecustomize.py` injection

## Execution

```bash
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_4_optimize.log"
mkdir -p "{{OUTPUT_DIR}}/logs" "{{OPTIMIZED_DIR}}"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'optimize','status':'running','phase_4_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail
echo "[$(date +%T)] === Phase 4/6: optimize — STARTING ==="

# Config summary
echo "[$(date +%T)] Config:"
echo "  Untuned shapes:  {{RESULTS_DIR}}/untuned_shapes_final.csv"
echo "  Real shapes:     {{RESULTS_DIR}}/real_shapes.json"
echo "  Output:          {{OPTIMIZED_DIR}}"
echo "  Max attempts:    {{MAX_OPTIMIZATION_ATTEMPTS}} / {{MAX_CONSECUTIVE_REJECTIONS}} max rejections"

python3 -u - << 'PYEOF'
import json, os
# Show what we're optimizing
f = "{{PROBLEMS_DIR}}/targets.json"
if os.path.exists(f):
    d = json.load(open(f))
    print(f"  Targets to optimize:")
    for t in d.get("targets",[]):
        status = "[valid]" if t["has_real_shapes"] else "[skip: no shapes]"
        print(f"    {t['kernel_type']:<16} {t['pct_total']:>5.1f}% of GPU active  {status}")
untuned = "{{RESULTS_DIR}}/untuned_shapes_final.csv"
if os.path.exists(untuned):
    n = sum(1 for l in open(untuned) if l.startswith("Gemm"))
    print(f"  Untuned GEMM shapes: {n} (all concurrencies)")
rs = "{{RESULTS_DIR}}/real_shapes.json"
if os.path.exists(rs):
    d = json.load(open(rs))
    print(f"  Real shapes (from trace): {d['total_shapes']} unique  M_values={d['unique_m_values']}")
PYEOF

# ══════════════════════════════════════════════════════════════════════
# STEP 0: TunableOps rocBLAS Offline Tuning (default first step)
# ══════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] [Step 0a] TunableOps offline GEMM tuning..."
echo "  Input:  {{RESULTS_DIR}}/untuned_shapes_final.csv"
echo "  Output: {{OPTIMIZED_DIR}}/tuned_gemm.csv"
T_TUNE_START=$(date +%s)

# Select a GPU for offline tuning that is DIFFERENT from the vLLM serving GPU.
# Running tuning on the same GPU as vLLM causes PyTorch CUDA context contention
# and results in a 30+ minute silent hang.
VLLM_GPU=$(cat "{{OUTPUT_DIR}}/gpu_selection.txt" 2>/dev/null || echo "")
TUNE_GPU=$(python3 - << 'PYEOF'
import subprocess, os, sys
vllm_gpu_str = open("{{OUTPUT_DIR}}/gpu_selection.txt").read().strip() \
    if os.path.exists("{{OUTPUT_DIR}}/gpu_selection.txt") else ""
vllm_gpus = set(vllm_gpu_str.split(',')) if vllm_gpu_str else set()
# Enumerate all GPUs via rocm-smi
r = subprocess.run(['rocm-smi','--showuse','--csv'], capture_output=True, text=True, timeout=10)
all_gpus = []
for line in r.stdout.splitlines():
    parts = [p.strip() for p in line.split(',')]
    if parts and parts[0].isdigit():
        all_gpus.append(parts[0])
if not all_gpus:
    import torch
    all_gpus = [str(i) for i in range(torch.cuda.device_count())]
# Pick least-busy GPU that vLLM is not using
for g in all_gpus:
    if g not in vllm_gpus:
        print(g); sys.exit(0)
# Fallback: same GPU (will warn below)
print(all_gpus[0] if all_gpus else "0")
PYEOF
)
export CUDA_VISIBLE_DEVICES="$TUNE_GPU"
if [ "$TUNE_GPU" = "$VLLM_GPU" ]; then
    echo "  WARNING: no free GPU found — tuning on same GPU as vLLM ($VLLM_GPU). May contend."
else
    echo "  Tuning GPU: $TUNE_GPU  (vLLM GPU: ${VLLM_GPU:-unknown})  [SEPARATE — no contention]"
fi

python3 {{SCRIPTS_DIR}}/tune_gemm_shapes.py \
    --untuned  "{{RESULTS_DIR}}/untuned_shapes_final.csv" \
    --output   "{{OPTIMIZED_DIR}}/tuned_gemm.csv" \
    --max-iter 50 \
    --max-duration 20

T_TUNE_END=$(date +%s)
echo "[$(date +%T)] [Step 0a] Tuning done. ($(( T_TUNE_END - T_TUNE_START ))s elapsed)"

N_TUNED=$(grep -c "^Gemm" "{{OPTIMIZED_DIR}}/tuned_gemm.csv" 2>/dev/null || echo 0)
echo "  Tuned entries: $N_TUNED"
[ "$N_TUNED" -eq 0 ] && { echo "  [FAIL] No tuned entries generated"; exit 1; }
echo "  [OK] tuned_gemm.csv ready"

# ── Step 0b: Micro-benchmark comparison (default vs tuned) ─────────────
echo "[$(date +%T)] [Step 0b] Micro-benchmark: default vs tuned (key decode shapes)..."

python3 -u - << 'PYEOF'
import torch, torch.nn.functional as F, time, json, os

def bench(fn, reps=300, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps*1e6

# Load real shapes
rs = json.load(open("{{RESULTS_DIR}}/real_shapes.json"))
shapes = rs.get("top_shapes_by_calls", [])[:5]
if not shapes:
    print("  WARNING: No real shapes to benchmark"); exit(0)

print(f"  {'Shape MKN':<30}  {'Default(us)':>12}  {'Tuned(us)':>10}  {'Speedup':>8}  Status")
print(f"  {'-'*30}  {'-'*12}  {'-'*10}  {'-'*8}  ------")

# Default (no TunableOps)
torch.cuda.tunable.enable(val=False)
default_times = {}
for entry in shapes:
    M, K, N = entry["MKN"]
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    t = bench(lambda: F.linear(x, w))
    default_times[(M,K,N)] = t

# Tuned
torch.cuda.tunable.enable(val=True)
torch.cuda.tunable.tuning_enable(val=False)
torch.cuda.tunable.read_file("{{OPTIMIZED_DIR}}/tuned_gemm.csv")
loaded = len(torch.cuda.tunable.get_results())
print(f"  Loaded {loaded} tuned entries from CSV")

total_default, total_tuned = 0, 0
best_speedup = 0
for entry in shapes:
    M, K, N = entry["MKN"]
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    t_tuned = bench(lambda: F.linear(x, w))
    t_def = default_times[(M,K,N)]
    speedup = t_def / t_tuned if t_tuned > 0 else 0
    total_default += t_def
    total_tuned   += t_tuned
    best_speedup   = max(best_speedup, speedup)
    status = "[IMPROVED]" if speedup > 1.05 else "[~SAME]" if speedup > 0.95 else "[REGRESSED]"
    print(f"  M={M:<4d} K={K:<5d} N={N:<7d}  {t_def:>11.1f}us  {t_tuned:>9.1f}us  {speedup:>7.3f}x  {status}")

overall_speedup = total_default / total_tuned if total_tuned > 0 else 0
verdict = "[PROCEED]" if overall_speedup > 1.05 else "[MARGINAL]" if overall_speedup > 0.98 else "[NO GAIN]"
print(f"\n  Overall micro-speedup: {overall_speedup:.3f}x  best_single: {best_speedup:.3f}x  {verdict}")
PYEOF

# ── Step 0c: Create injection shim ─────────────────────────────────────
echo "[$(date +%T)] [Step 0c] Creating sitecustomize.py injection shim..."

python3 {{SCRIPTS_DIR}}/create_inject.py \
    --tuned-csv  "{{OPTIMIZED_DIR}}/tuned_gemm.csv" \
    --output-dir "{{OPTIMIZED_DIR}}/pypath"

echo "  [OK] {{OPTIMIZED_DIR}}/pypath/sitecustomize.py ready"

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Triton Kernel Optimization (only if TunableOps gain < 10%)
# ══════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] Checking if Triton optimization is needed..."
python3 -u - << 'PYEOF'
import torch, torch.nn.functional as F, time, json

rs = json.load(open("{{RESULTS_DIR}}/real_shapes.json"))
shapes = rs.get("top_shapes_by_calls", [])[:5]

torch.cuda.tunable.enable(val=True)
torch.cuda.tunable.tuning_enable(val=False)
torch.cuda.tunable.read_file("{{OPTIMIZED_DIR}}/tuned_gemm.csv")

def bench(fn, reps=200, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps*1e6

total_default, total_tuned = 0, 0
torch.cuda.tunable.enable(val=False)
defaults = {}
for e in shapes:
    M,K,N = e["MKN"]
    x = torch.randn(M,K,dtype=torch.bfloat16,device='cuda')
    w = torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
    defaults[(M,K,N)] = bench(lambda: F.linear(x,w))

torch.cuda.tunable.enable(val=True)
for e in shapes:
    M,K,N = e["MKN"]
    x = torch.randn(M,K,dtype=torch.bfloat16,device='cuda')
    w = torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
    total_default += defaults[(M,K,N)]
    total_tuned   += bench(lambda: F.linear(x,w))

speedup = total_default / total_tuned if total_tuned > 0 else 1.0
with open("{{OPTIMIZED_DIR}}/tunableops_speedup.txt","w") as f:
    f.write(str(round(speedup,4)))

if speedup >= 1.10:
    print(f"  TunableOps gain: {speedup:.3f}x >= 1.10x  → Triton optimization NOT needed  [SKIP Triton]")
else:
    print(f"  TunableOps gain: {speedup:.3f}x < 1.10x  → Triton optimization RECOMMENDED  [PROCEED to Triton]")
    print(f"  (Read Phase 4 Triton section and {{SKILL_DIR}}/references/TRITON_KNOWLEDGE.md)
  Note: check shape coverage — if coverage < 50% of real call volume, E2E gain will be limited")
PYEOF

# ── Completion ─────────────────────────────────────────────────────────
SPEEDUP=$(cat "{{OPTIMIZED_DIR}}/tunableops_speedup.txt" 2>/dev/null || echo "?")
echo "[$(date +%T)] === Phase 4/6: optimize — DONE  tunableops_speedup=${SPEEDUP}x ==="

} 2>&1 | tee -a "$PHASE_LOG"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':list(dict.fromkeys(p.get('phases_completed',[])+['optimize'])),
          'current_phase':None,'status':'idle','phase_4_done':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```

## Triton Kernel Optimization (Step 1+, if TunableOps insufficient)

Only proceed here if Step 0 overall speedup < 1.10x.

For each target in `{{PROBLEMS_DIR}}/targets.json`:

```bash
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_4_optimize.log"

{
set -euo pipefail
echo "[$(date +%T)] [Triton] Setting up kernel workspaces..."
CUDA_VISIBLE_DEVICES=<selected_gpu> python3 {{SCRIPTS_DIR}}/kernel_agent.py setup \
    --targets      "{{PROBLEMS_DIR}}/targets.json" \
    --real-shapes  "{{RESULTS_DIR}}/real_shapes.json" \
    --model-config "{{MODEL}}/config.json" \
    --gpu-arch     "{{RESULTS_DIR}}/gpu_arch.json" \
    --output-dir   "{{OPTIMIZED_DIR}}" \
    --max-attempts {{MAX_OPTIMIZATION_ATTEMPTS}} \
    --max-rejections {{MAX_CONSECUTIVE_REJECTIONS}} \
    --knowledge-base "{{SKILL_DIR}}/references/TRITON_KNOWLEDGE.md"
} 2>&1 | tee -a "$PHASE_LOG"
```

For each optimization attempt, run correctness → benchmark → accept/reject → serving-test in sequence.

**Serving test is MANDATORY before accepting any kernel.** See `references/TRITON_KNOWLEDGE.md` for autotune regression risk.

Log each attempt:
```bash
{
echo "[$(date +%T)] [Triton attempt N] <kernel_type>: <what_was_tried>"
python3 {{SCRIPTS_DIR}}/kernel_agent.py correctness --kernel attempt_N.py --shapes '...' \
    && echo "  Correctness: [PASS]" || { echo "  Correctness: [FAIL]"; continue; }
python3 {{SCRIPTS_DIR}}/kernel_agent.py benchmark   --kernel attempt_N.py --shapes '...'
python3 {{SCRIPTS_DIR}}/kernel_agent.py serving-test --kernel attempt_N.py --n N --k K --m-values "1,2,4,8,16,32,64"
echo "  Serving test verdict: <PASS|FAIL>"
} 2>&1 | tee -a "$PHASE_LOG"
```
