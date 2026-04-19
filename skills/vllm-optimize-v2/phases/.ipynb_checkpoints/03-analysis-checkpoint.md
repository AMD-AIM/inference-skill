# Phase 3: Analysis & Target Selection

**Phase name**: `analysis`

## Objective
Extract three things that drive Phase 4:
1. Benchmark throughput/latency summary
2. GPU kernel time breakdown (% of total GPU time)
3. **Real GEMM shapes** from profiler traces (NOT from model config)

Then validate that enough data exists before proceeding to optimization.

---

## Step 1: Benchmark Summary

```bash
python3 << 'PYEOF'
import json, os

with open("{{RESULTS_DIR}}/benchmark_report.json") as f:
    data = json.load(f)

tp = {{TP}}
isl = data["config"]["ISL"]
osl = data["config"]["OSL"]

print(f"\nThroughput & Latency Summary  (ISL={isl} OSL={osl} TP={tp})")
print(f"{'Conc':>4}  {'TPS':>8}  {'TPS/GPU':>8}  {'OutTPS':>8}  {'Lat_avg':>8}  {'Lat_p90':>8}  {'Failures':>8}")
print("-" * 70)

for r in data.get("benchmark_results", []):
    if r.get("total_tps", 0) > 0:
        tps_gpu = r["total_tps"] / tp
        print(f"{r['concurrency']:>4}  {r['total_tps']:>8.1f}  {tps_gpu:>8.1f}  "
              f"{r.get('output_tps',0):>8.1f}  {r.get('lat_avg_s',0):>7.3f}s  "
              f"{r.get('lat_p90_s',0):>7.3f}s  {r.get('failures',0):>8}")
    else:
        print(f"{r['concurrency']:>4}  {'FAIL':>8}")

# Save summary
os.makedirs("{{RESULTS_DIR}}", exist_ok=True)
with open("{{RESULTS_DIR}}/benchmark_summary.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved: {{RESULTS_DIR}}/benchmark_summary.json")
PYEOF
```

## Step 2: GPU Kernel Breakdown at Each Concurrency Level

**Analysis correctness rules (learned from real measurements)**:
- Only `cat='kernel'`, `cat='gpu_memcpy'`, `cat='gpu_memset'` events are real GPU execution.
- `cat='cpu_op'` and `cat='cuda_runtime'` events are CPU-side — their `dur` is CPU time, NOT GPU time. Mixing them gives wrong proportions.
- GPU utilization = union(kernel intervals) / trace wall time.
- The bottleneck shifts with concurrency: low conc → CPU scheduling gaps dominate; high conc → GEMM/Attention compute.

```bash
python3 << 'PYEOF'
import json, os

meta = json.load(open("{{RESULTS_DIR}}/profile_meta.json"))
os.makedirs("{{RESULTS_DIR}}/gap_analysis", exist_ok=True)

for conc_str, m in sorted(meta.items(), key=lambda x: int(x[0])):
    conc = int(conc_str)
    trace_dir = m["trace_dir"]
    import glob
    if not glob.glob(trace_dir + "/*.json*"):
        print(f"SKIP conc={conc}: no traces")
        continue
    out = f"{{RESULTS_DIR}}/gap_analysis/gap_c{conc}.json"
    print(f"\nAnalyzing conc={conc}...")
    import subprocess, sys
    r = subprocess.run([
        sys.executable, "{{SCRIPTS_DIR}}/kernel_breakdown.py",
        "--trace-dir", trace_dir,
        "--output", out,
        "--label", f"conc={conc} ({'decode-only' if m.get('decode_only') else 'prefill+decode'})",
        "--top-n", "30",
    ], capture_output=False)
    if r.returncode != 0:
        print(f"ERROR analyzing conc={conc}")
PYEOF
```

Produce comparative summary across all concurrency levels:
```bash
python3 << 'PYEOF'
import json, os, glob

results_dir = "{{RESULTS_DIR}}/gap_analysis"
gaps = {}
for f in sorted(glob.glob(f"{results_dir}/gap_c*.json")):
    conc = int(f.split("_c")[-1].replace(".json",""))
    gaps[conc] = json.load(open(f))

print()
print(f"{'Conc':>5}  {'GPU%':>6}  {'Idle%':>6}  {'GEMM%':>6}  {'Attn%':>6}  {'MaxGap':>8}  Bottleneck")
print("-" * 85)
for conc in sorted(gaps):
    g = gaps[conc]
    util  = g.get("gpu_util_pct", 0)
    idle  = 100 - util
    cat   = g.get("category_breakdown", {})
    gemm  = cat.get("GEMM", {}).get("pct_active", 0)
    attn  = cat.get("Attention", {}).get("pct_active", 0)
    gs    = g.get("gap_stats", {})
    maxg  = gs.get("max_us", 0)
    bigs  = gs.get("n_gaps_gt_1ms", 0)
    
    # Verdict
    if idle > 10: verdict = f"CPU scheduling (idle={idle:.0f}%)"
    elif gemm > 70: verdict = f"GEMM compute ({gemm:.0f}%)"
    elif attn > 20: verdict = f"Attention ({attn:.0f}%)"
    else: verdict = "Mixed"
    
    print(f"  {conc:>3}  {util:>5.1f}%  {idle:>5.1f}%  {gemm:>5.1f}%  {attn:>5.1f}%  {maxg/1e3:>7.1f}ms  {verdict}")

# Save combined gap_analysis.json (use highest concurrency as primary)
primary_conc = max(gaps.keys())
primary = gaps[primary_conc]
primary["all_concurrency_results"] = gaps
with open("{{RESULTS_DIR}}/gap_analysis.json", "w") as f:
    json.dump(primary, f, indent=2)
print(f"\nPrimary gap_analysis.json = conc={primary_conc}")

# Validate: check coverage ≥80% for the primary concurrency
primary_cat = gaps[primary_conc].get("category_breakdown", {})
covered = sum(d.get("pct_active", 0) for d in primary_cat.values())
if covered < 80:
    print(f"WARNING: coverage {covered:.0f}% < 80% — trace may be too short or profiler stopped early")
else:
    print(f"Coverage: {covered:.0f}% of GPU active time categorized  (target ≥80%)")
PYEOF
```

## Step 3: Extract Real GEMM Shapes from Trace

**This is the most critical step for kernel optimization.** Shapes derived from model config are NOT acceptable (Constraint 1).

```bash
python3 {{SCRIPTS_DIR}}/extract_shapes.py \
    --trace-dir "{{PROFILE_DIR}}" \
    --output "{{RESULTS_DIR}}/real_shapes.json"
```

Validate shapes:
```bash
python3 << 'PYEOF'
import json, sys

shapes_path = "{{RESULTS_DIR}}/real_shapes.json"
try:
    with open(shapes_path) as f:
        shapes = json.load(f)
except FileNotFoundError:
    print("WARNING: real_shapes.json not found — will use model-config fallback in Phase 4")
    sys.exit(0)

shapes_list = shapes.get("benchmark_shapes", [])
m_values = sorted(set(s[0][0] for s in shapes_list if s))

print(f"Real shapes extracted: {len(shapes_list)} unique shapes")
print(f"Real M values: {m_values}")

# Must have BOTH small (decode) and larger (batch) M values
has_decode = any(m <= 4 for m in m_values)
has_batch  = any(m >= 8 for m in m_values)

if not has_decode:
    print("WARNING: no small M values (decode) found — profile may not have captured decode phase")
if not has_batch:
    print("WARNING: no larger M values (batch) found — profile may be too short")

if not shapes_list:
    print("WARNING: no GEMM shapes found — kernel optimization will use model-config fallback")
else:
    print("Real shapes OK — ready for Phase 4")
PYEOF
```

## Step 4: Identify Optimization Targets

Select kernels above the priority threshold and validate they have real shapes:

```bash
python3 << 'PYEOF'
import json, os, re

with open("{{RESULTS_DIR}}/gap_analysis.json") as f:
    gap = json.load(f)

threshold = {{OPTIMIZE_PRIORITY_THRESHOLD}}  # % of GPU time

# Load real shapes
real_shapes = {}
try:
    with open("{{RESULTS_DIR}}/real_shapes.json") as f:
        rs = json.load(f)
    real_shapes = {"gemm": rs.get("benchmark_shapes", [])}
except FileNotFoundError:
    pass

# Kernel type classifier
def classify(name):
    n = name.lower()
    if any(p in n for p in ['gemm', 'mm', 'matmul', 'wvsplitk', 'ck_tile', 'cijk_', 'hipblaslt']):
        return 'gemm'
    if any(p in n for p in ['rmsnorm', 'rms_norm', 'layernorm', 'fused_add_rms']):
        return 'rmsnorm'
    if any(p in n for p in ['silu', 'swiglu', 'act_and_mul', 'gelu_and_mul']):
        return 'swiglu'
    if any(p in n for p in ['attention', 'paged_attn', 'flash_attn']):
        return 'attention'
    if any(p in n for p in ['rotary', 'rope']):
        return 'rotary'
    return 'other'

targets = []
seen_types = set()

for k in gap.get("top_kernels", []):
    pct = k.get("pct_total", 0)
    if pct < threshold:
        continue
    ktype = classify(k["name"])
    if ktype == "other":
        continue
    if ktype in seen_types:
        continue   # one target per type
    seen_types.add(ktype)

    shapes = real_shapes.get(ktype, [])
    target = {
        "kernel_type": ktype,
        "kernel_name": k["name"],
        "pct_total": round(pct, 2),
        "calls": k.get("calls", 0),
        "avg_us": round(k.get("avg_us", 0), 2),
        "has_real_shapes": len(shapes) > 0,
        "real_shape_count": len(shapes),
    }
    targets.append(target)

os.makedirs("{{PROBLEMS_DIR}}", exist_ok=True)
manifest = {"targets": targets, "threshold": threshold,
            "total_kernel_time_us": gap.get("total_kernel_time_us", 0)}
with open("{{PROBLEMS_DIR}}/targets.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nOptimization targets (threshold={threshold}%):")
for t in targets:
    shapes_note = f"{t['real_shape_count']} real shapes" if t["has_real_shapes"] else "NO real shapes — WILL SKIP"
    print(f"  [{t['pct_total']:.1f}%] {t['kernel_type']:10s}  {t['kernel_name'][:50]}  ({shapes_note})")

# Validate: skip targets without real shapes (Constraint 1)
valid = [t for t in targets if t["has_real_shapes"]]
skipped = [t for t in targets if not t["has_real_shapes"]]
if skipped:
    print(f"\nSkipped (no real shapes): {[t['kernel_type'] for t in skipped]}")
print(f"\nValid optimization targets: {len(valid)}")
PYEOF
```

## Step 5: GPU Architecture Info

```bash
python3 -c "
import json
with open('{{OUTPUT_DIR}}/env_info.json') as f:
    e = json.load(f)
arch = {'gpu_vendor': e['gpu_vendor'], 'gpu_arch': e['gpu_arch'], 'gpu_count': e['gpu_count']}
with open('{{RESULTS_DIR}}/gpu_arch.json', 'w') as f:
    json.dump(arch, f, indent=2)
print(json.dumps(arch))
"
```

## Completion

Update `{{PROGRESS_FILE}}`:
```json
{"phases_completed": ["env", "server", "bench-profile", "analysis"],
 "details": {"targets": "<N>", "real_shapes_found": "<true/false>"}}
```

**Proceed to Phase 4 only if `valid_targets > 0`. If zero valid targets, write report and stop.**
