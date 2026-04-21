# Phase 3: Analysis & Target Selection

**Phase name**: `analysis`  **Phase number**: 3

## Objective
1. Benchmark throughput/latency summary  
2. GPU kernel breakdown at each concurrency (utilization, GEMM%, Attention%, idle gaps, top-20 kernels)  
3. Real GEMM shapes + optimization targets

## Execution

```bash
mkdir -p "{{RESULTS_DIR}}/gap_analysis"
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_3_analysis.log"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'analysis','status':'running','phase_3_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail

echo "[$(date +%T)] === Phase 3/6: analysis — STARTING ==="

# ── Step 1: Benchmark summary ──────────────────────────────────────────────
echo "[$(date +%T)] [Step 1] Benchmark summary..."
python3 -u - << 'PYEOF'
import json, sys

d = json.load(open("{{RESULTS_DIR}}/benchmark_report.json"))
isl, osl = d["config"]["ISL"], d["config"]["OSL"]
print(f"  ISL={isl}  OSL={osl}  TP={{TP}}  dtype={{DTYPE}}")
print(f"  {'Conc':>4}  {'OutTPS':>8}  {'TotTPS':>8}  {'P50':>8}  {'P90':>8}  {'Fails':>5}  Status")
print("  " + "-"*62)
for r in d.get("benchmark_results",[]):
    if r.get("total_tps",0) > 0:
        status = "[OK]" if r.get("failures",0)==0 else f"[WARN:{r['failures']} fails]"
        print(f"  {r['concurrency']:>4}  {r.get('output_tps',0):>8.1f}"
              f"  {r['total_tps']:>8.1f}"
              f"  {r.get('lat_p50_s',0):>7.3f}s"
              f"  {r.get('lat_p90_s',0):>7.3f}s"
              f"  {r.get('failures',0):>5}  {status}")
    else:
        print(f"  {r['concurrency']:>4}  {'FAIL':>8}  [FAIL: all requests failed]")
json.dump(d, open("{{RESULTS_DIR}}/benchmark_summary.json","w"), indent=2)
print("  Saved: benchmark_summary.json")
PYEOF

# ── Step 2: GPU kernel breakdown ───────────────────────────────────────────
# Correctness note: ONLY cat='kernel','gpu_memcpy','gpu_memset' are real GPU execution.
# cat='cpu_op','cuda_runtime' are CPU-side — their dur is CPU time, NOT GPU time.
echo "[$(date +%T)] [Step 2] GPU kernel breakdown at each concurrency..."
T_START=$(date +%s)

# run_breakdown.py launches one kernel_breakdown.py per trace dir IN PARALLEL,
# then collects output in concurrency order. Serial parsing took ~45s per trace.
python3 {{SCRIPTS_DIR}}/run_breakdown.py \
    --profile-meta "{{RESULTS_DIR}}/profile_meta.json" \
    --scripts-dir  "{{SCRIPTS_DIR}}" \
    --output-dir   "{{RESULTS_DIR}}/gap_analysis" \
    --top-n 30

T_END=$(date +%s)
echo "[$(date +%T)] [Step 2] Kernel breakdown done. ($(( T_END - T_START ))s elapsed)"

# ── Step 3: Multi-concurrency comparison table ─────────────────────────────
echo "[$(date +%T)] [Step 3] Comparison table..."
python3 -u - << 'PYEOF'
import json, os, glob, sys

gaps = {}
for f in sorted(glob.glob("{{RESULTS_DIR}}/gap_analysis/gap_c*.json")):
    conc = int(f.split("_c")[-1].replace(".json",""))
    gaps[conc] = json.load(open(f))

if not gaps:
    print("  ERROR: No gap_analysis results found  [FAIL]"); sys.exit(1)

print()
print(f"  {'Conc':>5}  {'GPU%':>6}  {'Idle%':>6}  {'GEMM%':>6}  {'Attn%':>6}  {'MaxGap':>9}  Bottleneck")
print("  "+"-"*80)
for conc in sorted(gaps):
    g    = gaps[conc]
    util = g.get("gpu_util_pct",0)
    idle = 100-util
    cat  = g.get("category_breakdown",{})
    gemm = cat.get("GEMM",{}).get("pct_active",0)
    attn = cat.get("Attention",{}).get("pct_active",0)
    gs   = g.get("gap_stats",{})
    maxg = gs.get("max_us",0)/1000
    n1ms = gs.get("n_gaps_gt_1ms",0)
    if   idle>10:   verdict=f"CPU scheduling (idle={idle:.0f}%, {n1ms} gaps>1ms)"
    elif gemm>70:   verdict=f"GEMM compute ({gemm:.0f}%) — TunableOps target"
    elif attn>20:   verdict=f"Attention ({attn:.0f}%) — KV cache growth"
    else:           verdict="Mixed"
    print(f"  {conc:>5}  {util:>5.1f}%  {idle:>5.1f}%  {gemm:>5.1f}%  {attn:>5.1f}%"
          f"  {maxg:>8.2f}ms  {verdict}")

# Save combined gap_analysis.json — primary = highest concurrency
primary = max(gaps.keys())
pg = dict(gaps[primary])   # shallow copy to avoid circular reference
# Include other concurrencies WITHOUT self-reference
pg["all_concurrency_results"] = {str(k):v for k,v in gaps.items() if k != primary}
with open("{{RESULTS_DIR}}/gap_analysis.json","w") as f:
    json.dump(pg, f, indent=2)

covered = sum(d.get("pct_active",0) for d in pg.get("category_breakdown",{}).values())
status = "[OK]" if covered >= 80 else f"[WARN: coverage {covered:.0f}% < 80%]"
print(f"\n  Primary: conc={primary}  coverage={covered:.0f}%  {status}")
print(f"  Saved: gap_analysis.json")
PYEOF

# ── Step 4: Extract real GEMM shapes ──────────────────────────────────────
echo "[$(date +%T)] [Step 4] Extracting real GEMM shapes from trace (record_shapes=True required)..."

# Extract real shapes from ALL profiled traces and take the union.
# This is critical for correctness when using small concurrency levels (e.g., 1,4,16):
# the peak trace (conc=16) captures only M=16 GEMM shapes. Shapes at M=1 (conc=1) and
# M=4 (conc=4) are only visible in the lower-concurrency traces. Without unioning,
# tuning only covers the peak concurrency and conc=1/4 see zero E2E improvement.
python3 -u - << 'PYEOF'
import json, os, subprocess, sys, glob

meta    = json.load(open("{{RESULTS_DIR}}/profile_meta.json"))
results_dir = "{{RESULTS_DIR}}"
scripts_dir = "{{SCRIPTS_DIR}}"
merged_mkn  = {}   # (M,K,N) → total_calls

for cs, m in sorted(meta.items(), key=lambda x: int(x[0])):
    td = m["trace_dir"]
    if not glob.glob(td + "/*.json*"):
        print(f"  SKIP conc={cs}: no traces", flush=True); continue
    tmp_out = f"{results_dir}/real_shapes_c{cs}.json"
    r = subprocess.run([sys.executable, f"{scripts_dir}/extract_shapes.py",
                        "--trace-dir", td, "--output", tmp_out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  WARN: extract_shapes failed for conc={cs}: {r.stderr[:200]}", flush=True)
        continue
    d = json.load(open(tmp_out))
    n_shapes = d.get("total_shapes", 0)
    m_vals   = d.get("unique_m_values", [])
    print(f"  conc={cs}: {n_shapes} shapes  M_values={m_vals}", flush=True)
    for entry in d.get("shapes", []) + d.get("top_shapes_by_calls", []):
        key = tuple(entry["MKN"])
        merged_mkn[key] = merged_mkn.get(key, 0) + entry.get("calls", 1)

# Build merged real_shapes.json
all_shapes = sorted(merged_mkn.keys(), key=lambda k: -merged_mkn[k])
m_vals = sorted(set(k[0] for k in all_shapes))
out = {
    "total_shapes": len(all_shapes),
    "unique_m_values": m_vals,
    "shapes": [{"MKN": list(k), "calls": merged_mkn[k]} for k in all_shapes],
    "top_shapes_by_calls": [{"MKN": list(k), "calls": merged_mkn[k]} for k in all_shapes[:20]],
    "source": "union of all profiled traces",
}
with open(f"{results_dir}/real_shapes.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"  Merged real shapes: {len(all_shapes)} unique  M_values={m_vals}  [OK]", flush=True)
PYEOF

python3 -u - << 'PYEOF'
import json, os, sys

f = "{{RESULTS_DIR}}/real_shapes.json"
if not os.path.exists(f):
    print("  ERROR: real_shapes.json not found  [FAIL]"); sys.exit(1)
d = json.load(open(f))
n = d.get("total_shapes",0)
m_vals = d.get("unique_m_values",[])
if n == 0:
    print(f"  Real GEMM shapes: 0  [FAIL]")
    print(f"  Fix: profiler must be started with record_shapes=True in --profiler-config")
    sys.exit(1)
print(f"  Real GEMM shapes: {n} unique  M_values={m_vals}  [OK]")
print(f"  Top 5 by call count:")
for entry in d.get("top_shapes_by_calls",[])[:5]:
    print(f"    M={entry['MKN'][0]}  K={entry['MKN'][1]}  N={entry['MKN'][2]}  calls={entry['calls']}")
PYEOF

# ── Step 5: Optimization targets ──────────────────────────────────────────
echo "[$(date +%T)] [Step 5] Identifying optimization targets..."
python3 -u - << 'PYEOF'
import json, os, sys

gap = json.load(open("{{RESULTS_DIR}}/gap_analysis.json"))
THRESHOLD = {{OPTIMIZE_PRIORITY_THRESHOLD}}
try:
    rs = json.load(open("{{RESULTS_DIR}}/real_shapes.json"))
    has_shapes = rs.get("total_shapes",0) > 0
except FileNotFoundError:
    has_shapes = False

seen, targets = set(), []
for k in gap.get("top_kernels",[]):
    pct = k.get("pct_total",0)
    if pct < THRESHOLD: continue
    ktype = k.get("category","other")
    if ktype in ("Other","Element-wise") or ktype in seen: continue
    seen.add(ktype)
    targets.append({"kernel_type":ktype,"kernel_name":k["name"],
                    "pct_total":round(pct,2),"calls":k.get("calls",0),
                    "avg_us":round(k.get("avg_us",0),2),
                    "has_real_shapes":has_shapes,
                    "real_shape_count":rs.get("total_shapes",0) if has_shapes else 0})

os.makedirs("{{PROBLEMS_DIR}}", exist_ok=True)
with open("{{PROBLEMS_DIR}}/targets.json","w") as f:
    json.dump({"targets":targets,"threshold_pct":THRESHOLD},f,indent=2)

valid = [t for t in targets if t["has_real_shapes"]]
print(f"  Threshold: {THRESHOLD}%  —  {len(targets)} targets found, {len(valid)} valid (have real shapes)")
print(f"  {'Type':<16}  {'%GPU':>6}  {'calls':>7}  {'avg_us':>8}  {'shapes':>7}  Status")
print("  "+"-"*60)
for t in targets:
    shapes_note = f"{t['real_shape_count']:>7}" if t["has_real_shapes"] else "      0"
    status = "[OPTIMIZE]" if t["has_real_shapes"] else "[SKIP: no shapes]"
    print(f"  {t['kernel_type']:<16}  {t['pct_total']:>5.1f}%  "
          f"{t['calls']:>7}  {t['avg_us']:>7.1f}us  {shapes_note}  {status}")
if len(valid) == 0:
    print("  WARNING: No valid optimization targets — check GEMM shape extraction")
PYEOF

# ── Completion ─────────────────────────────────────────────────────────────
N_TARGETS=$(python3 -c "
import json,os
f='{{PROBLEMS_DIR}}/targets.json'
t=json.load(open(f)) if os.path.exists(f) else {'targets':[]}
print(len([x for x in t.get('targets',[]) if x.get('has_real_shapes')]))
" 2>/dev/null || echo "?")
echo "[$(date +%T)] === Phase 3/6: analysis — DONE  valid_targets=${N_TARGETS} ==="

} 2>&1 | tee -a "$PHASE_LOG"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':list(dict.fromkeys(p.get('phases_completed',[])+['analysis'])),
          'current_phase':None,'status':'idle','phase_3_done':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```

**Proceed to Phase 4 only if valid_targets > 0.**
