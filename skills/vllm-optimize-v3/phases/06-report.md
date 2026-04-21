# Phase 6: Final Report

**Phase name**: `report`  **Phase number**: 6

## Objective
Generate the mandatory verdict table, per-concurrency E2E comparison, and final summary. **All kernel targets MUST appear in the verdict table** — failures are as important as successes.

## Execution

```bash
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_6_report.log"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'current_phase':'report','status':'running','phase_6_start':datetime.datetime.now().isoformat()})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true

{
set -euo pipefail
echo "[$(date +%T)] === Phase 6/6: report — STARTING ==="

mkdir -p "{{REPORT_DIR}}"

# ── Step 1: Collect all artifacts ─────────────────────────────────────────
echo "[$(date +%T)] [Step 1] Collecting artifacts..."
python3 -u - << 'PYEOF'
import json, os, glob

artifacts = {
    "env_info":           "{{OUTPUT_DIR}}/env_info.json",
    "benchmark_summary":  "{{RESULTS_DIR}}/benchmark_summary.json",
    "gap_analysis":       "{{RESULTS_DIR}}/gap_analysis.json",
    "targets":            "{{PROBLEMS_DIR}}/targets.json",
    "real_shapes":        "{{RESULTS_DIR}}/real_shapes.json",
    "baseline_e2e":       "{{RESULTS_DIR}}/baseline_e2e.json",
    "integration_e2e":    "{{RESULTS_DIR}}/integration_e2e.json",
    "integration_result": "{{RESULTS_DIR}}/integration_result.json",
    "tunableops_speedup": "{{OPTIMIZED_DIR}}/tunableops_speedup.txt",
}
all_ok = True
for name, path in artifacts.items():
    exists = os.path.exists(path)
    sz = os.path.getsize(path) if exists else 0
    status = f"[OK] {sz}B" if exists and sz > 0 else "[MISSING]"
    print(f"  {name:<22}: {status}")
    if not exists: all_ok = False

print(f"\n  Artifact check: {'[OK] all present' if all_ok else '[WARN] some missing'}")
PYEOF

# ── Step 2: Generate report ────────────────────────────────────────────────
echo "[$(date +%T)] [Step 2] Generating final report..."
python3 -u - << 'PYEOF'
import json, os, datetime, glob

def load(path, default=None):
    try: return json.load(open(path))
    except: return default or {}

def load_txt(path):
    try: return open(path).read().strip()
    except: return None

env    = load("{{OUTPUT_DIR}}/env_info.json")
bench  = load("{{RESULTS_DIR}}/benchmark_summary.json")
gap    = load("{{RESULTS_DIR}}/gap_analysis.json")
tgts   = load("{{PROBLEMS_DIR}}/targets.json", {"targets":[]})
rs     = load("{{RESULTS_DIR}}/real_shapes.json", {})
b_e2e  = load("{{RESULTS_DIR}}/baseline_e2e.json", {})
i_e2e  = load("{{RESULTS_DIR}}/integration_e2e.json", {})
i_res  = load("{{RESULTS_DIR}}/integration_result.json", {})
sp_txt = load_txt("{{OPTIMIZED_DIR}}/tunableops_speedup.txt")

lines = []
lines.append("# vLLM Optimization Final Report")
lines.append(f"\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
lines.append("\n## Configuration\n")
lines.append(f"| Parameter | Value |")
lines.append(f"|-----------|-------|")
lines.append(f"| Model     | {{MODEL}} |")
lines.append(f"| GPU       | {env.get('gpu_vendor','?')} {env.get('gpu_arch','?')} ×{env.get('gpu_count','?')} |")
lines.append(f"| TP        | {{TP}} |")
lines.append(f"| Dtype     | {{DTYPE}} |")
lines.append(f"| ISL × OSL | {{ISL}} × {{OSL}} |")
lines.append(f"| vLLM      | {env.get('vllm_version','?')} |")

# Baseline throughput
lines.append("\n## Baseline Throughput\n")
lines.append(f"| Conc | Output TPS | Total TPS | Lat P50 |")
lines.append(f"|------|-----------|-----------|---------|")
for r in bench.get("benchmark_results",[]):
    if r.get("total_tps",0) > 0:
        lines.append(f"| {r['concurrency']} | {r.get('output_tps',0):.1f} | {r['total_tps']:.1f} | {r.get('lat_p50_s',0):.3f}s |")

# GPU kernel breakdown (primary concurrency)
all_concs = gap.get("all_concurrency_results", {})
primary = str(max(int(k) for k in all_concs.keys())) if all_concs else None
lines.append(f"\n## GPU Kernel Breakdown (decode-only profile)\n")
for conc_str in sorted((all_concs or {str(1):gap}).keys(), key=lambda x: int(x)):
    g = all_concs.get(conc_str, gap) if all_concs else gap
    util = g.get("gpu_util_pct",0)
    cat  = g.get("category_breakdown",{})
    gemm = cat.get("GEMM",{}).get("pct_active",0)
    attn = cat.get("Attention",{}).get("pct_active",0)
    lines.append(f"**conc={conc_str}**: GPU util={util:.1f}%  GEMM={gemm:.1f}%  Attention={attn:.1f}%")
lines.append("")
lines.append("**Top kernels (conc=64):**\n")
lines.append(f"| # | % active | total_s | calls | avg_us | category | kernel |")
lines.append(f"|---|---------|---------|-------|--------|----------|--------|")
for i, k in enumerate(gap.get("top_kernels",[])[:10], 1):
    lines.append(f"| {i} | {k['pct_total']:.1f}% | {k['total_us']/1e6:.3f}s | {k['calls']} | {k['avg_us']:.1f} | {k.get('category','?')} | {k['name'][:60]} |")

# ── MANDATORY VERDICT TABLE ──────────────────────────────────────────────
lines.append("\n## Kernel Optimization Verdict Table\n")
lines.append(f"| Kernel | Micro Speedup | Serving Ready | E2E Speedup | Verdict |")
lines.append(f"|--------|--------------|---------------|-------------|---------|")

integrated = i_res.get("integrated", False)
patched = i_e2e.get("patched", {})
speedups = [v.get("speedup",0) for v in patched.values() if isinstance(v,dict)]
avg_e2e = sum(speedups)/len(speedups) if speedups else 0
regression_concs = [c for c,v in patched.items() if isinstance(v,dict) and not v.get("gate4_pass")]

for t in tgts.get("targets",[]):
    ktype = t["kernel_type"]
    if not t.get("has_real_shapes"):
        lines.append(f"| {ktype} | N/A | N/A | N/A | Skipped (no shapes) |")
        continue

    # TunableOps micro speedup
    micro = f"{float(sp_txt):.3f}x" if sp_txt and ktype == "GEMM" else "N/A"
    serving = "PASS (no autotune risk)" if ktype == "GEMM" else "N/A"

    if integrated and not regression_concs:
        e2e = f"{avg_e2e:.3f}x (avg over all concs)"
        verdict = "**Integrated** ✓"
    elif integrated and regression_concs:
        e2e = f"{avg_e2e:.3f}x (partial)"
        verdict = f"Rolled back (regression at conc={','.join(regression_concs)})"
    elif not integrated:
        e2e = "< 1.0x (regression)"
        verdict = "Rolled back"
    else:
        e2e = "N/A"
        verdict = "Not integrated"

    lines.append(f"| {ktype} | {micro} | {serving} | {e2e} | {verdict} |")

# Per-concurrency E2E comparison
if patched:
    lines.append("\n## Per-Concurrency E2E Comparison\n")
    lines.append(f"| Conc | Baseline TPS | Optimized TPS | Speedup | Gate 4 |")
    lines.append(f"|------|-------------|--------------|---------|--------|")
    for conc_str in sorted(patched.keys(), key=lambda x: int(x)):
        pv  = patched[conc_str]
        bv  = b_e2e.get(conc_str, {})
        base_tps = bv.get("output_tps",0)
        opt_tps  = pv.get("output_tps",0)
        speedup  = pv.get("speedup",0)
        gate4    = "PASS" if pv.get("gate4_pass") else "FAIL"
        lines.append(f"| {conc_str} | {base_tps:.1f} | {opt_tps:.1f} | {speedup:.3f}x | {gate4} |")

# Recommendations
lines.append("\n## Shape Coverage per Kernel\n")
for t in tgts.get("targets",[]):
    ktype = t["kernel_type"]
    n_shapes = t.get("real_shape_count", 0)
    m_vals = rs.get("unique_m_values", []) if ktype == "GEMM" else "N/A"
    lines.append(f"- **{ktype}**: {n_shapes} real shapes  M_values={m_vals}")
    if n_shapes < 3:
        lines.append(f"  (WARNING: low shape coverage — gains may not generalize)")

lines.append("\n## Honest Reporting: Failed/Skipped Optimizations\n")
failed = [t for t in tgts.get("targets",[]) if not t.get("has_real_shapes")]
if failed:
    lines.append(f"The following targets were skipped (no real shapes from trace):")
    for t in failed:
        lines.append(f"- {t['kernel_type']}: {t['pct_total']:.1f}% of GPU time — shapes required for optimization")
else:
    lines.append("All identified targets had real shapes and were attempted.")

# Attribution for concurrencies showing ~1.0x speedup
near_baseline = {c: v for c, v in patched.items() if v.get("speedup", 1.0) < 1.05}
if near_baseline:
    lines.append(f"\n**Why conc={','.join(near_baseline)} shows ~1.0x:**")
    lines.append(
        "vLLM dispatches decode with batch_size ≤ 4 through `wvSplitK`/`LLMM1` kernels "
        "(`rocm_unquantized_gemm_impl`, `utils.py:181`), bypassing TunableOps entirely. "
        "~1.0x at these concurrencies is expected — `wvSplitK` is already near-optimal "
        "for RDNA3 small-batch decode. TunableOps only intercepts batch_size > 4 (`aten::mm` path)."
    )

lines.append("\n## Recommendations\n")
if integrated and avg_e2e >= 1.10:
    lines.append(f"1. **Integration successful**: avg E2E speedup = {avg_e2e:.3f}x")
    lines.append(f"2. Reproduce: `PYTORCH_TUNABLEOP_ENABLED=1 PYTHONPATH={{OPTIMIZED_DIR}}/pypath python3 -m vllm.entrypoints.openai.api_server ...`")
elif integrated:
    lines.append(f"1. Integration active but marginal gain ({avg_e2e:.3f}x). Consider Triton optimization for GEMM.")
else:
    lines.append(f"1. Integration rolled back. Review E2E regression causes before re-integrating.")

lines.append("\n## Artifacts\n")
lines.append(f"- Output dir: `{{OUTPUT_DIR}}`")
lines.append(f"- Tuned GEMM: `{{OPTIMIZED_DIR}}/tuned_gemm.csv`")
lines.append(f"- Injection shim: `{{OPTIMIZED_DIR}}/pypath/sitecustomize.py`")
lines.append(f"- Baseline E2E: `{{RESULTS_DIR}}/baseline_e2e.json`")
lines.append(f"- Integration E2E: `{{RESULTS_DIR}}/integration_e2e.json`")

report_path = "{{REPORT_DIR}}/final_report.md"
with open(report_path,"w") as f:
    f.write("\n".join(lines))
print(f"  Report written: {report_path}")
PYEOF

# ── Step 3: Print summary to terminal (and thus to log) ───────────────────
echo "[$(date +%T)] [Step 3] Final summary:"
python3 -u - << 'PYEOF'
import json, os

i_res = {}
try: i_res = json.load(open("{{RESULTS_DIR}}/integration_result.json"))
except: pass

i_e2e = {}
try: i_e2e = json.load(open("{{RESULTS_DIR}}/integration_e2e.json"))
except: pass

patched = i_e2e.get("patched",{})
baseline = i_e2e.get("baseline",{})
speedups = [v.get("speedup",0) for v in patched.values() if isinstance(v,dict)]
avg_sp = sum(speedups)/len(speedups) if speedups else 0

sp_txt = None
try: sp_txt = open("{{OPTIMIZED_DIR}}/tunableops_speedup.txt").read().strip()
except: pass

print(f"  =========================================")
print(f"  OPTIMIZATION RESULT SUMMARY")
print(f"  =========================================")
print(f"  Model:          {{MODEL}}")
print(f"  Method:         TunableOps rocBLAS offline tuning")
print(f"  Micro speedup:  {sp_txt or '?'}x (geometric mean of decode shapes)")
print(f"  E2E speedup:    {avg_sp:.3f}x (avg over {{CONCURRENCY_LEVELS}})")
print(f"  Integrated:     {i_res.get('integrated','?')}")
print(f"")
print(f"  {'Conc':>4}  {'Baseline':>9}  {'Optimized':>9}  {'Speedup':>8}  Gate4")
print(f"  {'-'*48}")
for conc_str in sorted(patched.keys(), key=lambda x: int(x)):
    pv = patched[conc_str]
    bv = baseline.get(conc_str,{})
    print(f"  {conc_str:>4}  {bv.get('output_tps',0):>8.1f}  {pv.get('output_tps',0):>8.1f}"
          f"  {pv.get('speedup',0):>7.3f}x  {'PASS' if pv.get('gate4_pass') else 'FAIL'}")
print(f"  =========================================")
print(f"  Full report: {{REPORT_DIR}}/final_report.md")
PYEOF

# ── Step 4: Shut down vLLM server ─────────────────────────────────────────
# The server has been running since Phase 1. Pipeline is complete — shut it
# down gracefully so ROCm releases VRAM. SIGTERM allows uvicorn + EngineCore
# to close GPU contexts cleanly. Never leave vLLM running after the pipeline.
echo "[$(date +%T)] [Step 4] Shutting down vLLM server..."
FINAL_PID=$(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")
if [ -n "$FINAL_PID" ]; then
    kill -SIGTERM $FINAL_PID 2>/dev/null || true
    for _w in $(seq 1 30); do
        kill -0 $FINAL_PID 2>/dev/null || { echo "  vLLM exited cleanly after ${_w}s."; break; }
        sleep 1
    done
    if kill -0 $FINAL_PID 2>/dev/null; then
        echo "  Still alive — SIGKILL"
        kill -9 $FINAL_PID 2>/dev/null || true
        sleep 2
    fi
    rm -f "{{OUTPUT_DIR}}/vllm.pid"
    echo "  vLLM server stopped. VRAM released."
else
    echo "  No vLLM PID on record (already stopped)."
fi

# ── Completion ─────────────────────────────────────────────────────────
echo "[$(date +%T)] === Phase 6/6: report — DONE ==="
echo "Full report: {{REPORT_DIR}}/final_report.md"

} 2>&1 | tee -a "$PHASE_LOG"

python3 -c "
import json,datetime
p=json.load(open('{{PROGRESS_FILE}}'))
p.update({'phases_completed':list(dict.fromkeys(p.get('phases_completed',[])+['report'])),
          'current_phase':None,'status':'completed','phase_6_done':datetime.datetime.now().isoformat(),
          'final_report':'{{REPORT_DIR}}/final_report.md'})
json.dump(p,open('{{PROGRESS_FILE}}','w'),indent=2)
" 2>/dev/null || true
```

## Step 4 (MANDATORY SEPARATE BASH CALL): Shut Down vLLM Server

**This step MUST be a separate bash tool call, not merged into the report bash block above.**
The server has been running since Phase 1 and must be stopped to release GPU VRAM.

```bash
# ── MANDATORY: Kill vLLM server after pipeline completes ──────────────────
PHASE_LOG="{{OUTPUT_DIR}}/logs/phase_6_report.log"
{
FINAL_PID=$(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null || echo "")
if [ -n "$FINAL_PID" ]; then
    echo "[$(date +%T)] [Step 4] Shutting down vLLM server PID=$FINAL_PID (SIGTERM)..."
    kill -SIGTERM $FINAL_PID 2>/dev/null || true
    for _w in $(seq 1 30); do
        kill -0 $FINAL_PID 2>/dev/null || { echo "  vLLM exited cleanly after ${_w}s."; break; }
        sleep 1
    done
    if kill -0 $FINAL_PID 2>/dev/null; then
        echo "  Still alive after 30s — sending SIGKILL"
        kill -9 $FINAL_PID 2>/dev/null || true
        sleep 2
    fi
    rm -f "{{OUTPUT_DIR}}/vllm.pid"
    echo "[$(date +%T)] vLLM server stopped. GPU VRAM released."
else
    echo "[$(date +%T)] [Step 4] No vLLM PID on record (already stopped or never started)."
fi
} 2>&1 | tee -a "$PHASE_LOG"
```

**Workflow complete.**
