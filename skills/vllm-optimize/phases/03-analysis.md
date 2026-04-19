# Phase 3: Analysis {{SKIP_LABEL}}

## Objective
Analyze benchmark results AND profiler traces together. Extract:
1. Throughput/latency metrics from benchmark
2. GPU kernel time breakdown from profiler
3. Real GEMM shapes from trace (for kernel optimization)

## Steps

### 1. Benchmark Analysis

Parse benchmark results and compute derived metrics:

```bash
python3 << 'PYEOF'
import json

with open("{{OUTPUT_DIR}}/results/benchmark_report.json") as f:
    data = json.load(f)

tp = {{TP}}
print("Throughput & Latency (TP={})".format(tp))
print("-" * 90)
print(f"{'Conc':>4} | {'ISLxOSL':>9} | {'Total TPS':>9} | {'TPS/GPU':>8} | {'Out TPS':>8} | {'Interactivity':>13} | {'Latency':>8}")
print("-" * 90)

for r in data.get("benchmark_results", []):
    tps_gpu = r["total_tps"] / tp
    # Interactivity = output_tokens_per_second / concurrency
    avg_out = r.get("avg_completion_tokens", 0)
    if avg_out > 0 and r["avg_latency_s"] > 0:
        tpot = r["avg_latency_s"] / avg_out  # time per output token
        interactivity = 1.0 / tpot if tpot > 0 else 0
    else:
        interactivity = r["output_tps"] / max(r["concurrency"], 1)
    print(f"{r['concurrency']:>4} | {data.get('config',{}).get('ISL','?')}x{data.get('config',{}).get('OSL','?'):>4} | {r['total_tps']:>9.1f} | {tps_gpu:>8.1f} | {r['output_tps']:>8.1f} | {interactivity:>10.1f} t/s/u | {r['avg_latency_s']:>7.2f}s")

# Save summary
with open("{{OUTPUT_DIR}}/results/benchmark_summary.json", "w") as f:
    json.dump(data, f, indent=2)
PYEOF
```

### 2. Profile Analysis — GPU Kernel Breakdown

Run gap analysis on profiler traces. Use `trace_analyzer.py` if available, otherwise inline analysis:

```bash
python3 {{SCRIPTS_DIR}}/trace_analyzer.py \
    --trace-dir "{{PROFILE_DIR}}" \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" \
    --top-n 50 2>/dev/null || echo "trace_analyzer.py not available, using inline analysis"
```

If trace_analyzer fails, use the inline fallback from the Phase 5 doc (gap analysis with kernel categorization).

### 3. Categorize Kernels

```bash
python3 {{SCRIPTS_DIR}}/classify_kernel.py \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --output "{{OUTPUT_DIR}}/results/gap_analysis/category_breakdown.json" 2>/dev/null || true
```

### 4. Extract Real GEMM Shapes from Trace

**CRITICAL**: These shapes drive kernel optimization. They must come from the ACTUAL profiled workload.

```bash
python3 {{SCRIPTS_DIR}}/extract_trace_shapes.py \
    --trace-dir "{{PROFILE_DIR}}" \
    --output "{{OUTPUT_DIR}}/results/real_shapes.json"
```

After extraction, verify:
- M values should include BOTH small (decode) AND larger values (batch decode at profile concurrency)
- N values should match the model's actual projection sizes (may differ from intermediate_size due to merged projections)

### 5. Generate GPU Architecture Info

```bash
python3 -c "
import json, subprocess, re
gpu_arch = 'unknown'; gpu_vendor = 'unknown'; gpu_count = 0

# AMD: use rocm-smi for count, rocminfo for arch (pick most specific gfx name)
try:
    r = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        gpu_lines = [l for l in r.stdout.split('\n') if 'GPU[' in l]
        if gpu_lines:
            gpu_vendor = 'amd'
            gpu_count = len(gpu_lines)
except Exception:
    pass

if gpu_vendor == 'amd':
    try:
        r = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            arches = re.findall(r'\bgfx(\d+)\b', r.stdout)
            if arches:
                best = max(arches, key=len)
                gpu_arch = f'gfx{best}'
    except Exception:
        pass

if gpu_vendor == 'unknown':
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            gpu_vendor = 'nvidia'
            gpu_arch = f'sm_{r.stdout.strip().replace(\".\", \"\")}'
            r2 = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if r2.returncode == 0:
                gpu_count = int(r2.stdout.strip().split('\n')[0])
    except Exception:
        pass

if gpu_vendor == 'unknown':
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            prop = torch.cuda.get_device_properties(0)
            gpu_arch = getattr(prop, 'gcnArchName', getattr(prop, 'name', 'unknown')).split(':')[0]
            gpu_vendor = 'amd' if 'gfx' in gpu_arch else 'nvidia'
    except Exception:
        pass

with open('{{OUTPUT_DIR}}/results/gpu_arch.json', 'w') as f:
    json.dump({'gpu_vendor': gpu_vendor, 'gpu_arch': gpu_arch, 'gpu_count': gpu_count}, f, indent=2)
print(f'GPU: {gpu_vendor} {gpu_arch} x{gpu_count}')
"
```

### 6. Generate Analysis Report

Write `{{REPORT_DIR}}/analysis_report.md` combining:
- Benchmark throughput/latency table
- GPU kernel breakdown (from gap analysis)
- Real shapes extracted
- Key findings and bottleneck identification

## Completion
Update progress.json:
```json
{
  "phase": "analysis",
  "phases_completed": ["env", "vllm-setup", "benchmark-and-profile", "analysis"],
  "current_step": "analysis complete"
}
```
