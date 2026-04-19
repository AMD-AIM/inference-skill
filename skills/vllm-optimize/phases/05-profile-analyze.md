# Phase 5: Profile Analysis {{SKIP_LABEL}}

## Objective
Analyze profiling traces to identify GPU kernel-level performance bottlenecks and optimization opportunities.

{{PROFILE_ANALYSIS_NOTE}}

## IMPORTANT: Always Re-run Analysis From Scratch
When this phase is entered, **always re-run the full analysis pipeline from step 1**, even if previous analysis artifacts already exist.

```bash
echo "Cleaning previous profile analysis artifacts..."
rm -rf "{{OUTPUT_DIR}}/results/gap_analysis"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_collective_csvs"
rm -rf "{{OUTPUT_DIR}}/results/phase_split"
rm -f  "{{OUTPUT_DIR}}/results/profile_analysis.json"
rm -f  "{{OUTPUT_DIR}}/results/gpu_arch.json"
echo "Cleanup done — starting fresh analysis"
```

## Steps

### 1. Discover and Validate Trace Files
Locate torch profiler trace files in `{{PROFILE_DIR}}/`. Filter out logs and result JSONs. Validate that each file contains `traceEvents` (Chrome Trace Event format).

```bash
python3 -c "
import gzip, glob, os

trace_dir = '{{PROFILE_DIR}}'
valid_traces = []
PEEK_BYTES = 65536

for f in sorted(glob.glob(os.path.join(trace_dir, '*.json*'))):
    basename = os.path.basename(f)
    if '_docker.log' in basename or 'async_llm' in basename.lower():
        continue
    try:
        opener = gzip.open if f.endswith('.gz') else open
        with opener(f, 'rt') as fh:
            prefix = fh.read(PEEK_BYTES)
        if '\"traceEvents\"' not in prefix:
            print(f'SKIPPED (no traceEvents): {basename}')
            continue
        valid_traces.append(f)
        print(f'VALID: {basename}')
    except Exception as e:
        print(f'SKIPPED (error): {basename}: {e}')

if not valid_traces:
    print('WARNING: No valid trace files found')
else:
    print(f'Found {len(valid_traces)} valid trace file(s)')
"
```

### 2. Run Gap Analysis
Use `trace_analyzer.py` to extract GPU kernel statistics and produce `gap_analysis.json`:

```bash
python3 {{SCRIPTS_DIR}}/trace_analyzer.py \
    --trace-dir "{{PROFILE_DIR}}" \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" \
    --top-n 50
```

If `trace_analyzer.py` fails or is not available, run the analysis manually:

```python
import gzip, json, glob, os
from collections import defaultdict

PROFILE_DIR = "{{PROFILE_DIR}}"
OUTPUT_DIR = "{{OUTPUT_DIR}}/results/gap_analysis"

KERNEL_PATTERNS = [
    'wvSplitK', 'ck_tile', 'ck::kernel', 'Cijk_', '__amd_', 'rocclr',
    'gemm', 'mm', 'matmul', 'flash_attn', 'attention', 'paged_attention',
    'allreduce', 'allgather', 'broadcast', 'reduce_scatter',
    'softmax', 'layernorm', 'rmsnorm', 'norm',
    'moe', 'experts', 'topk', 'routing', 'sorting',
    'triton_', 'triton_red', 'triton_fused',
    'vectorized_elementwise', 'elementwise_kernel',
    'copyBuffer', 'memcpy', 'memset',
    'fused_recurrent', 'causal_conv', 'mamba',
    'gdn_', 'act_and_mul', 'silu', 'swiglu'
]

EXCLUDE_PATTERNS = [
    'execute_context', 'profiler', 'frontend', 'python', 'Module:',
    'vllm/model_executor', 'vllm::moe', 'vllm::gdn',
    'triton/runtime/jit', 'pybind11', 'builtin method',
    'nn.Module', 'forward', 'layer.py', 'runner'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_kernels = defaultdict(lambda: {"total_us": 0, "count": 0})

for trace_file in glob.glob(os.path.join(PROFILE_DIR, '*.json*')):
    basename = os.path.basename(trace_file)
    if 'async_llm' in basename.lower() or '_docker.log' in basename:
        continue
    try:
        opener = gzip.open if trace_file.endswith('.gz') else open
        with opener(trace_file, 'rt') as f:
            data = json.load(f)
    except Exception:
        continue

    for e in data.get('traceEvents', []):
        if 'dur' not in e or e.get('dur', 0) <= 0:
            continue
        name = e.get('name', '')
        cat = e.get('cat', '').lower()
        if cat not in ['kernel', 'cuda', 'gpu']:
            continue
        if any(p.lower() in name.lower() for p in EXCLUDE_PATTERNS):
            continue
        if any(p.lower() in name.lower() for p in KERNEL_PATTERNS):
            all_kernels[name]['total_us'] += e['dur']
            all_kernels[name]['count'] += 1

sorted_kernels = sorted(all_kernels.items(), key=lambda x: x[1]['total_us'], reverse=True)
total_time = sum(s['total_us'] for _, s in sorted_kernels)

gap_analysis = {
    "top_kernels": [
        {"name": name, "total_us": stats['total_us'], "avg_us": stats['total_us'] / max(stats['count'], 1),
         "calls": stats['count'], "pct_total": stats['total_us'] / total_time * 100 if total_time > 0 else 0}
        for name, stats in sorted_kernels[:50]
    ],
    "total_kernel_time_us": total_time,
    "unique_kernels": len(all_kernels)
}

with open(os.path.join(OUTPUT_DIR, 'gap_analysis.json'), 'w') as f:
    json.dump(gap_analysis, f, indent=2)

# Print summary
for i, (name, stats) in enumerate(sorted_kernels[:25]):
    pct = stats['total_us'] / total_time * 100 if total_time > 0 else 0
    print(f"{i+1}. {name[:55]} | {stats['count']} calls | {stats['total_us']/1000:.2f}ms | {pct:.2f}%")
```

### 3. Categorize Kernels
Use `classify_kernel.py` to categorize kernels into functional groups:

```bash
python3 {{SCRIPTS_DIR}}/classify_kernel.py \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --output "{{OUTPUT_DIR}}/results/gap_analysis/category_breakdown.json"
```

### 4. Run Fusion Analysis
Use `analyze_fusion_vllm.py` to detect fusable operator patterns:

```bash
python3 {{SCRIPTS_DIR}}/analyze_fusion_vllm.py \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --framework vllm \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json"
```

### 5. Extract Real GEMM Shapes from Trace

**CRITICAL for kernel optimization**: Extract the actual (M, K, N) shapes dispatched during inference. These are the shapes that Phase 7's kernel agent must optimize for — NOT shapes guessed from model config.

```bash
python3 {{SCRIPTS_DIR}}/extract_trace_shapes.py \
    --trace-dir "{{PROFILE_DIR}}" \
    --output "{{OUTPUT_DIR}}/results/real_shapes.json"
```

If the script is not available, copy it from the skill:
```bash
cp {{SKILL_DIR}}/scripts/extract_trace_shapes.py {{SCRIPTS_DIR}}/ 2>/dev/null || true
```

The output `real_shapes.json` contains:
- Every unique (M, K, N) actually observed, with call counts
- `unique_m_values`: the real batch sizes (e.g., [1, 129] not [1, 64, 256])
- `benchmark_shapes`: ready-to-use shape pairs for kernel agent benchmarking

### 6. Generate GPU Architecture Info
```bash
python3 -c "
import json, os, subprocess, re

gpu_arch = 'unknown'
gpu_vendor = 'unknown'
gpu_count = 0

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
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_vendor = 'nvidia'
            cap = result.stdout.strip().split('\n')[0].replace('.', '')
            gpu_arch = f'sm_{cap}'
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

gpu_info = {'gpu_vendor': gpu_vendor, 'gpu_arch': gpu_arch, 'gpu_count': gpu_count}
with open('{{OUTPUT_DIR}}/results/gpu_arch.json', 'w') as f:
    json.dump(gpu_info, f, indent=2)
print(json.dumps(gpu_info, indent=2))
"
```

### 6. Generate Profile Analysis Report
Create `{{REPORT_DIR}}/profiling_report.md` with:
- Configuration summary
- GPU kernel breakdown table
- Category breakdown
- Key findings

## Completion
Update progress.json:
```json
{
  "phase": "profile-analyze",
  "phases_completed": ["env", "vllm-setup", "benchmark", "benchmark-analyze", "profiling", "profile-analyze"],
  "current_step": "profile analysis complete",
  "details": {
    "trace_files_analyzed": "<N>",
    "unique_kernels": "<M>",
    "top_bottleneck": "<kernel_name>"
  }
}
```