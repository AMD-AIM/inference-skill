# Phase 4: Analysis

## Objective
Analyze profiler traces and generate GPU kernel performance report with proper filtering.

## Steps

### 1. Discover Trace Files

```python
import glob
import gzip
import os

PROFILE_DIR = os.environ.get('PROFILE_DIR', './vllm_results/profiles')

valid_traces = []
for f in sorted(glob.glob(f"{PROFILE_DIR}/*.json*")):
    if 'rocprof' in f.lower():
        continue
    try:
        with gzip.open(f, 'rt') as fh:
            content = fh.read(65536)
        if '"traceEvents"' in content:
            valid_traces.append(f)
            print(f"VALID: {os.path.basename(f)}")
    except:
        pass

TRACE_FILE = valid_traces[0] if valid_traces else None
```

### 2. Gap Analysis

**CRITICAL**: Filter out Python profiler annotations, only show actual GPU hardware kernels.

```python
import gzip
import json
from collections import defaultdict

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

with gzip.open(TRACE_FILE, 'rt') as f:
    data = json.load(f)

kernel_times = defaultdict(lambda: {"total_us": 0, "count": 0})

for e in data.get('traceEvents', []):
    if 'dur' not in e or e.get('dur', 0) <= 0:
        continue
    
    name = e.get('name', '')
    cat = e.get('cat', '').lower()
    
    # Must be kernel/cuda/gpu category
    if cat not in ['kernel', 'cuda', 'gpu']:
        continue
    
    # Exclude Python-level names
    if any(p.lower() in name.lower() for p in EXCLUDE_PATTERNS):
        continue
    
    # Must match actual kernel patterns
    if any(p.lower() in name.lower() for p in KERNEL_PATTERNS):
        kernel_times[name]['total_us'] += e['dur']
        kernel_times[name]['count'] += 1

sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1]['total_us'], reverse=True)
total_time = sum(s['total_us'] for _, s in sorted_kernels)

print(f"\nTotal GPU kernel time: {total_time/1000:.2f} ms")
print(f"Unique kernels: {len(sorted_kernels)}")

for i, (name, stats) in enumerate(sorted_kernels[:25]):
    pct = stats['total_us'] / total_time * 100
    print(f"{i+1}. {name[:55]} | {stats['count']} calls | {stats['total_us']/1000:.2f}ms | {pct:.2f}%")
```

### 3. Categorize Kernels

```python
CATEGORIES = {
    'MoE': ['wvSplitK', 'moe', 'ck_tile', 'topk', 'sorting', 'Cijk_'],
    'Attention': ['attention', 'paged_attention', 'flash'],
    'Normalization': ['layernorm', 'rmsnorm', 'norm', 'triton_red'],
    'Memory': ['copyBuffer', 'memcpy', 'memset'],
    'Activation': ['silu', 'swiglu', 'act_and_mul', 'gdn', 'mamba', 'fused_recurrent'],
    'Elementwise': ['vectorized_elementwise', 'elementwise_kernel']
}

category_time = {}
for name, stats in sorted_kernels:
    cat = 'Other'
    for c, patterns in CATEGORIES.items():
        if any(p.lower() in name.lower() for p in patterns):
            cat = c
            break
    category_time[cat] = category_time.get(cat, 0) + stats['total_us']

print("\n=== Kernel Category Breakdown ===")
for cat, time_us in sorted(category_time.items(), key=lambda x: -x[1]):
    print(f"{cat}: {time_us/1000:.2f}ms ({time_us/total_time*100:.1f}%)")
```

### 4. Generate Report

```markdown
# vLLM Profiling Report

## Configuration
- Model: {MODEL}
- GPU: {GPU}
- Sequence: ISL={ISL}, OSL={OSL}

## GPU Kernel Breakdown

| Kernel | Calls | Time(ms) | % |
|--------|-------|----------|---|
| ... | ... | ... | ... |

## Category Summary

| Category | Time % |
|----------|--------|
| MoE | ~X% |
| Attention | ~X% |
| ... | ... |

## Key Findings

1. Bottleneck: {category} at ~X%
2. Optimized: {category} at ~Y%

## Recommendations

- Focus optimization on {bottleneck_category}
```

## Output Files

| File | Description |
|------|-------------|
| gap_analysis/gap_analysis.csv | Kernel breakdown CSV |
| gap_analysis/gap_analysis.json | Kernel stats JSON |
| profiling_report.md | Full markdown report |

## Common Issues

### Trace file not generated
- Ensure `--profiler-config.profiler torch` is set
- Check server log for initialization messages

### Only seeing "execute_context"
- That's Python profiler annotation, NOT actual kernel
- Use EXCLUDE_PATTERNS to filter it out

### Gzip EOF error
- Usually profiler was interrupted
- Increase timeout or check server stability

## Completion

Report generated in results directory.