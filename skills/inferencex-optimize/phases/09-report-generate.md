# Phase 9: Final Optimization Report {{SKIP_LABEL}}

## Objective
Generate a comprehensive report combining benchmark, profiling, and optimization results with ACTUAL MEASURED end-to-end speedup.

## Steps

### 1. Gather All Results

Read:
- `{{RESULTS_DIR}}/benchmark_summary.json` (Phase 3)
- `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` (Phase 5)
- `{{OUTPUT_DIR}}/results/profile_analysis.json` (Phase 5, if exists)
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (Phase 6, enriched with profiling data)
- `{{PROBLEMS_DIR}}/kernel_type_classification.json` (Phase 6, kernel types + adjusted profiling %)
- `{{PROBLEMS_DIR}}/fusion_opportunities.json` (Phase 6)
- `{{PROBLEMS_DIR}}/bottleneck_kernels.json` (Phase 6)
- `{{PROBLEMS_DIR}}/geak_results.json` (Phase 7)
- `{{RESULTS_DIR}}/optimization_comparison.json` (Phase 8)
- Individual `*_best.json` tracker files from `{{PROBLEMS_DIR}}`
- `{{REPORT_DIR}}/benchmark_report.md` (Phase 3, if exists)
- `{{REPORT_DIR}}/profiling_report.md` (Phase 5, if exists)
- `{{OUTPUT_DIR}}/results/gpu_arch.json` (if exists)
- `{{ENV_INFO_FILE}}` (Phase 0, GEAK availability + GPU arch)

### 2. Generate Final Report

Create `{{REPORT_DIR}}/optimization_report.md` using this template:

```markdown
# InferenceX Optimization Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU from gpu_arch.json>
- **Framework**: {{FRAMEWORK}}
- **Model**: <model name from config>
- **Precision**: <precision>
- **Tensor Parallelism**: <TP value>
- **Sequence Length**: ISL=<ISL>, OSL=<OSL>
- **Concurrency**: <concurrency used>

## Executive Summary
- **ACTUAL Measured End-to-End Speedup**: X.Xx
- **Kernels Optimized**: N of M candidates (speedup > 1.0x)
- **Baseline Throughput**: X tok/s
- **Optimized Throughput**: X tok/s
- **TTFT Change**: X.Xms -> X.Xms
- **ITL Change**: X.Xms -> X.Xms

## Profile Bottlenecks

Top GPU kernels from gap analysis (steady-state profiling):

| Rank | Kernel Name | Calls | Total Time (us) | % Total | Category |
|------|-------------|-------|-----------------|---------|----------|
| 1 | ... | ... | ... | ... | ... |

### Category Breakdown
| Category | Time (us) | % of Total |
|----------|----------|------------|
| ... | ... | ... |

### Optimization Impact (Adjusted Percentages)

Communication time (e.g., cross_device_reduce) is excluded from optimization targeting because profiling uses eager mode which inflates allreduce overhead, and communication kernels are non-optimizable.

| Kernel | Raw % (Total) | Adjusted % (Non-Comm) | Kernel Type | GEAK Mode |
|--------|---------------|----------------------|-------------|-----------|
| ... | ... | ... | ... | ... |

## Optimization Details

### Fusion Opportunities
| Pattern | Components | Combined % | Status | Speedup |
|---------|-----------|-----------|--------|---------|
| fused_residual_rmsnorm | add + RMSNorm | X.X% | Applied / Skipped | X.Xx |
| fused_swiglu | silu + mul | X.X% | Applied / Skipped | X.Xx |

### Per-Kernel Results
| Problem | Type | Opt.Time% | GEAK Mode | Ref (ms) | Opt (ms) | Speedup | Status |
|---------|------|-----------|-----------|----------|----------|---------|--------|
| ... | ... | ... | ... | ... | ... | ... | Applied / Failed / Skipped |

### GEAK Optimization Details

| Kernel | GEAK Mode | GEAK Config | Patch Recovered | Attempts | Final Speedup |
|--------|-----------|-------------|-----------------|----------|---------------|
| ... | simple/kernel-url/manual | geak.yaml/mini_kernel.yaml | yes/no/N/A | N | X.Xx |

## End-to-End Results (ACTUAL MEASURED)

⚠️ All values below are from real InferenceX benchmark runs, NOT estimates.

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Total Token Throughput (tok/s) | ... | ... | +X.X% |
| Mean TTFT (ms) | ... | ... | ... |
| Mean ITL (ms) | ... | ... | ... |

### E2E vs Kernel-Level Gap

If individual kernel speedups are significant but E2E speedup is small, this indicates
that torch.compile/CUDAGraphs are masking kernel-level improvements at the graph level.
Consider benchmarking in eager mode (--enforce-eager / --disable-cuda-graph) to isolate
the kernel-level impact.

## Roofline Analysis

If TraceLens roofline data is available from Phase 5 (`profile_analysis.json` gemm_shapes):

| Op | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline | Optimized? |
|----|-------|------------|----------|------------|-------------|------------|
| ... | ... | ... | ... | memory/compute | ...% | Yes/No |

## Comparison Outputs (Seed=42)

If both baseline and optimized runs used deterministic seeding, include side-by-side verification
of text output quality to confirm optimization did not affect model accuracy.

## Files Generated
- Benchmark report: {{REPORT_DIR}}/benchmark_report.md
- Profiling report: {{REPORT_DIR}}/profiling_report.md
- Problem files: {{PROBLEMS_DIR}}/
- Optimized kernels: {{OPTIMIZED_DIR}}/
- Plugin: {{OPTIMIZED_DIR}}/vllm_plugin/ or sglang_plugin/
- Optimization manifest: {{PROBLEMS_DIR}}/optimization_manifest.json
- Kernel results: {{PROBLEMS_DIR}}/geak_results.json
- E2E comparison: {{RESULTS_DIR}}/optimization_comparison.json
- This report: {{REPORT_DIR}}/optimization_report.md

## Recommendations
1. <data-driven recommendation based on bottleneck analysis>
2. <recommendation about what to optimize next>
3. <recommendation about production deployment considerations>
```

### 3. Generate Machine-Readable Summary

Save `{{REPORT_DIR}}/optimization_summary.json`:

```bash
python3 -c "
import json, os, glob

summary = {
    'config_key': '{{CONFIG_KEY}}',
    'framework': '{{FRAMEWORK}}',
    'phases_completed': True,
}

# Load env info
env_path = '{{ENV_INFO_FILE}}'
if os.path.isfile(env_path):
    env = json.load(open(env_path))
    summary['gpu_arch'] = env.get('gpu_arch', 'unknown')
    summary['geak_mode'] = 'auto' if env.get('geak_available') else 'manual'

# Load comparison data
comp_path = '{{RESULTS_DIR}}/optimization_comparison.json'
if os.path.isfile(comp_path):
    comp = json.load(open(comp_path))
    summary['baseline_throughput'] = comp.get('baseline', {}).get('total_token_throughput', 0)
    summary['optimized_throughput'] = comp.get('optimized', {}).get('total_token_throughput', 0)
    summary['speedup'] = comp.get('speedup', 1.0)
    summary['validated'] = comp.get('validated', False)

# Load kernel results
results_path = '{{PROBLEMS_DIR}}/geak_results.json'
if os.path.isfile(results_path):
    results = json.load(open(results_path))
    summary['kernels_attempted'] = len(results)
    summary['kernels_improved'] = sum(1 for r in results if r.get('speedup', 0) > 1.0)
    summary['patches_recovered'] = sum(1 for r in results if r.get('patch_recovered', False))
    summary['kernel_results'] = results

# Load manifest
manifest_path = '{{PROBLEMS_DIR}}/optimization_manifest.json'
if os.path.isfile(manifest_path):
    manifest = json.load(open(manifest_path))
    summary['total_problem_files'] = len(manifest.get('optimizations', []))

with open('{{REPORT_DIR}}/optimization_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
"
```

### 4. Print Report Path

```bash
echo ""
echo "============================================"
echo "  Optimization Report Generated"
echo "============================================"
echo "Report: {{REPORT_DIR}}/optimization_report.md"
echo "Summary: {{REPORT_DIR}}/optimization_summary.json"
echo "============================================"
```

## Completion

Update progress.json:
```json
{
  "phase": "report-generate",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze", "problem-generate", "kernel-optimize", "integration", "report-generate"],
  "current_step": "optimization workflow complete",
  "details": {
    "report": "{{REPORT_DIR}}/optimization_report.md",
    "summary": "{{REPORT_DIR}}/optimization_summary.json"
  }
}
```
