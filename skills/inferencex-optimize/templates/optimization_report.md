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

Communication time is excluded from optimization targeting (non-optimizable).

| Kernel | Raw % (Total) | Adjusted % (Non-Comm) | Kernel Type | GEAK Mode |
|--------|---------------|----------------------|-------------|-----------|
| ... | ... | ... | ... | ... |

## Optimization Details

### Fusion Opportunities
| Pattern | Components | Combined % | Status | Speedup |
|---------|-----------|-----------|--------|---------|
| ... | ... | ... | Applied / Skipped | X.Xx |

### Per-Kernel Results
| Problem | Type | Opt.Time% | GEAK Mode | Ref (ms) | Opt (ms) | Speedup | Status |
|---------|------|-----------|-----------|----------|----------|---------|--------|
| ... | ... | ... | ... | ... | ... | ... | Applied / Failed / Skipped |

### GEAK Optimization Details
| Kernel | GEAK Mode | GEAK Config | Patch Recovered | Attempts | Final Speedup |
|--------|-----------|-------------|-----------------|----------|---------------|
| ... | simple/kernel-url/manual | geak.yaml/mini_kernel.yaml | yes/no/N/A | N | X.Xx |

## End-to-End Results (ACTUAL MEASURED)

All values below are from real InferenceX benchmark runs, NOT estimates.

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Total Token Throughput (tok/s) | ... | ... | +X.X% |
| Mean TTFT (ms) | ... | ... | ... |
| Mean ITL (ms) | ... | ... | ... |

### E2E vs Kernel-Level Gap

If individual kernel speedups are significant but E2E speedup is small, this indicates
torch.compile/CUDAGraphs are masking kernel-level improvements at the graph level.

## Roofline Analysis

| Op | MxNxK | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline | Optimized? |
|----|-------|------------|----------|------------|-------------|------------|
| ... | ... | ... | ... | memory/compute | ...% | Yes/No |

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
