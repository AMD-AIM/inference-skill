# Inference Optimization Report

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

| Kernel | Raw % (Total) | Adjusted % (Non-Comm) | Library | Source Form | Bucket | Strategy |
|--------|---------------|----------------------|---------|-------------|--------|----------|
| ... | ... | ... | aiter/vllm/... | triton/hip_cpp/ck_template/... | A/B/C | in_place_optimize / dispatch_redirect_to_triton / dispatch_redirect_to_open_lib / in_place_optimize_no_harness / unfeasible_record_only |

## Optimization Details

### Per-Kernel Results
| Kernel | Library | Source Form | Strategy | Lib-Bench Speedup | Library Tests | Allocator Test | Dispatch Verified | Status |
|--------|---------|-------------|----------|-------------------|---------------|----------------|-------------------|--------|
| ... | ... | ... | ... | X.Xx | P/F | pass/fail/N-A | yes/no | Applied / Failed / Skipped |

### Dispatch Redirect Commits
| Source Symbol | Source Lib | Target Symbol | Target Lib | Dispatch Site File | Honored Post-Rebuild |
|---------------|------------|---------------|------------|--------------------|----------------------|
| ... | ... | ... | ... | ... | yes / no |

### ⚠ Bucket B Winners (`optimization_unverified_per_kernel = true`)

These kernels had no built-in per-kernel test harness. Their reported
speedup was measured by the no-harness fallback (single-iteration vLLM
decode against a stored bf16 reference); only the Phase 8 e2e benchmark
provides production attribution.

| Kernel | Library | Source Form | Reason for No Harness | Inner-Loop latency_ms (Baseline → Optimized) | Notes |
|--------|---------|-------------|-----------------------|----------------------------------------------|-------|
| ... | ... | ck_template / tensile_asm / aten_native / ... | no_test_harness / rebuild_too_expensive | X.X → Y.Y | ... |

### Forks and Pinned Commits
| Library | Repo | Pinned Commit | Patched Commit (HEAD of geak/main) | Rebuild Status |
|---------|------|---------------|-------------------------------------|----------------|
| ... | ... | <sha> | <sha> | ok / failed |

## End-to-End Results (ACTUAL MEASURED)

All values below are from real Inference benchmark runs, NOT estimates.

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Total Token Throughput (tok/s) | ... | ... | +X.X% |
| Mean TTFT (ms) | ... | ... | ... |
| Mean ITL (ms) | ... | ... | ... |

### E2E vs Library-Bench Gap

If per-kernel library-bench speedups are significant but E2E speedup is
small, this typically indicates torch.compile / HIPGraphs / dispatch
overhead masking the win at the graph level. Cross-check
`results/dispatch_verification.json`: when `dispatch_verified == true`
and the win still does not propagate, the gap is real and lives in the
graph layer, not in the patch.

### Dispatch Verification (rocprofv3)

| Field | Value |
|-------|-------|
| dispatch_verified | true / false |
| expected_symbol_total_count | N |
| vendor_symbol_leaked_count | N |
| redirect_required_count | N |
| redirect_honored_count | N |

## Roofline Analysis

| Op | MxNxK | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline | Optimized? |
|----|-------|------------|----------|------------|-------------|------------|
| ... | ... | ... | ... | memory/compute | ...% | Yes/No |

## Files Generated
- Benchmark report: {{REPORT_DIR}}/benchmark_report.md
- Profiling report: {{REPORT_DIR}}/profiling_report.md
- Optimization manifest: {{PROBLEMS_DIR}}/optimization_manifest.json
- Redirect plan (when applicable): {{PROBLEMS_DIR}}/redirect_plan.json
- Kernel-source resolution audit: {{PROBLEMS_DIR}}/kernel_source_map_resolved.json
- Per-kernel GEAK results: {{PROBLEMS_DIR}}/geak_results.json
- Forks (in-place patched libraries): {{OUTPUT_DIR}}/forks/<lib>/ on `geak/main`
- Forks manifest: {{OUTPUT_DIR}}/forks/manifest.json
- Per-library rebuild logs: {{RESULTS_DIR}}/rebuild_<lib>.log
- Baseline dispatch trace (rocprofv3): {{RESULTS_DIR}}/baseline_dispatch_trace.json
- Pre-flight dispatch trace (Phase 7): {{RESULTS_DIR}}/preflight_dispatch_trace.json
- Post-rebuild dispatch verification (Phase 8): {{RESULTS_DIR}}/dispatch_verification.json
- Integration manifest: {{RESULTS_DIR}}/integration_manifest.json
- E2E comparison: {{RESULTS_DIR}}/optimization_comparison.json
- This report: {{REPORT_DIR}}/optimization_report.md

## Pipeline Status
- **Completion**: <completed | completed with warnings | completed with blockers | pipeline incomplete>
  - `completed`: all phases passed, integration gate = pass, no blockers
  - `completed with warnings`: integration gate = warn (0.97 ≤ speedup < 1.0), no blockers
  - `completed with blockers`: integration gate = fail OR late-phase blockers present (no `pipeline_blockers.json` required when the fail gate alone drives the status)
  - `pipeline incomplete`: early-phase blockers (`benchmark`, `profile-analyze`) OR integration expected but comparison missing
- **Phases completed**: N of M
- **Terminal blockers**: N

## Blockers
<!-- populated from results/pipeline_blockers.json when present; omitted when empty -->

| Phase | Target | Blocker | Classification | Terminal Action |
|-------|--------|---------|----------------|----------------|
| ... | ... | ... | ... | ... |

## Deferred Work
<!-- only populated when the pipeline has no blockers; warn-band caveats should still be called out above -->
1. <data-driven recommendation based on remaining optimization headroom>
2. <recommendation grounded in profiling or gap analysis findings>
3. <production deployment consideration if applicable>
