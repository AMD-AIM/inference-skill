# InferenceX Profiling Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU>
- **Framework**: <framework from config>
- **Model**: <model name>
- **Precision**: <precision>
- **Tensor Parallelism**: <TP value>
- **Sequence Length**: ISL=<ISL>, OSL=<OSL>
- **Concurrency**: <concurrency used for profiling>

## Trace Structure
- **Per-rank traces detected**: <WORLD_SIZE>
- **Merged trace available**: <yes/no>
- **Collective trace basis**: <TP-N-EXTEND / full rank-N / unavailable>
- **Primary single-trace TraceLens input**: <rank-0 EXTEND trace / rank-0 full trace / merged fallback>
- **Phase-split input**: <merged trace / rank-0 full trace / unavailable>

## GPU Utilization

| Metric | Full Trace | Prefill-Decode | Decode-Only |
|--------|------------|----------------|-------------|
| Computation Time (%) | <from gpu_timeline.csv> | <from prefill-decode gpu_timeline.csv> | <from decode-only gpu_timeline.csv> |
| Exposed Comm Time (%) | ... | ... | ... |
| Exposed Memcpy Time (%) | ... | ... | ... |
| GPU Busy Time (%) | ... | ... | ... |
| GPU Idle Time (%) | ... | ... | ... |

## Top GPU Kernels (Steady-State)

From gap analysis of the profiled steady-state iterations:

| Rank | Kernel Name | Calls | Total Time (us) | Avg (us) | % Total |
|------|-------------|-------|-----------------|----------|---------|
| 1 | ... | ... | ... | ... | ... |

## Kernel Category Breakdown

From TraceLens primary single-trace ops_summary_by_category:

| Category | Count | Total Time (ms) | % of Kernel Time |
|----------|-------|-----------------|------------------|
| ... | ... | ... | ... |

## Phase-Split Roofline Analysis

The phase-split input trace (prefer merged/full trace, not already-split phase traces) is split into prefill-decode and decode-only phases using TraceLens-internal's `split_vllm_trace_annotation.py`, then analyzed with `generate_perf_report_pytorch_inference.py` for per-phase roofline insights against <GPU> specs (<mem_bw> GB/s HBM bandwidth, <peak_tflops> TFLOPS bf16 peak).

### Prefill-Decode Phase (<N> steps, BS=<batch_size>)

| Metric | Value |
|--------|-------|
| Computation Time (%) | <from prefill-decode gpu_timeline.csv> |
| GPU Busy Time (%) | <from prefill-decode gpu_timeline.csv> |
| Dominant Bound Type | <compute or memory, from unified_perf_summary.csv> |

**GEMM Roofline (prefill-decode)** — REQUIRED, read from `tracelens_prefill_decode_csvs/GEMM.csv`. List ALL GEMM shapes. For each row, extract `name`, `param: M`, `param: N`, `param: K`, `FLOPS/Byte_first`, `TFLOPS/s_mean`, `Compute Spec`, `Pct Roofline_mean`. Determine bound type: if `FLOPS/Byte < mem_bw / peak_tflops` -> memory-bound, else -> compute-bound.

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline (%) |
|---------|-------|------------|----------|------------|------------------|
| <from GEMM.csv> | <M>×<N>×<K> | <FLOPS/Byte_first> | <TFLOPS/s_mean> | <memory or compute> | <Pct Roofline_mean> |

**Attention Roofline (prefill-decode)** — if `SDPA_fwd.csv` or `FLASH_ATTN_fwd.csv` exists, include it in the same format.

**Top ops by time (prefill-decode)** — from `unified_perf_summary.csv`, list top 10 ops with their roofline metrics:

| Op Name | Category | Time (ms) | % Total | FLOPS/Byte | TFLOPS/s | Pct Roofline (%) |
|---------|----------|-----------|---------|------------|----------|------------------|
| <from unified_perf_summary.csv> | ... | ... | ... | ... | ... | ... |

### Decode-Only Phase (<N> steps, BS=<batch_size>)

| Metric | Value |
|--------|-------|
| Computation Time (%) | <from decode-only gpu_timeline.csv> |
| GPU Busy Time (%) | <from decode-only gpu_timeline.csv> |
| Dominant Bound Type | <compute or memory, from unified_perf_summary.csv> |

**GEMM Roofline (decode-only)** — REQUIRED, read from `tracelens_decode_only_csvs/GEMM.csv`. Same format as prefill-decode.

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline (%) |
|---------|-------|------------|----------|------------|------------------|
| <from GEMM.csv> | <M>×<N>×<K> | <FLOPS/Byte_first> | <TFLOPS/s_mean> | <memory or compute> | <Pct Roofline_mean> |

**Attention Roofline (decode-only)** — if `SDPA_fwd.csv` or `FLASH_ATTN_fwd.csv` exists, include it.

**Top ops by time (decode-only)** — from `unified_perf_summary.csv`, top 10 ops:

| Op Name | Category | Time (ms) | % Total | FLOPS/Byte | TFLOPS/s | Pct Roofline (%) |
|---------|----------|-----------|---------|------------|----------|------------------|
| <from unified_perf_summary.csv> | ... | ... | ... | ... | ... | ... |

### Phase Comparison

| Metric | Prefill-Decode | Decode-Only |
|--------|----------------|-------------|
| GPU Computation (%) | ... | ... |
| GPU Busy (%) | ... | ... |
| FusedMoE Time (%) | ... | ... |
| GEMM Time (%) | ... | ... |
| Attention Time (%) | ... | ... |
| RMSNorm Time (%) | ... | ... |
| Communication Time (%) | ... | ... |
| Dominant Bound | compute / memory | compute / memory |
| Batch Size | ... | ... |

## Profile Bottlenecks & Optimization Opportunities
- <bottleneck 1: description and recommendation>
- <bottleneck 2: description and recommendation>
- ...

## Raw Profile Data
- Gap analysis: `results/gap_analysis/`
- TraceLens primary single-trace CSVs: `results/tracelens_rank0_csvs/` (directory name kept for backward compatibility)
- Collective analysis staged traces: `results/collective_trace_inputs/`
- Phase-split traces: `results/phase_split/`
- Prefill-decode roofline CSVs: `results/tracelens_prefill_decode_csvs/`
- Decode-only roofline CSVs: `results/tracelens_decode_only_csvs/`
- GPU arch config: `results/gpu_arch.json`
- Profile analysis JSON: `results/profile_analysis.json`
- Profiler summary: `profiles/profiler_out_0.txt`
- Primary single-trace input: `profiles/<trace_file_name>`
- Phase-split input trace: `profiles/<phase_split_input_trace_file_name or unavailable>`
- Traces viewable at: https://ui.perfetto.dev/
