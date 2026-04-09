# Phase 4: Analysis

## Objective
Analyze profiler traces and generate GPU kernel performance report.

## Steps

### 1. Discover and Analyze Traces

Run: `python3 scripts/analyze_traces.py --profile-dir "$PROFILE_DIR" --output-dir "$OUTPUT_DIR/gap_analysis"`

The script:
1. Discovers valid torch profiler traces (filters out rocprof, validates `traceEvents` key)
2. Filters to actual GPU kernels (excludes Python profiler annotations)
3. Aggregates kernel time and counts
4. Categorizes kernels (MoE, Attention, Normalization, Memory, Activation, Elementwise)
5. Outputs `gap_analysis.json` with kernel breakdown and category summary

### 2. Generate Report

Create `$OUTPUT_DIR/profiling_report.md` using template from `templates/analysis_report.md`.

Populate with:
- Top 25 GPU kernels by cumulative time
- Category breakdown (MoE, Attention, etc.)
- Key findings and bottleneck identification
- Optimization recommendations based on category distribution

## Output Files
| File | Description |
|------|-------------|
| gap_analysis/gap_analysis.json | Kernel stats JSON |
| profiling_report.md | Full markdown report |

## Troubleshooting
- **Trace not generated**: Ensure `--profiler-config.profiler torch` set at server start
- **Only "execute_context"**: Python annotation, not GPU kernel — filtered by the analysis script
- **Gzip EOF error**: Profiler interrupted; increase timeout or check server stability

## Completion
Report generated in results directory.
