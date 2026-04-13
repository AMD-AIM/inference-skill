> **ARCHIVE**: This file is a reference copy of the original phase runbook. The active
> agent docs are in `agents/phase-NN-*.md`. Script paths in this file reference the
> pre-reorganization flat layout (`scripts/*.py`); the actual scripts are now under
> `scripts/{env,container,profiling,optimize,plugin,report}/`.

# Phase 9: Final Optimization Report {{SKIP_LABEL}}

## Objective
Generate a comprehensive report combining benchmark, profiling, and optimization results with ACTUAL MEASURED end-to-end speedup.

## Steps

### 1. Gather All Results

Read these artifacts:
- `{{RESULTS_DIR}}/benchmark_summary.json` (Phase 3)
- `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` (Phase 5)
- `{{OUTPUT_DIR}}/results/profile_analysis.json` (Phase 5)
- `{{PROBLEMS_DIR}}/optimization_manifest.json` (Phase 6)
- `{{PROBLEMS_DIR}}/kernel_type_classification.json` (Phase 6)
- `{{PROBLEMS_DIR}}/fusion_opportunities.json` (Phase 6)
- `{{PROBLEMS_DIR}}/geak_results.json` (Phase 7)
- `{{RESULTS_DIR}}/optimization_comparison.json` (Phase 8)
- `{{REPORT_DIR}}/benchmark_report.md` and `profiling_report.md` (Phases 3, 5)
- `{{OUTPUT_DIR}}/results/gpu_arch.json` and `{{ENV_INFO_FILE}}`

### 2. Generate Final Report

Create `{{REPORT_DIR}}/optimization_report.md` using template from `{{TEMPLATES_DIR}}/optimization_report.md`. Populate all sections from gathered artifacts.

### 3. Generate Machine-Readable Summary

Run:
```bash
python3 "{{SCRIPTS_DIR}}/generate_optimization_summary.py" \
    --output "{{REPORT_DIR}}/optimization_summary.json" \
    --config-key "{{CONFIG_KEY}}" --framework "{{FRAMEWORK}}" \
    --env-info "{{ENV_INFO_FILE}}" \
    --results-dir "{{RESULTS_DIR}}" --problems-dir "{{PROBLEMS_DIR}}"
```

### 4. Print Report Path
```bash
echo "Report: {{REPORT_DIR}}/optimization_report.md"
echo "Summary: {{REPORT_DIR}}/optimization_summary.json"
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
