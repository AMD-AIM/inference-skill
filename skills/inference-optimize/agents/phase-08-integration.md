# Phase 8: Integration (rebuild + rocprofv3 dispatch verify + e2e)

## Instructions

You are a phase agent responsible for installing the patched forks into
the live environment, verifying with rocprofv3 that the expected kernel
symbols actually fire (and the vendor symbols do not), running the
standard vLLM e2e benchmark, and validating the throughput delta. You
read exactly 2 files: this document and your handoff at
`handoff/to-phase-08.md`.

This phase **replaces** the legacy plugin-injection path. There is no
`generate_vllm_plugin.py`, no `inject_plugin.py`, no
`integration_plugin/` directory, no atexit telemetry, no patch counters,
no `dispatch_plugin_example.py`. The library-rebuild approach reinstalls
the patched fork in-place; vLLM's normal import order picks it up.

**Tools**: Shell commands, Docker, Python, file I/O, rocprofv3.
**Outputs**: Write `agent-results/phase-08-result.md`. Write
`{{RESULTS_DIR}}/dispatch_verification.json`,
`{{RESULTS_DIR}}/integration_manifest.json`,
`{{RESULTS_DIR}}/optimization_comparison.json`.

**MANDATORY**: This phase MUST produce real measured data. FORBIDDEN:
estimating speedup, copying baseline numbers, or skipping the benchmark.

**SKIP_INTEGRATION**: If the handoff sets `SKIP_INTEGRATION=true`, this
phase should NOT have been dispatched -- the orchestrator removes it from
the phase list. If you find yourself running with this flag set, run
Steps 1-2 only (rebuild + dispatch verify) and skip the e2e benchmark.
Document the skip in the result doc.

## Runbook

### Config Resolution

Before any Docker or benchmark commands, read
`{{OUTPUT_DIR}}/results/sweep_configs.json` (or `{{OUTPUT_DIR}}/config.json`
when sweep metadata is folded there) and export: `RUNNER`, `IMAGE`,
`FRAMEWORK`, `MODEL`, `PRECISION`, `TP`, `EP`, `CONC`, `ISL`, `OSL`,
`MAX_MODEL_LEN`, `EXP_NAME`, `BENCHMARK_SCRIPT`, plus GPU selectors.

### Measured-data gate

Phase 8 is **not** complete until four artifacts exist:
1. Per-library rebuild logs at `{{RESULTS_DIR}}/rebuild_<lib>.log` with
   non-zero `libraries_rebuilt_ok_count`.
2. `{{RESULTS_DIR}}/dispatch_verification.json` showing
   `dispatch_verified == true`, `vendor_symbol_leaked_count == 0`, and
   `redirect_honored_count == redirect_required_count`.
3. A patched-server benchmark JSON captured after rebuild (no plugin
   wiring required -- the editable install shadows the wheel copy).
4. A passing `validate_optimization.py` run.

### 0. GPU State Cleanup
```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f sglang 2>/dev/null || true
sleep 5
```

### 1. Rebuild Forked Libraries

For each fork that has passing winners + allocator test green:

```bash
python3 "{{SCRIPTS_DIR}}/integrate/rebuild_libraries.py" \
    --output-dir "{{OUTPUT_DIR}}"
```

Reads `rebuild_command` from `{{OUTPUT_DIR}}/forks/manifest.json`
(per-library; pure-Python libs use `pip install -e .`, C++ ext libs use
`pip install -e . --no-build-isolation` and may need `AITER_REBUILD=1` or
`MAX_JOBS=...`). Logs to `{{RESULTS_DIR}}/rebuild_<lib>.log`. The
wheel-installed copy is shadowed by the editable install (Python's import
order resolves the dev path first).

### 2. Verify Dispatch with rocprofv3

```bash
python3 "{{SCRIPTS_DIR}}/integrate/verify_dispatch.py" \
    --output-dir "{{OUTPUT_DIR}}" --mode post-rebuild \
    --manifest "{{PROBLEMS_DIR}}/optimization_manifest.json"
```

Runs `rocprofv3 --kernel-trace --output-format json` on a minimal vLLM
decode, parses kernel symbol counts, and confirms per
`expected_dispatch_symbols` and `vendor_baseline_symbols`:
- Expected GEAK-optimized symbol(s) are present with `count > 0`.
- Vendor baseline symbol(s) are absent or `count == 0` (dispatch actually
  swapped, no double-load).
- For `dispatch_redirect_*` strategies, the redirect was honored: vendor
  symbol gone, redirect-target symbol present.
- Diffed against `baseline_dispatch_trace.json` (from Phase 6) for a
  structured before/after.

Writes `{{RESULTS_DIR}}/dispatch_verification.json`:
```
{
  "mode": "post-rebuild",
  "rocprofv3_trace_path": "...",
  "expected_symbols": [{name, count, status: present|missing}, ...],
  "vendor_symbols":   [{name, count, status: absent|leaked}, ...],
  "expected_symbol_total_count":   int,
  "vendor_symbol_leaked_count":    int,
  "redirect_required_count":       int,
  "redirect_honored_count":        int,
  "dispatch_verified":             bool
}
```

If `dispatch_verified == false`, abort -- there is no value in burning
e2e wall-clock when the patched kernel is not actually firing. Surface
the blocker classification per `protocols/rerun-protocol.md`
(`dispatch_unverified` or `redirect_not_honored`).

### 3. Run the Standard vLLM e2e Benchmark

Boot vLLM with the rebuilt forks via the same launcher Phase 2 used. No
plugin injection, no PYTHONPATH override, no `inject_plugin.py`.

```bash
python3 "{{SCRIPTS_DIR}}/integrate/run_e2e.py" \
    --output-dir "{{OUTPUT_DIR}}" \
    --launcher "$BASELINE_LAUNCHER" \
    --label "after_rebuild"
```

Writes `{{RESULTS_DIR}}/e2e_after_rebuild.log` and a `{e2e_ran,
returncode, duration_sec}` summary on stdout.

### 4. Validate Results
```bash
python3 "{{SCRIPTS_DIR}}/report/validate_optimization.py" --results-dir "{{RESULTS_DIR}}"
```

Writes `{{RESULTS_DIR}}/optimization_comparison.json` (kept name, kept
schema for adjacent compatibility -- only the build chain feeding it
changed). Treat `artifacts_valid = false` or `performance_gate = fail` as
validation failure. A `performance_gate = warn` result is still a usable
measured outcome.

### 5. Write Integration Manifest

After validation, write `{{RESULTS_DIR}}/integration_manifest.json` (kept
name, new schema):

```json
{
  "schema_version": "2.0",
  "libraries_rebuilt": [
    {"lib": "fla", "commit": "<sha>", "install_log_path": "results/rebuild_fla.log"},
    {"lib": "vllm", "commit": "<sha>", "install_log_path": "results/rebuild_vllm.log"}
  ],
  "dispatch_verified": true,
  "e2e_ran":           true,
  "artifacts_valid":   true
}
```

Populate from `{{RESULTS_DIR}}/dispatch_verification.json`,
`{{OUTPUT_DIR}}/forks/manifest.json`, and the `e2e_after_rebuild` summary.

### 6. Clean Up

```bash
docker stop "$CONTAINER_NAME" 2>/dev/null; docker rm "$CONTAINER_NAME" 2>/dev/null
```

### Completion

Write `agent-results/phase-08-result.md`. Include in `## Key Findings` for
monitor consumption:

- `dispatch_verified`: bool
- `expected_symbol_total_count`: integer
- `vendor_symbol_leaked_count`: integer
- `redirect_honored_count`: integer
- `redirect_required_count`: integer
- `libraries_rebuilt_ok_count`: integer
- `libraries_rebuild_failed_count`: integer
- `e2e_speedup`: float (from `optimization_comparison.json`)
- `validation_status`: pass | warn | fail (mirrors `performance_gate`)

Reference `results/dispatch_verification.json`,
`results/integration_manifest.json`,
`results/optimization_comparison.json`, and the per-library
`results/rebuild_<lib>.log` files in `## Artifacts`.

If the handoff contains a `## Root Cause Analysis` section from a prior
failed attempt, adjust per the new blocker enum in
`protocols/rerun-protocol.md`.

Do NOT write to `progress.json` -- the orchestrator manages progress
tracking.

### Removed Outputs (do NOT emit)

`optimized/integration_plugin/_runtime_report.json`, all
`_runtime_counters_attempt*_rank*.json`, `vllm_plugin/`,
`sglang_plugin/`, `bench_dispatch.py`, `launch_patched.py`,
`run_e2e_benchmark.sh`, `manual-edits-required-attempt*.md` (the plugin
escape hatch is gone with the plugin path).
