# Monitor Mode — Upgraded Optimize Mode with Full Transparency

Monitor mode is optimize mode with full transparency. The same phases execute, the same quality checks apply, the same verdicts are issued. The upgrade: the main agent surfaces the **complete** monitor sub-agent findings to the user after every phase boundary.

## Architecture

```
Same dispatch loop as optimize mode:

  for each phase in [env, config, benchmark, ..., report-generate]:
    1. Spawn phase agent → produces result doc
    2. Spawn monitor sub-agent → evaluates result, writes review + context
    3. Handle verdict (retry/RCA/fallback as normal)
    4. Read full review + context + running-summary → present to user    ← THE UPGRADE
```

No new agents, no new modes, no new infrastructure. The monitor sub-agent already produces all the diagnostic data. The orchestrator reads it and shows it.

## What the User Sees

After each phase completes, the user receives a structured digest:

```
## Phase {NN}: {phase_name} [{verdict}]

**Monitor Assessment:**
{Narrative summary from monitor/phase-{NN}-review.md}

**Quality Checks:**
- {check_name}: {PASS|FAIL} — {details}

**Detection Rules:**
- {field} {op} {threshold} → actual: {value} → {triggered|ok}

**Key Scalars:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| {metric} | {value} | {threshold or —} | {pass/warn/fail} |

**Running Totals:**
- Phases completed: {N}/{total}
- Reruns so far: {count}
- Sticky values: {from running-summary.md frontmatter}
```

For WARN or FAIL phases: the **Failure Details** and **Rerun Guidance** sections from the review doc are shown verbatim.

## Phase-Specific Rich Output

Beyond the standard digest, surface these phase-specific details:

### Phase 02: Benchmark
- Throughput (tok/s)
- TTFT median and p99 latency
- Number of benchmark runs completed

### Phase 03: Benchmark Analyze
- Bottleneck count and categories
- Gap analysis summary
- Best concurrency point

### Phase 05: Profile Analyze
- GPU utilization %
- Roofline coverage %
- Top kernel breakdown (name, time %, category)
- Allreduce overhead %
- Phase-split status

### Phase 06: Problem Generate
- Number of optimization targets
- Excluded kernel categories (e.g., communication) and their time %
- Problem grouping summary (fused vs. individual)

### Phase 07: Kernel Optimize
Per-kernel status table from `problems/geak_results.json`:

| Kernel | Status | Speedup | Method |
|--------|--------|---------|--------|
| {name} | compiled/blocked/failed | {x}x | GEAK/manual |

Also show: `problems/optimization_manifest.json` target list, blocked reasons, total compiled vs. total targets.

### Phase 08: Integration
- Performance gate: pass/warn/fail
- E2E speedup factor (e.g., 0.973x)
- TTFT regression % (e.g., +56%)
- Per-kernel integration status
- Baseline vs. optimized comparison table

### Phase 09: Report Generate
- Report file paths
- Final speedup number
- Blockers summary (from `results/pipeline_blockers.json` if present)

## Final Summary

After the last phase completes, present a consolidated report:

```
# Final Summary: {CONFIG_KEY}

## Overall Verdict: {PASS | WARN | FAIL}
(Maximum severity across all phases)

## Per-Phase Results
| Phase | Index | Critical | Verdict | Key Finding |
|-------|-------|----------|---------|-------------|
| env | 00 | No | PASS | — |
| config | 01 | No | PASS | — |
| benchmark | 02 | Yes | PASS | 2064 tok/s, TTFT p99=1012ms |
| ... | ... | ... | ... | ... |

## Key Metrics
- Throughput: {tok/s}
- E2E Speedup: {factor}x
- TTFT Regression: {pct}%
- GPU Utilization: {pct}%
- Roofline Coverage: {pct}%
- Kernels: {compiled}/{total} optimized, {blocked} blocked

## Recommendations
- {Actionable recommendations based on WARN/FAIL findings}
```

## Post-Hoc Analysis

When MODE=monitor and the user provides an existing run directory (from intake), the orchestrator runs the same dispatch loop against existing artifacts:

1. Read `progress.json` to determine which phases completed.
2. For each completed phase:
   - Phase agents **re-read** existing artifacts (they do not re-execute benchmark/profile/optimize).
   - Monitor sub-agents evaluate the results and write reviews.
   - The main agent presents the full digest — same format, same transparency.
3. Present the Final Summary after processing all completed phases.

The user sees exactly what they would have seen during a live run, applied retroactively.

## In-Progress Runs

If `progress.json` shows `status: "running"`:

- Only process phases listed in `phases_completed`.
- Note `current_phase` in the overview.
- Label the report as **"Partial Analysis (run in progress)"**.
- Do not assign an overall verdict — state that the run is incomplete.

## Interactive Follow-Up

After the final summary, remain available for drill-down questions:

- "Show me the details of phase X" — read and present the full phase result doc
- "What went wrong with kernel optimization?" — read `problems/geak_results.json`, `problems/optimization_manifest.json`
- "Show the comparison data" — read `results/optimization_comparison.json`
- "What does the profile analysis show?" — read `results/profile_analysis.json`
- "Show the gap analysis" — read `results/gap_analysis/gap_analysis.json`

Read additional artifacts on demand. Do not pre-read all artifacts — only read what the user asks about or what is needed for quality checks.
