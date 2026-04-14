# Test Fixtures Manifest

Minimum cardinality: **8 representative** + **8 adversarial** fixtures.

Each fixture is a JSON file containing the scenario description, input data, and expected outcomes. Fixtures are loaded by `test_validate_optimization.py` and the control-plane suite.

## Representative Fixtures (8)

| ID | File | Scenario |
|----|------|----------|
| R1 | `representative/full_pass.json` | Full optimize mode, all phases pass, integration speedup > 1.0 |
| R2 | `representative/optimize_only_pass.json` | optimize-only mode, existing gap_analysis, clean pass |
| R3 | `representative/retry_then_pass.json` | Critical phase fails once, retries, succeeds on attempt 2 |
| R4 | `representative/blocker_partial_report.json` | Phase blocked, budget exhausted, partial report generated |
| R5 | `representative/skip_integration.json` | SKIP_INTEGRATION=true, no comparison expected, completed |
| R6 | `representative/warn_band.json` | Integration lands in 0.97-1.0 warn band, completed with warnings |
| R7 | `representative/ttft_upgrade.json` | Warn→fail upgrade due to severe TTFT regression |
| R8 | `representative/benchmark_mode.json` | benchmark-only mode, no integration phase, completed |

## Adversarial Fixtures (8)

| ID | File | Scenario |
|----|------|----------|
| A1 | `adversarial/malformed_monitor_json.json` | Monitor review output is not valid markdown frontmatter |
| A2 | `adversarial/malformed_rca_json.json` | RCA output missing required fields |
| A3 | `adversarial/partial_progress_json.json` | progress.json missing phases_completed or current_phase |
| A4 | `adversarial/missing_comparison.json` | optimization_comparison.json absent despite integration expected |
| A5 | `adversarial/empty_winners.json` | geak_results.json with zero winning kernels |
| A6 | `adversarial/unsupported_schema_version.json` | schema_version="99.0" in progress.json |
| A7 | `adversarial/corrupt_handoff.json` | Handoff YAML frontmatter malformed |
| A8 | `adversarial/budget_exhausted.json` | All retry budgets consumed, multiple fallbacks used |

## Fixture Format

Each fixture JSON has the structure:

```json
{
  "id": "R1",
  "name": "full_pass",
  "description": "...",
  "mode": "optimize",
  "inputs": {
    "progress": { ... },
    "comparison": { ... },
    "blockers": { ... },
    "config": { ... }
  },
  "expected": {
    "pipeline_status": "completed",
    "all_phases_completed": true,
    "performance_gate": "pass",
    "artifacts_valid": true
  }
}
```

## Adding Fixtures

When expanding the fixture set (required before Commit 5 cutover with minimum 50+20):

1. Add the JSON file to the appropriate directory.
2. Update this manifest.
3. Run the full control-plane suite to verify the new fixture.
