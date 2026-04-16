# Parity Contract

This document defines exactly which state transitions and artifact fields participate in the canonical parity hash. The parity hash proves that the deterministic runner produces the same control-plane decisions as the legacy orchestration path.

## Parity Hash Algorithm

SHA-256 over a deterministic JSON serialization of the included fields, sorted by key at every nesting level. The hash is computed after each phase completes and stored in `results/parity/parity-manifest.json`.

## Included Fields

These fields participate in the parity hash. Any difference between the runner path and legacy path in these fields is a parity violation.

### Phase Ordering

The ordered sequence of phase keys dispatched during the run, including phases skipped due to mode selection or conditional dependencies.

```json
{
  "phase_sequence": ["env", "config", "benchmark", "benchmark-analyze", "..."]
}
```

### Verdict Sequence

The ordered sequence of monitor verdicts (PASS / WARN / FAIL) for each dispatched phase, including verdicts from retried attempts.

```json
{
  "verdict_sequence": [
    {"phase": "env", "attempt": 1, "verdict": "PASS"},
    {"phase": "benchmark", "attempt": 1, "verdict": "FAIL"},
    {"phase": "benchmark", "attempt": 2, "verdict": "PASS"}
  ]
}
```

### Retry Counts

Per-phase retry count and total retry count at run completion.

```json
{
  "retry_counts": {"benchmark": 1, "kernel-optimize": 0},
  "total_reruns": 1
}
```

### Fallback Triggers

Ordered list of fallback activations, each recording the failing phase and the target phase it rolled back to.

```json
{
  "fallbacks_used": [
    {"phase_key": "profile-analyze", "fallback_target": "profile"}
  ]
}
```

### Blocker Emissions

Ordered list of pipeline blockers emitted, each recording the phase, terminal action, and blocker classifications. The blocker `summary` text is excluded (it may contain non-deterministic LLM prose).

```json
{
  "blockers_emitted": [
    {
      "phase": "kernel-optimize",
      "terminal_action": "budget_exhausted",
      "blocker_classifications": ["compilation_failure"]
    }
  ]
}
```

### Mode and Phase Selection

The selected mode and the resolved phase list after applying `SKIP_INTEGRATION` and `conditional_deps`.

```json
{
  "mode": "optimize",
  "resolved_phases": ["env", "config", "benchmark", "..."]
}
```

### Terminal State

The final pipeline status and completion flag.

```json
{
  "final_status": "completed",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "..."]
}
```

## Excluded Fields

These fields are explicitly excluded from the parity hash. Differences in these fields between the runner and legacy paths are expected and acceptable.

| Field | Reason for Exclusion |
|-------|---------------------|
| Timestamps (ISO 8601 dates, durations) | Wall-clock time is non-deterministic across runs. |
| File sizes | Byte counts vary with serialization whitespace, compression, etc. |
| Human-readable summaries | Monitor explanation text, RCA prose, and running-summary body text are LLM-generated and inherently non-deterministic. |
| Sticky value contents | The values themselves (e.g., `baseline_tpot: 12.5`) depend on hardware measurements, not control-plane logic. The fact that a sticky value *was set* is captured by the verdict sequence. |
| Evidence-manifest.json | Derived index — varies with file sizes and timestamps. |
| Log file paths and contents | Infrastructure-dependent. |
| Runner overhead timings | Checkpoint durations, dispatch latency. |

## Parity Manifest Format

After each phase, the runner writes (or updates) `results/parity/parity-manifest.json`:

```json
{
  "schema_version": "1.0",
  "run_id": "<output-dir-basename>",
  "parity_hash": "<sha256-hex>",
  "snapshot": {
    "phase_sequence": ["..."],
    "verdict_sequence": [{"...": "..."}],
    "retry_counts": {},
    "total_reruns": 0,
    "fallbacks_used": [],
    "blockers_emitted": [],
    "mode": "optimize",
    "resolved_phases": ["..."],
    "final_status": "running",
    "phases_completed": ["..."]
  },
  "computed_at": "<ISO 8601>"
}
```

The `computed_at` timestamp is metadata and is itself excluded from the hash computation.

## Parity Verification

Parity is verified by computing the hash over the included fields from both the runner path and the legacy path and asserting equality. The verification script is `scripts/orchestrate/verify_parity.py` <!-- planned: not yet implemented --> (added in Commit 4).

### Failure Modes

- **Hash mismatch**: The runner made a different control-plane decision than the legacy path. This blocks promotion (Commit 7) and must be investigated.
- **Missing manifest**: The runner did not emit a parity manifest. This is a runner bug.
- **Schema version mismatch**: The manifest uses a different schema version than expected. Apply the migration rules from ARCHITECTURE.md's rework protocol.

## V2 Fields Excluded from V1 Parity

The following fields are only present when `V2_MONITOR=true` and are excluded from the V1 parity hash:

| Field | Reason for Exclusion |
|-------|---------------------|
| `response_sequence` | V2 only. Records L1/L2 response actions per phase. Not present in V1 runs. |
| `human_interventions` | V2 only. Records human-in-the-loop escalation events. Not present in V1 runs. |
| `human_extensions` | V2 only. Records budget or scope extensions granted by humans. Not present in V1 runs. |

## Cutover Criteria

The runner path can become the default (Commit 7) only when:

1. Parity hash matches on the full frozen fixture set (representative + adversarial).
2. Structured predicate agreement reaches 95% (Commit 5 gate).
3. Rollback test procedure passes (ARCHITECTURE.md 5-step sequence).
4. `pass^3 = 100%` on the deterministic control-plane suite (three consecutive green runs).
