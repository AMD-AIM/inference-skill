# Inference Harness Architecture

This document defines the canonical architecture for the Inference optimization harness. It is the single source of truth for the four-plane model, evidence lifecycle, rollback procedure, and rework protocol.

## Four-Plane Model

The harness is organized into four connected planes. Code owns mechanics; markdown agents own reasoning.

### Policy Plane

Defines what the system should do. All contracts, schemas, and protocol documents live here.

- **ORCHESTRATOR.md**: Dispatch loop, handoff generation, monitor invocation, rerun rules, progress tracking.
- **phase-registry.json**: Phase metadata, mode maps, quality checks, detection rules, context sources, timeouts, rerun budgets.
- **monitor.md**: Monitor agent role, review process, verdict assignment, running summary maintenance.
- **Protocol docs**: `handoff-format.md`, `phase-result.schema.md`, `monitor-feedback.schema.md`, `rerun-protocol.md`, `analyzer-manifest.schema.md`.
- **Versioned JSON schemas** (added by Commit 2): `progress.schema.json`, `monitor-verdict.schema.json`, `rca.schema.json`, `runner-failure.schema.json`.

### Control Plane

Deterministic Python code that owns sequencing, prerequisites, retry budgets, resume validation, rollback, handoff generation, context-source resolution, context-budget enforcement, and atomic `progress.json` writes.

Today this role is filled by the LLM orchestrator agent. Starting at Commit 4, `scripts/orchestrate/runner.py` takes over mechanical orchestration while the LLM retains intake UX, monitor explanation, and RCA interpretation.

Key components (post-runner):
- **DeterministicRunner**: Mode resolution, dependency checks, artifact prerequisites, timeout handling, retry budgets, fallback invalidation, handoff skeleton generation, atomic progress writes.
- **ContextBudgetEnforcer**: Resolves `context_sources` to sidecar files, compresses accumulated state (key + verdict + sticky only), enforces `max_context_lines` before each dispatch.
- **ParityChecker**: Computes and validates the canonical parity hash per `PARITY_CONTRACT.md`.

### Execution Plane

Markdown-driven phase agents, monitor agent, and RCA agent. These write attempt-scoped artifacts and human-readable summaries. They receive context from the control plane and return structured results.

- **Phase agents**: `agents/phase-00-env-setup.md` through `agents/phase-09-report-generate.md`.
- **Monitor agent**: `orchestrator/monitor.md` — spawned fresh after each phase.
- **RCA agent**: `agents/analysis-agent.md` — spawned on critical-phase FAIL.

### Evidence Plane

All runtime artifacts produced during a run. Governed by the evidence lifecycle rules below.

- `progress.json`: Run state, phases completed, retry counts, current phase, fallbacks used.
- Attempt-scoped artifacts: `agent-results/`, `monitor/`, `handoff/`.
- Parity manifests: Per-run parity hash and field snapshots (starting Commit 4).
- Validator outputs: `test_report.json`, `test_report.md` from the e2e validator.

## Non-Negotiable Invariants

These are enforced by `tests/test_invariants.py`, not just prose.

| ID | Invariant | Enforced From |
|----|-----------|---------------|
| INV-1 | `progress.json` has exactly one writer on the runner path: the runner. | Commit 0 (stub), Commit 4 (real) |
| INV-2 | Every control-plane JSON artifact carries `schema_version` and is validated on write and on resume. | Commit 2 |
| INV-3 | Critical phases fail closed on missing or malformed monitor JSON, RCA JSON, handoff data, or validator-critical artifacts. | Commit 3 |
| INV-4 | Attempt artifacts are immutable and attempt-scoped; retries append, never mutate. | Commit 4 |
| INV-5 | The deterministic runner is the default orchestration path; legacy remains available via explicit rollback (`USE_RUNNER=false`) with parity/rollback proof. | Commit 7+ |
| INV-6 | LLM judgment stays only where it adds value: intake UX, monitor explanation, RCA interpretation. Mechanical work moves to code. | Commit 4 |
| INV-7 | Release-critical PASS/WARN/FAIL authority comes from schemas, validators, report logic, runner state, and structured predicates. Prose monitor output is explanatory evidence, not final authority. | Commit 5 |
| INV-8 | Registry fields must be implemented and tested or absent. No dormant documented contracts. | All commits |

## Evidence Lifecycle Rules

Every artifact in the evidence plane follows these rules.

### Naming

- Attempt-scoped: `{plane}/{phase-key}-{attempt-N}.{ext}` (e.g., `agent-results/phase-02-attempt-1.md`).
- Singleton artifacts keep their current names: `progress.json`, `running-summary.md`.

### Retention

- Attempt-scoped artifacts are retained for the life of the run directory.
- Parity manifests are retained until the next successful parity check replaces them.
- Validator outputs are overwritten per run.

### Cleanup

- `scripts/orchestrate/cleanup.py` <!-- planned: not yet implemented --> removes parity manifests older than 7 days from completed runs.
- No automatic deletion of attempt artifacts.

### Index

- `evidence-manifest.json` is regenerated by the runner after each phase, listing all current artifacts with paths, sizes, and timestamps.

## Rollback Test Procedure

Every push gate that says "rollback proof" means this concrete 5-step sequence:

1. Set `USE_RUNNER=false` (or equivalent flag) in the run configuration.
2. Re-run the full frozen fixture suite on the legacy path.
3. Assert zero new failures relative to the pre-milestone baseline fixture report.
4. Assert `progress.json` is not written by any non-runner code path (checked by a test that monitors file writes).
5. Record the fixture report diff as the rollback proof artifact.

## Rework Protocol

When a later milestone reveals that an earlier contract needs changing:

1. The fix lands as a patch to the earliest affected milestone's test suite first.
2. The patch must pass all push gates for the affected milestone and every subsequent milestone that has already landed.
3. The patch is a separate commit, not folded into the later milestone's commit.
4. If the contract change is backward-incompatible, add a `schema_version` bump and explicit migration behavior in the resume path.

## V2 Monitor

V2 Monitor adds deterministic L1 predicates as a floor below the existing L2 (LLM) monitor. Final verdict = max(L1, L2) by severity rank.

### L1/L2 Relationship

L1 evaluates structured predicate rules against phase context scalars. L2 (LLM judgment) can only upgrade verdicts (PASS->WARN, WARN->FAIL), never downgrade.

### Predicate Categories

The L1 predicate taxonomy covers nine categories:

1. `infrastructure`
2. `logic`
3. `data_quality`
4. `performance_regression`
5. `effort_waste`
6. `cross_kernel_interference`
7. `geak_false_claim`
8. `baseline_drift`
9. `stale_artifact`

### V2 Invariants

| ID | Invariant | Notes |
|----|-----------|-------|
| INV-9 | `V2_MONITOR=false` preserves exact V1 behavior (verified by golden-file regression tests). | Commit 8+ |
| INV-10 | REDIRECT and ABORT are response actions, never verdicts. Verdicts are strictly PASS/WARN/FAIL. | Commit 8+ |
| INV-11 | Phase agents are idempotent — they overwrite outputs unconditionally, never skip-if-exists. | Commit 8+ |

### Predicate Result Shape

Each phase's L1 predicate evaluation is written to `monitor/phase-{NN}-predicate.json`. This file records which predicates fired, their individual verdicts, and the merged L1 verdict for that phase.

## In-Flight Migration Rule

Runs started under the legacy path before the runner lands:

- Resume under the legacy path. Do not force-migrate mid-run.
- The runner path applies to new runs started with `USE_RUNNER=true` (default for skill-guided runs).
- `progress.json` written by the legacy path is read-compatible with the runner path (validated by resume tests in Commit 2).
