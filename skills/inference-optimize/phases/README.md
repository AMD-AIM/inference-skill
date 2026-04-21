# Phase Docs (Optional Archive)

The original phase runbooks that lived here (`00-env-setup.md` through `09-report-generate.md`) have been superseded by the multi-agent architecture.
This directory is retained only as an archive pointer for backward-compatible references.

Canonical phase agent docs are now at:

- `agents/phase-00-env-setup.md`
- `agents/phase-01-config-parse.md`
- `agents/phase-02-benchmark.md`
- `agents/phase-03-benchmark-analyze.md`
- `agents/phase-04-profile.md`
- `agents/phase-05-profile-analyze.md`
- `agents/phase-06-problem-generate.md`
- `agents/phase-07-kernel-optimize.md`
- `agents/phase-08-integration.md`
- `agents/phase-09-report-generate.md`

## Runtime Path

The **canonical runtime path** uses the deterministic runner (`scripts/orchestrate/runner.py`) for all mechanical orchestration. The runner owns sequencing, prerequisites, retry budgets, context-budget enforcement, atomic progress writes, and parity artifact emission.

To revert to the legacy LLM orchestrator path, set `USE_RUNNER=false` in the run configuration.

### Key documents

- **Architecture**: `docs/ARCHITECTURE.md` — four-plane model, invariants, lifecycle rules
- **Parity contract**: `docs/PARITY_CONTRACT.md` — which fields participate in parity verification
- **Runner-LLM boundary**: `protocols/runner-llm-boundary.md` — what the runner handles vs. LLM agents
- **Orchestrator dispatch**: `orchestrator/ORCHESTRATOR.md` and `orchestrator/phase-registry.json`
