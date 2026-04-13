# Phase Docs (Archive)

The original phase runbooks that lived here (`00-env-setup.md` through `09-report-generate.md`) have been superseded by the multi-agent architecture.

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

The orchestrator dispatches work via `orchestrator/ORCHESTRATOR.md` and `orchestrator/phase-registry.json`. Phase agents read their own doc plus a handoff file -- they do not read these archive stubs.
