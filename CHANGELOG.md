# Changelog

## v2.0.0

Multi-agent architecture rebuild of the InferenceX optimization pipeline.

### Added
- Multi-agent orchestration system (`orchestrator/ORCHESTRATOR.md`, `phase-registry.json`, `monitor.md`)
- 10 self-contained phase agents under `agents/phase-NN-*.md`
- Communication protocols: handoff format, phase result schema, monitor feedback schema, rerun protocol, analyzer manifest
- Monitor agent with quality checks, failure taxonomy, and rolling summary
- GEAK-accelerated kernel optimization (Triton + HIP/CK modes)
- Declarative `conditional_deps` in phase registry for skip-integration flow
- Phase timeout policy with per-phase overrides
- Interrupt/resume guidance via `progress.json` state tracking
- Monitor failure handling (non-blocking, does not consume rerun budget)
- E2E validator (`tests/e2e_optimize_test.py`) with multi-agent workspace validation

### Changed
- Reorganized scripts from flat layout to categorized subdirectories (`env/`, `container/`, `profiling/`, `optimize/`, `plugin/`, `report/`)
- Phase runbooks moved from `phases/*.md` (now archive stubs) to `agents/phase-NN-*.md`
- `progress.json` now includes `retry_counts`, `current_phase`, `status`, `total_reruns`, `fallbacks_used`
- `MONITOR_LEVEL` dispatch now uses 3-branch selection (minimal/strict/standard)

### Removed
- `vllm-optimize` skill (consolidated into `inferencex-optimize`)
- `benchmark+profile` mode alias (use `full` instead)
- Deprecated `--dest` flag from `install.sh`
