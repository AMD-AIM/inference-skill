---
name: vllm-optimize
description: "Run vLLM benchmark, profiling, and kernel optimization workflow for AMD and NVIDIA GPUs. Provides automated model loading, concurrency sweep, torch profiling, GPU kernel analysis, and kernel optimization with GEAK support."
compatibility: claude-code, opencode
metadata:
  workflow: vllm
  audience: performance-engineers
  distribution: standalone-skill-repo
---

# vLLM Optimize Skill

End-to-end vLLM inference performance optimization: benchmark → profile → kernel analysis → Triton optimization → E2E verification → report.

**Primary target**: AMD GPUs (RDNA3 gfx1100, CDNA3 MI300X/MI355X). NVIDIA supported.

**Optimization target**: Triton kernels only.

---

## The One Thing You Must Understand

**Micro speedup ≠ E2E speedup.** Real measured failures on AMD RDNA3 + vLLM:

| What happened | Micro | E2E | Root cause |
|---|---|---|---|
| Autotune on GEMM | 1.1x faster | 2.3x **slower** | Every new batch size triggers 100-iteration sweep |
| Shape-specific fast path | 1.8x on covered shapes | 0.94x overall | Fast path covers 60%, fallback overhead dominates |
| Fused RMSNorm | 2.1x faster | 0.97x | Not on critical path at serving concurrency |

Every gate and validation step in this skill exists because of these failures. **Do not skip them.**

---

## Pipeline (7 phases)

```
Phase 0: env              — GPU detection, vLLM/PyTorch verification, env_info.json
Phase 1: server           — Kill stale vLLM, download model, start server with profiler
Phase 2: bench-profile    — Concurrency sweep benchmark + profiler trace at same workload
Phase 3: analysis         — Kernel breakdown, real shapes from trace, optimization targets
Phase 4: optimize         — Triton kernel optimization loop with five-gate trust chain
Phase 5: integrate        — Deploy via PYTHONPATH hook, E2E verification at all concurrencies
Phase 6: report           — Verdict table, per-concurrency TPS, recommendations
```

The vLLM server starts once in Phase 1 and runs through Phase 5. No restart between phases.

---

## Hard Constraints (non-negotiable)

1. **Real shapes only** — All kernel optimization shapes come from profiler traces with `record_shapes=True`. Never from model config.
2. **Five-gate trust chain** — Gate 1: serving readiness; Gate 2: shape coverage; Gate 3: correctness; Gate 4: per-concurrency E2E TPS ≥ baseline; Gate 5: single-user decode no regression.
3. **No system files** — Never modify `/opt/`, `/usr/`, pip packages. Use PYTHONPATH + import hooks.
4. **Same workload** — Benchmark and profile at identical ISL/OSL/concurrency.
5. **Kill EngineCore first** — When stopping vLLM, kill `VLLM::EngineCore` before the API server.
6. **Honest reporting** — Final report MUST include verdict table for EVERY kernel target with micro speedup, serving readiness, E2E speedup, and verdict.
7. **AMD first** — Default `DTYPE=bfloat16`. Kernel patterns: `hipblaslt_*`, `ck_*`, `aiter::*`. Always apply `tl.assume(stride > 0)` (5-15% gain).

---

## Modes

- `optimize` (default): all 7 phases — full pipeline
- `profile-only`: phases 0-3 — benchmark, profile, analyze; no optimization
- `optimize-only`: phases 0, 1, 4-6 — requires existing Phase 3 artifacts

---

## Guided Intake

See [`INTAKE.md`](INTAKE.md) for the full intake algorithm.

**Round 1 questions** (all at once, first turn):
- `Run plan`: optimize / profile-only / optimize-only
- `Output`: new timestamped dir / custom path
- `GPUs`: auto-select / current env / specify IDs

After Round 1, read [`RUNTIME.md`](RUNTIME.md) and do lightweight GPU discovery, then ask:
- TP, ISL/OSL, concurrency, optimization budget (one batched form)

Confirm plan → execute.

---

## Status Contract

- One short kickoff message before Round 1 questions.
- `Phase N/6: <name> — starting` before each phase.
- `Phase N/6: <name> — done` after each phase.
- Heartbeat every 30-60 seconds during long-running commands.
- Stream stdout/stderr with `tee` for all benchmark/profile runs.

---

## Files

| File | Purpose |
|------|---------|
| [`INTAKE.md`](INTAKE.md) | Full intake algorithm and question sets |
| [`RUNTIME.md`](RUNTIME.md) | Variable map, workspace bootstrap, execution rules |
| [`phases/00-env-setup.md`](phases/00-env-setup.md) | Phase 0 |
| [`phases/01-server-setup.md`](phases/01-server-setup.md) | Phase 1 |
| [`phases/02-bench-profile.md`](phases/02-bench-profile.md) | Phase 2 |
| [`phases/03-analysis.md`](phases/03-analysis.md) | Phase 3 |
| [`phases/04-kernel-optimize.md`](phases/04-kernel-optimize.md) | Phase 4 |
| [`phases/05-integration.md`](phases/05-integration.md) | Phase 5 |
| [`phases/06-report.md`](phases/06-report.md) | Phase 6 |
| [`scripts/select_gpus.py`](scripts/select_gpus.py) | GPU selection |
| [`scripts/extract_shapes.py`](scripts/extract_shapes.py) | Real shape extraction from traces |
| [`scripts/kernel_breakdown.py`](scripts/kernel_breakdown.py) | Kernel time breakdown from traces |
| [`scripts/kernel_agent.py`](scripts/kernel_agent.py) | Optimization scaffold (setup/benchmark/correctness/accept/reject/serving-test) |
| [`scripts/gemm_patch.py`](scripts/gemm_patch.py) | Runtime GEMM dispatch patching via meta_path hook |
| [`references/TRITON_KNOWLEDGE.md`](references/TRITON_KNOWLEDGE.md) | AMD Triton optimization patterns |

---

## First-Turn Rule

Do not read any file before sending the first visible reply. Send one short kickoff status, then the Round 1 question form.
