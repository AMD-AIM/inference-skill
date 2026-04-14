#!/usr/bin/env python3
"""Deterministic runner for the InferenceX optimization harness.

Owns all mechanical orchestration: mode resolution, dependency checks,
artifact prerequisites, context-source resolution, context-budget enforcement,
retry budgets, fallback invalidation, handoff generation, atomic progress
writes, and parity artifact emission.

Shadow mode (default): runs alongside the legacy orchestrator without
overriding it. Set USE_RUNNER=true to make this the active path.

Usage:
    python3 runner.py --config <config.json> --registry <phase-registry.json> \
                      --output-dir <dir> [--shadow]
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import tempfile

SCHEMA_VERSION = "1.0"
MAX_CONTEXT_LINES_DEFAULT = 500


def load_json(path):
    with open(path) as f:
        return json.load(f)


def atomic_write_json(path, data):
    """Write JSON atomically: write to temp, then rename."""
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirname, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def compute_parity_hash(snapshot):
    """SHA-256 over deterministic JSON serialization of parity fields."""
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def truncate_context(lines, max_lines):
    """Deterministic truncation with marker."""
    if len(lines) <= max_lines:
        return lines
    keep = max_lines - 1
    omitted = len(lines) - keep
    return lines[:keep] + [f"[truncated: {omitted} lines omitted]"]


class RunnerState:
    """Mutable run state — single writer of progress.json."""

    def __init__(self, output_dir, mode, resolved_phases):
        self.output_dir = output_dir
        self.mode = mode
        self.resolved_phases = resolved_phases
        self.phases_completed = []
        self.current_phase = None
        self.status = "running"
        self.retry_counts = {}
        self.total_reruns = 0
        self.fallbacks_used = []
        self.verdict_sequence = []
        self.blockers_emitted = []

    def to_progress(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "phases_completed": list(self.phases_completed),
            "current_phase": self.current_phase,
            "status": self.status,
            "retry_counts": dict(self.retry_counts),
            "total_reruns": self.total_reruns,
            "fallbacks_used": list(self.fallbacks_used),
            "mode": self.mode,
            "resolved_phases": list(self.resolved_phases),
        }

    def write_progress(self):
        path = os.path.join(self.output_dir, "progress.json")
        atomic_write_json(path, self.to_progress())

    def parity_snapshot(self):
        return {
            "phase_sequence": list(self.resolved_phases),
            "verdict_sequence": list(self.verdict_sequence),
            "retry_counts": dict(self.retry_counts),
            "total_reruns": self.total_reruns,
            "fallbacks_used": list(self.fallbacks_used),
            "blockers_emitted": list(self.blockers_emitted),
            "mode": self.mode,
            "resolved_phases": list(self.resolved_phases),
            "final_status": self.status,
            "phases_completed": list(self.phases_completed),
        }

    def write_parity_manifest(self):
        snapshot = self.parity_snapshot()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": os.path.basename(self.output_dir),
            "parity_hash": compute_parity_hash(snapshot),
            "snapshot": snapshot,
            "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        parity_dir = os.path.join(self.output_dir, "results", "parity")
        os.makedirs(parity_dir, exist_ok=True)
        atomic_write_json(os.path.join(parity_dir, "parity-manifest.json"), manifest)

    @classmethod
    def from_progress(cls, output_dir, progress):
        mode = progress.get("mode", "optimize")
        resolved = progress.get("resolved_phases", [])
        state = cls(output_dir, mode, resolved)
        state.phases_completed = progress.get("phases_completed", [])
        state.current_phase = progress.get("current_phase")
        state.status = progress.get("status", "running")
        state.retry_counts = progress.get("retry_counts", {})
        state.total_reruns = progress.get("total_reruns", 0)
        state.fallbacks_used = progress.get("fallbacks_used", [])
        return state


class DeterministicRunner:
    """Control-plane runner implementing the dispatch loop mechanically."""

    def __init__(self, config, registry, output_dir, shadow=False):
        self.config = config
        self.registry = registry
        self.output_dir = output_dir
        self.shadow = shadow
        self.max_context_lines = registry.get("max_context_lines", MAX_CONTEXT_LINES_DEFAULT)
        self.phases = registry["phases"]
        self.modes = registry["modes"]
        self.rerun_limits = registry["rerun"]
        self.timeouts = registry["timeouts"]
        self.context_sources = registry["context_sources"]

    def resolve_mode(self):
        mode = self.config.get("MODE", "optimize")
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}")
        return mode

    def resolve_phase_list(self, mode):
        phase_list = list(self.modes[mode])
        skip_integration = str(self.config.get("SKIP_INTEGRATION", "false")).lower() == "true"
        if skip_integration:
            phase_list = [p for p in phase_list if p != "integration"]
            for p_key in phase_list:
                p = self.phases.get(p_key, {})
                cdeps = p.get("conditional_deps")
                if cdeps and cdeps.get("when_absent") == "integration":
                    pass
        return phase_list

    def check_prerequisites(self, phase_key):
        phase = self.phases[phase_key]
        errors = []
        for artifact in phase.get("requires_artifacts", []):
            path = os.path.join(self.output_dir, artifact)
            if not os.path.exists(path):
                errors.append(f"Missing artifact: {artifact}")
        return errors

    def resolve_context(self, phase_key):
        """Resolve required_context for a phase to concrete values."""
        phase = self.phases[phase_key]
        context = {}
        for var in phase.get("required_context", []):
            source = self.context_sources.get(var, {})
            src_type = source.get("source", "config")
            if src_type == "config":
                context[var] = self.config.get(var, "")
            elif src_type == "artifact":
                artifact_path = os.path.join(self.output_dir, source.get("path", ""))
                field = source.get("field")
                if os.path.isfile(artifact_path):
                    try:
                        data = load_json(artifact_path)
                        context[var] = data.get(field, "") if field else data
                    except (json.JSONDecodeError, OSError):
                        context[var] = ""
                else:
                    context[var] = ""
            else:
                context[var] = self.config.get(var, "")
        return context

    def build_handoff(self, phase_key, context, attempt, prior_feedback=None):
        """Generate a handoff document."""
        phase = self.phases[phase_key]
        lines = [
            "---",
            f"phase: {phase_key}",
            f"phase_index: {phase['index']}",
            f"attempt: {attempt}",
            f"mode: {self.config.get('MODE', 'optimize')}",
            "---",
            "",
            "## Context",
        ]
        for k, v in sorted(context.items()):
            val = str(v) if not isinstance(v, str) else v
            lines.append(f"- **{k}**: {val}")

        lines.extend(["", "## Instructions"])
        lines.append(f"Execute phase {phase_key} (index {phase['index']}).")
        lines.append(f"Write results to agent-results/phase-{phase['index']:02d}-result.md.")

        if prior_feedback:
            lines.extend(["", "## Prior Attempt Feedback"])
            lines.append(prior_feedback)

        lines = truncate_context(lines, self.max_context_lines)
        return "\n".join(lines)

    def write_handoff(self, phase_key, content):
        handoff_dir = os.path.join(self.output_dir, "handoff")
        os.makedirs(handoff_dir, exist_ok=True)
        idx = self.phases[phase_key]["index"]
        path = os.path.join(handoff_dir, f"to-phase-{idx:02d}.md")
        with open(path, "w") as f:
            f.write(content)
        return path

    def write_runner_failure(self, error_type, message, phase=None):
        failure = {
            "schema_version": SCHEMA_VERSION,
            "error_type": error_type,
            "message": message,
            "phase": phase,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "recoverable": error_type not in {"budget_exhausted", "manual_intervention_required"},
        }
        path = os.path.join(self.output_dir, "runner_failure.json")
        atomic_write_json(path, failure)
        return failure

    def get_timeout(self, phase_key):
        overrides = self.timeouts.get("overrides", {})
        return overrides.get(phase_key, self.timeouts.get("default_minutes", 30))

    def run(self, dispatch_fn=None):
        """Execute the dispatch loop.

        dispatch_fn: callable(phase_key, handoff_path) -> verdict_dict
            If None (shadow mode), simulates PASS for all phases.
        """
        mode = self.resolve_mode()
        phase_list = self.resolve_phase_list(mode)
        state = RunnerState(self.output_dir, mode, phase_list)
        state.write_progress()

        for phase_key in phase_list:
            state.current_phase = phase_key
            state.write_progress()

            prereq_errors = self.check_prerequisites(phase_key)
            if prereq_errors:
                self.write_runner_failure("missing_artifact", "; ".join(prereq_errors), phase_key)
                state.status = "failed"
                state.write_progress()
                state.write_parity_manifest()
                return state

            phase_reruns = 0
            while True:
                attempt = phase_reruns + 1
                context = self.resolve_context(phase_key)
                feedback = None
                if phase_reruns > 0:
                    feedback = f"Attempt {attempt} after {phase_reruns} prior failure(s)."

                handoff = self.build_handoff(phase_key, context, attempt, feedback)
                handoff_path = self.write_handoff(phase_key, handoff)

                if dispatch_fn:
                    verdict = dispatch_fn(phase_key, handoff_path)
                else:
                    verdict = {"verdict": "PASS", "attempt": attempt}

                state.verdict_sequence.append({
                    "phase": phase_key,
                    "attempt": attempt,
                    "verdict": verdict.get("verdict", "PASS"),
                })

                v = verdict.get("verdict", "PASS")
                if v == "PASS" or v == "WARN":
                    state.phases_completed.append(phase_key)
                    state.retry_counts[phase_key] = phase_reruns
                    break

                state.total_reruns += 1
                phase_reruns += 1

                if (phase_reruns > self.rerun_limits["max_per_phase"]
                        or state.total_reruns > self.rerun_limits["max_total"]):
                    phase = self.phases[phase_key]
                    ft = phase.get("fallback_target")
                    already_used = any(
                        f["phase_key"] == phase_key and f["fallback_target"] == ft
                        for f in state.fallbacks_used
                    ) if ft else True

                    if ft and not already_used:
                        state.fallbacks_used.append({
                            "phase_key": phase_key,
                            "fallback_target": ft,
                        })
                        fb_idx = phase_list.index(ft) if ft in phase_list else 0
                        state.phases_completed = state.phases_completed[:fb_idx]
                        phase_reruns = 0
                        break
                    else:
                        state.blockers_emitted.append({
                            "phase": phase_key,
                            "terminal_action": "budget_exhausted",
                            "blocker_classifications": [],
                        })
                        tp = phase.get("terminal_policy")
                        if tp == "allow_partial_report" and "report-generate" in phase_list:
                            state.phases_completed.append(phase_key)
                            break
                        else:
                            self.write_runner_failure("budget_exhausted", f"Budget exhausted for {phase_key}", phase_key)
                            state.status = "failed"
                            state.write_progress()
                            state.write_parity_manifest()
                            return state

            state.write_progress()
            state.write_parity_manifest()

        state.current_phase = None
        state.status = "completed"
        state.write_progress()
        state.write_parity_manifest()
        return state


def main():
    parser = argparse.ArgumentParser(description="Deterministic runner")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--registry", required=True, help="Path to phase-registry.json")
    parser.add_argument("--output-dir", required=True, help="Run output directory")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode (no real dispatch)")
    args = parser.parse_args()

    config = load_json(args.config)
    registry = load_json(args.registry)
    runner = DeterministicRunner(config, registry, args.output_dir, shadow=args.shadow)
    state = runner.run()

    print(f"Runner finished: status={state.status}, "
          f"phases={len(state.phases_completed)}, "
          f"reruns={state.total_reruns}")
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
