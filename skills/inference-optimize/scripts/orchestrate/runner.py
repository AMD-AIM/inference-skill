#!/usr/bin/env python3
"""Deterministic runner for the inference optimization harness.

Owns all mechanical orchestration: mode resolution, dependency checks,
artifact prerequisites, context-source resolution, context-budget enforcement,
retry budgets, fallback invalidation, handoff generation, atomic progress
writes, and parity artifact emission.

Runner is the canonical control plane for skill-guided runs. Legacy behavior
remains available only when USE_RUNNER=false is set explicitly.

Usage:
    python3 runner.py --config <config.json> --registry <phase-registry.json> \
                      --output-dir <dir>
"""

import argparse
from collections.abc import Mapping, Sequence
import datetime
import hashlib
import json
import logging
import os
import sys
import tempfile

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
MAX_CONTEXT_LINES_DEFAULT = 8000
CONTEXT_VALUE_CHAR_LIMIT_DEFAULT = 800
CONTEXT_KEYS_PREVIEW_DEFAULT = 10
CONTEXT_ITEMS_PREVIEW_DEFAULT = 5
CURSOR_AGENT_DOC_MAX_LINES_DEFAULT = 600


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
    """Deterministic truncation with marker (disabled when max_lines <= 0)."""
    if max_lines is None or max_lines <= 0:
        return lines
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
        self.human_extensions = {}
        self.rca_skipped = {}  # phase_key -> reason string for shadow/test RCA callback fallback
        # budget_mode replaces the legacy ad-hoc budget_override.unlimited.
        # default = enforce positive rerun.max_per_phase/max_total caps from registry
        # diagnostic = orchestrator-set after systemic RCA accept_finding (no further e2e attempts)
        # extended = user-authorized only, lifts the per-phase/total caps
        self.budget_mode = {
            "mode": "default",
            "set_by": "orchestrator",
            "set_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rationale": "initial state",
        }
        # rca_history tracks per-attempt fingerprints for repeated-failure detection
        self.rca_history = []  # list of {phase, attempt, fingerprint, root_cause_class, timestamp}
        # Per-phase repeated-failure tracking:
        #   rca_fingerprints[phase_key]            -> latest fingerprint for the phase
        #   last_changed_handoff_hash[phase_key]   -> hash of the most recent handoff content
        #   systemic_rca_triggered[phase_key]      -> bool, true after systemic RCA fired for the phase
        self.rca_fingerprints = {}
        self.last_changed_handoff_hash = {}
        self.systemic_rca_triggered = {}
        # Populated by enter_awaiting_user_instruction(); surfaced via to_progress()
        # so an external caller can render the user-decision question.
        self.terminal_state = None
        self.awaiting_user_instruction_phase = None

    def to_progress(self):
        progress = {
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
        if self.human_extensions:
            progress["human_extensions"] = dict(self.human_extensions)
        if self.rca_skipped:
            progress["rca_skipped"] = dict(self.rca_skipped)
        if self.rca_fingerprints:
            progress["rca_fingerprints"] = dict(self.rca_fingerprints)
        if self.systemic_rca_triggered:
            progress["systemic_rca_triggered"] = dict(self.systemic_rca_triggered)
        if self.terminal_state is not None:
            progress["terminal_state"] = dict(self.terminal_state)
        if self.awaiting_user_instruction_phase:
            progress["awaiting_user_instruction_phase"] = self.awaiting_user_instruction_phase
        # Always emit budget_mode so its provenance survives restarts.
        progress["budget_mode"] = dict(self.budget_mode)
        return progress

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
        state.human_extensions = progress.get("human_extensions", {})
        state.rca_fingerprints = progress.get("rca_fingerprints", {})
        state.systemic_rca_triggered = progress.get("systemic_rca_triggered", {})
        state.terminal_state = progress.get("terminal_state")
        state.awaiting_user_instruction_phase = progress.get(
            "awaiting_user_instruction_phase"
        )
        history_path = os.path.join(output_dir, "results", "rca_history.json")
        if os.path.isfile(history_path):
            try:
                history_data = load_json(history_path)
                state.rca_history = history_data.get("history", [])
            except (json.JSONDecodeError, OSError):
                state.rca_history = []
        if "budget_mode" in progress:
            state.budget_mode = dict(progress["budget_mode"])
        return state


class DeterministicRunner:
    """Control-plane runner implementing the dispatch loop mechanically."""

    V2_ONLY_KEYS = {"HUMAN_LOOP", "CROSS_KERNEL_INTERACTION_THRESHOLD",
                     "CROSS_KERNEL_FALLBACK_PCT", "ESCALATION_WARN_SECONDS",
                     "ESCALATION_ABORT_SECONDS", "MAX_HUMAN_EXTENSIONS"}

    def __init__(self, config, registry, output_dir, shadow=False):
        self.config = config
        self.shadow = shadow
        self.registry = registry
        self.output_dir = output_dir
        self.max_context_lines = int(registry.get("max_context_lines", MAX_CONTEXT_LINES_DEFAULT))
        self.context_value_char_limit = int(
            registry.get("context_value_char_limit", CONTEXT_VALUE_CHAR_LIMIT_DEFAULT)
        )
        self.context_keys_preview = int(
            registry.get("context_keys_preview", CONTEXT_KEYS_PREVIEW_DEFAULT)
        )
        self.context_items_preview = int(
            registry.get("context_items_preview", CONTEXT_ITEMS_PREVIEW_DEFAULT)
        )
        self.cursor_agent_doc_max_lines = int(
            registry.get("cursor_agent_doc_max_lines", CURSOR_AGENT_DOC_MAX_LINES_DEFAULT)
        )
        self.phases = registry["phases"]
        self.modes = registry["modes"]
        self.rerun_limits = registry["rerun"]
        self.timeouts = registry["timeouts"]
        self.context_sources = registry["context_sources"]

        # Standalone test handle for repeated_rca_fingerprint(). The live
        # dispatch loop reads state.rca_history (RunnerState) instead.
        self.rca_history = []

        # V2 monitor feature flag
        self.v2_monitor = str(config.get("V2_MONITOR", registry.get("v2_monitor", False))).lower() in ("true", "1")
        self.human_loop = str(config.get("HUMAN_LOOP", "false")).lower() in ("true", "1")
        self._validate_v2_config()

    def _validate_v2_config(self):
        """Warn if V2-only config keys are set without V2_MONITOR enabled."""
        if self.v2_monitor:
            return
        set_keys = [k for k in self.V2_ONLY_KEYS if self.config.get(k) is not None]
        if set_keys:
            if "HUMAN_LOOP" in set_keys and self.human_loop:
                logger.warning("HUMAN_LOOP=true requires V2_MONITOR -- auto-promoting")
                self.v2_monitor = True
                return
            logger.warning(f"V2-only config keys set but V2_MONITOR=false (ignored): {set_keys}")

    def resolve_mode(self):
        mode = self.config.get("MODE", "optimize")
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}")
        return mode

    def resolve_phase_list(self, mode):
        phase_list = list(self.modes[mode])
        dep_overrides = {}
        skip_integration = str(self.config.get("SKIP_INTEGRATION", "false")).lower() == "true"
        if skip_integration:
            phase_list = [p for p in phase_list if p != "integration"]
            for p_key in phase_list:
                p = self.phases.get(p_key, {})
                cdeps = p.get("conditional_deps")
                if cdeps and cdeps.get("when_absent") not in phase_list:
                    dep_overrides[p_key] = [cdeps["fallback_dep"]]
        return phase_list, dep_overrides

    def check_prerequisites(self, phase_key):
        phase = self.phases[phase_key]
        errors = []
        for artifact in phase.get("requires_artifacts", []):
            path = os.path.join(self.output_dir, artifact)
            if not os.path.exists(path):
                errors.append(f"Missing artifact: {artifact}")
        if phase_key == "integration":
            optimized_dir = os.path.join(self.output_dir, "optimized")
            has_optimized_artifact = False
            if os.path.isdir(optimized_dir):
                for _root, _dirs, files in os.walk(optimized_dir):
                    if any(not name.startswith(".") for name in files):
                        has_optimized_artifact = True
                        break
            integration_manifest = os.path.join(
                self.output_dir, "problems", "integration_manifest.json")
            redirect_manifest = os.path.join(
                self.output_dir, "problems", "redirect_integration_manifest.json")
            if (not has_optimized_artifact
                    and not os.path.isfile(integration_manifest)
                    and not os.path.isfile(redirect_manifest)):
                errors.append(
                    "Missing integration input: optimized/ is empty and no "
                    "alternative integration manifest exists")
        return errors

    def phases_requiring_rca(self, phase_list):
        """Return critical phases in this run that require RCA wiring."""
        required = []
        for phase_key in phase_list:
            phase_meta = self.phases.get(phase_key, {})
            if phase_meta.get("critical") and phase_meta.get("rca_artifact"):
                required.append(phase_key)
        return required

    def _resolve_from_config(self, source, var):
        """Resolve a variable from config source."""
        config_key = source.get("config_key", var)
        return self.config.get(config_key, "")

    def _resolve_from_artifact(self, source):
        """Resolve a variable from an artifact file."""
        fallback = source.get("artifact_fallback", source)
        artifact_path = os.path.join(self.output_dir, fallback.get("path", source.get("path", "")))
        field = fallback.get("field", source.get("field"))
        if os.path.isfile(artifact_path):
            try:
                data = load_json(artifact_path)
                return data.get(field, "") if field else data
            except (json.JSONDecodeError, OSError):
                return ""
        return ""

    @staticmethod
    def _is_present(value):
        """Return True when a resolved context value should be considered usable."""
        return value not in ("", None, [], {}, ())

    def _resolve_context_entry(self, var, source):
        """Resolve one context variable and record where it came from."""
        src_type = source.get("source", "config")
        config_key = source.get("config_key", var)

        def resolve_candidate(candidate):
            if candidate == "config":
                return self._resolve_from_config(source, var), {
                    "resolved_source": "config",
                    "config_key": config_key,
                }

            if candidate == "artifact":
                fallback = source.get("artifact_fallback", source)
                return self._resolve_from_artifact(source), {
                    "resolved_source": "artifact",
                    "path": fallback.get("path", source.get("path", "")),
                    "field": fallback.get("field", source.get("field")),
                }

            return self.config.get(var, ""), {
                "resolved_source": "config",
                "config_key": var,
            }

        if isinstance(src_type, list):
            for candidate in src_type:
                value, meta = resolve_candidate(candidate)
                if self._is_present(value):
                    return value, meta
            return "", {
                "resolved_source": "none",
                "config_key": config_key,
            }

        value, meta = resolve_candidate(src_type)
        return value, meta

    def resolve_context_with_meta(self, phase_key):
        """Resolve required_context for a phase to concrete values + source metadata."""
        phase = self.phases[phase_key]
        context = {}
        context_meta = {}
        for var in phase.get("required_context", []):
            source = self.context_sources.get(var, {})
            value, meta = self._resolve_context_entry(var, source)
            context[var] = value
            context_meta[var] = meta
        return context, context_meta

    def resolve_context(self, phase_key):
        """Resolve required_context for a phase to concrete values."""
        context, _ = self.resolve_context_with_meta(phase_key)
        return context

    @staticmethod
    def _truncate_text(text, max_chars):
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars]}... [truncated: {omitted} chars omitted]"

    def _summarize_list_item(self, item):
        if isinstance(item, Mapping):
            keys = sorted(str(k) for k in item.keys())[: self.context_keys_preview]
            return {"type": "dict", "keys_preview": keys}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return {"type": "list", "size": len(item)}
        if isinstance(item, str):
            return self._truncate_text(item, min(80, self.context_value_char_limit))
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        return self._truncate_text(str(item), min(80, self.context_value_char_limit))

    def _summarize_structured_value(self, value):
        if isinstance(value, Mapping):
            keys = sorted(str(k) for k in value.keys())
            return {
                "type": "dict",
                "size": len(keys),
                "keys_preview": keys[: self.context_keys_preview],
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            preview = [
                self._summarize_list_item(item)
                for item in list(value)[: self.context_items_preview]
            ]
            return {
                "type": "list",
                "size": len(value),
                "items_preview": preview,
            }
        return value

    def _render_source_hint(self, meta):
        if not meta:
            return ""
        resolved_source = meta.get("resolved_source")
        if resolved_source == "artifact":
            path = meta.get("path", "")
            field = meta.get("field")
            locator = path if not field else f"{path}#{field}"
            return f"source={locator}" if locator else "source=artifact"
        if resolved_source == "config":
            config_key = meta.get("config_key")
            return f"source=config:{config_key}" if config_key else "source=config"
        return ""

    def _format_context_value(self, value, meta=None):
        """Render context values with deterministic compaction for large payloads."""
        if isinstance(value, str):
            rendered = self._truncate_text(value, self.context_value_char_limit)
        elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
            try:
                serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
            except TypeError:
                serialized = repr(value)
            if len(serialized) <= self.context_value_char_limit:
                rendered = serialized
            else:
                summary = self._summarize_structured_value(value)
                rendered = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        else:
            rendered = self._truncate_text(str(value), self.context_value_char_limit)

        source_hint = self._render_source_hint(meta)
        if source_hint:
            return f"{rendered} ({source_hint})"
        return rendered

    def build_handoff(self, phase_key, context, attempt, prior_feedback=None, dep_overrides=None, context_meta=None):
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
            meta = context_meta.get(k, {}) if context_meta else {}
            val = self._format_context_value(v, meta)
            lines.append(f"- **{k}**: {val}")

        deps = phase.get("deps", [])
        if dep_overrides and phase_key in dep_overrides:
            deps = dep_overrides[phase_key]
        lines.extend(["", f"## Dependencies: {', '.join(deps) if deps else 'none'}", ""])
        lines.extend(["## Instructions"])
        lines.append(f"Execute phase {phase_key} (index {phase['index']}).")
        lines.append(f"Write results to agent-results/phase-{phase['index']:02d}-result.md.")

        if prior_feedback:
            lines.extend(["", "## Prior Attempt Feedback"])
            lines.append(prior_feedback)

        lines = truncate_context(lines, self.max_context_lines)
        return "\n".join(lines)

    @staticmethod
    def build_retry_feedback(attempt, monitor_verdict, rca_result):
        """Build rich retry feedback for the next handoff.

        Phase agents consume this under `## Prior Attempt Feedback`.
        Include the RCA path/guidance here because the deterministic
        runner does not rewrite handoffs after RCA in-place; the next
        loop iteration will call build_handoff() with this text.
        """
        parts = [f"Attempt {attempt} failed."]
        if monitor_verdict:
            failure_type = monitor_verdict.get("failure_type")
            if failure_type:
                parts.append(f"- **Failure type**: {failure_type}")
            summary = monitor_verdict.get("summary") or monitor_verdict.get("analysis")
            if summary:
                parts.append(f"- **Monitor summary**: {summary}")
        if rca_result:
            parts.append("")
            parts.append("## Root Cause Analysis")
            artifact = rca_result.get("artifact") or rca_result.get("output_path")
            if artifact:
                parts.append(f"- **RCA artifact**: {artifact}")
            summary = rca_result.get("summary") or rca_result.get("analysis")
            if summary:
                parts.append(f"- **Summary**: {summary}")
            recommendation = rca_result.get("retry_recommendation")
            if recommendation:
                parts.append(f"- **Retry recommendation**: {recommendation}")
            guidance = rca_result.get("retry_guidance")
            if guidance:
                parts.append("")
                parts.append("### Retry Guidance")
                parts.append(str(guidance))
            blockers = rca_result.get("blocker_classifications")
            if blockers:
                parts.append(f"- **Blocker classifications**: {blockers}")
        return "\n".join(parts)

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

    def get_agent_doc_path(self, phase_key):
        """Return the path to the phase agent's markdown doc."""
        agent_file = self.phases[phase_key].get("agent", "")
        return os.path.join(self.output_dir, "..", "agents", agent_file) if agent_file else ""

    def build_cursor_prompt(self, phase_key, handoff_content, agent_doc_root=None):
        """Assemble a bounded Cursor Task prompt: capped agent doc + handoff."""
        agent_file = self.phases[phase_key].get("agent", "")
        doc_content = ""
        if agent_doc_root and agent_file:
            doc_path = os.path.join(agent_doc_root, agent_file)
            if os.path.isfile(doc_path):
                with open(doc_path) as f:
                    doc_content = f.read()
        parts = []
        if doc_content:
            doc_lines = doc_content.split("\n")
            if self.cursor_agent_doc_max_lines > 0:
                doc_lines = truncate_context(doc_lines, self.cursor_agent_doc_max_lines)
            parts.append("\n".join(doc_lines))
        parts.append(handoff_content)
        combined = "\n\n---\n\n".join(parts)
        lines = combined.split("\n")
        lines = truncate_context(lines, self.max_context_lines)
        return "\n".join(lines)

    def derive_sticky_inputs(self):
        """Auto-derive stick scalars from filesystem state.

        Currently:
          - phase_split_inputs_ready: true when results/phase_split/ exists with
            at least one regular file. Removes the need for any caller to set
            this manually.

        Returns a dict of derived sticky scalars; callers merge this into the
        running summary frontmatter.
        """
        derived = {}
        ps_dir = os.path.join(self.output_dir, "results", "phase_split")
        ps_ready = False
        if os.path.isdir(ps_dir):
            for name in os.listdir(ps_dir):
                if os.path.isfile(os.path.join(ps_dir, name)):
                    ps_ready = True
                    break
        derived["phase_split_inputs_ready"] = ps_ready
        return derived

    def compute_rca_fingerprint(self, root_cause_class, key_signal_names):
        """Deterministic fingerprint = sha256(root_cause_class + '|' + sorted(signals))."""
        signals = ",".join(sorted(key_signal_names or []))
        material = f"{root_cause_class or ''}|{signals}"
        return hashlib.sha256(material.encode()).hexdigest()

    def archive_prior_rca(self, phase_key, attempt):
        """Copy results/<phase>_rca.json to <phase>_rca_attempt{N}.json
        before the next attempt overwrites it. Returns the prior fingerprint
        if the archived file declares one, else None."""
        phase_meta = self.phases.get(phase_key, {})
        rca_artifact = phase_meta.get("rca_artifact") or {}
        rca_path_rel = rca_artifact.get("output")
        if not rca_path_rel:
            return None
        rca_path = os.path.join(self.output_dir, rca_path_rel)
        if not os.path.isfile(rca_path):
            return None
        # Archive
        base, ext = os.path.splitext(rca_path)
        archive_path = f"{base}_attempt{attempt}{ext}"
        try:
            with open(rca_path) as src:
                content = src.read()
            with open(archive_path, "w") as dst:
                dst.write(content)
        except OSError:
            pass
        # Recover fingerprint from the archived content
        try:
            data = json.loads(content)
        except (ValueError, NameError):
            return None
        return data.get("fingerprint")

    def repeated_rca_fingerprint(self, phase_key, current_fingerprint):
        """Return True if the previous per-phase RCA for this phase shared the
        same fingerprint as the current one. Reads from state.rca_history
        (the runner appends each attempt's fingerprint to it)."""
        if not current_fingerprint:
            return False
        prior = [h for h in self.rca_history if h.get("phase") == phase_key]
        if not prior:
            return False
        return prior[-1].get("fingerprint") == current_fingerprint

    def enter_diagnostic_budget_mode(self, state, rationale):
        """Enter budget_mode = diagnostic. No further e2e benchmark attempts
        will be dispatched. Caller is responsible for routing to report-generate."""
        state.budget_mode = {
            "mode": "diagnostic",
            "set_by": "orchestrator",
            "set_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rationale": rationale,
        }
        state.write_progress()

    def budget_caps_enforced(self, state):
        """Budget caps enforcement is bypassed only in extended mode (user-set).
        diagnostic mode short-circuits BEFORE the cap check (no further attempts
        are dispatched at all), so for diagnostic we still report 'enforced'.
        """
        return state.budget_mode.get("mode") != "extended"

    @staticmethod
    def rerun_limit_exceeded(current_value, limit_value):
        """Positive limits are enforced; non-positive values mean uncapped retries."""
        return limit_value is not None and limit_value > 0 and current_value > limit_value

    def rerun_budgets_exhausted(self, state, phase_reruns):
        """Return True only when an active rerun cap has been exceeded."""
        return (
            self.rerun_limit_exceeded(phase_reruns, self.rerun_limits.get("max_per_phase", 0))
            or self.rerun_limit_exceeded(state.total_reruns, self.rerun_limits.get("max_total", 0))
        )

    def write_pipeline_blockers(self, state):
        """Persist state.blockers_emitted to results/pipeline_blockers.json."""
        blockers_path = os.path.join(self.output_dir, "results", "pipeline_blockers.json")
        atomic_write_json(blockers_path, {
            "schema_version": SCHEMA_VERSION,
            "blockers": list(state.blockers_emitted),
        })

    def write_user_decision_request(self, phase_key, reason, rca_result, review_path):
        """Pause the run for a user decision after a critical FAIL.

        Replaces the legacy ``allow_partial_report`` auto-skip path. The
        runner emits a structured request file under
        ``monitor/user_decision_request.json``; an external caller (the
        outer dispatcher, an LLM orchestrator, or a CLI prompt) chooses
        the next step (retry with new instructions, fallback, generate
        a user-requested report, or stop). The runner never picks
        partial-report autonomously.
        """
        monitor_dir = os.path.join(self.output_dir, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        request = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase_key,
            "reason": reason,
            "monitor_review": review_path,
            "rca_artifact": (rca_result or {}).get("artifact"),
            "rca_summary": (rca_result or {}).get("analysis")
                or (rca_result or {}).get("summary"),
            "rca_terminal_action": (rca_result or {}).get("terminal_action"),
            "rca_fingerprint": (rca_result or {}).get("fingerprint"),
            "rca_root_cause_class": (rca_result or {}).get("root_cause_class"),
            "options": [
                {
                    "id": "retry",
                    "label": "Retry the failed phase with new instructions",
                    "requires": "additional_handoff_guidance",
                },
                {
                    "id": "fallback",
                    "label": "Roll back and re-run the dependency phase",
                    "requires": "fallback_target",
                },
                {
                    "id": "stop",
                    "label": "Stop the run and leave artifacts as-is",
                },
                {
                    "id": "generate_report_anyway",
                    "label": "Force Phase 9 to render a partial report",
                    "warning": "non-default; the run is NOT a clean optimization",
                },
            ],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        atomic_write_json(os.path.join(monitor_dir, "user_decision_request.json"),
                          request)
        return request

    def enter_awaiting_user_instruction(self, state, phase_key, reason,
                                         rca_result=None, review_path=None,
                                         blocker_extra=None):
        """Mark the run as ``awaiting_user_instruction`` and pause.

        The caller is responsible for returning to the dispatcher after
        this method, since the run cannot continue until a user
        decision arrives. The terminal_state block is populated so
        downstream consumers can render the user-decision question.
        """
        terminal = {
            "outcome": "awaiting_user_instruction",
            "blocking_phase": phase_key,
            "reason": reason,
            "rca_stop_recommended": bool(
                rca_result and rca_result.get("terminal_action") == "stop_with_blocker"
            ),
            "user_instruction_required": True,
            "rca_fingerprint": (rca_result or {}).get("fingerprint"),
            "retry_exhausted": reason == "budget_exhausted",
        }
        state.terminal_state = terminal
        blocker = {
            "phase": phase_key,
            "terminal_action": reason,
            "blocker_classifications": (rca_result or {}).get(
                "blocker_classifications", []
            ),
            "rca_artifact": (rca_result or {}).get("artifact"),
            "user_instruction_required": True,
        }
        if blocker_extra:
            blocker.update(blocker_extra)
        state.blockers_emitted.append(blocker)
        self.write_pipeline_blockers(state)
        self.write_user_decision_request(
            phase_key=phase_key,
            reason=reason,
            rca_result=rca_result,
            review_path=review_path,
        )
        state.status = "awaiting_user_instruction"
        state.awaiting_user_instruction_phase = phase_key
        state.write_progress()

    # --- V2 stub methods (filled by Phases 3-5) ---

    def _evaluate_v2(self, phase_key, verdict, v, result_path=None):
        """V2 two-layer monitor: L1 predicates as floor, L2 (LLM) can only upgrade."""
        return self.evaluate_structured_predicates(
            phase_key, verdict, v, result_path=result_path, v2=True)

    @staticmethod
    def normalize_monitor_verdict(verdict_value):
        """Normalize monitor verdicts to the binary contract.

        Only ``PASS`` and ``FAIL`` are accepted. Any other value is
        treated as ``FAIL``. This covers legacy ``WARN`` artifacts read
        during resume as well as defensive normalization of new
        non-binary verdicts such as ``PASS_with_caveats`` or
        ``FAIL_pushed_through`` — the phase-orchestrator's
        self-checklist refuses to advance until the monitor rewrites
        such reviews with a real binary verdict, but the runner
        normalizes here so that a buggy callback cannot silently mark a
        critical phase as PASS.
        """
        if verdict_value == "PASS":
            return "PASS"
        return "FAIL"

    @staticmethod
    def is_invalid_verdict(verdict_value):
        """True when the input verdict is neither PASS nor FAIL.

        Used to surface monitor contract violations as structured
        ``invalid_verdict`` blocker entries; the runner still proceeds
        with the FAIL-normalized verdict so the loop converges.
        """
        return verdict_value not in ("PASS", "FAIL")

    def _maybe_escalate(self, phase_key, v, verdict):
        """Check if human escalation is needed. Returns escalation result or None."""
        if not self.human_loop:
            return None
        if not verdict.get("escalation_required"):
            return None

        try:
            from . import escalation
        except ImportError:
            import escalation

        phase_meta = self.phases[phase_key]
        esc_context = escalation.build_escalation_context(
            phase_key=phase_key,
            verdict=v,
            failure_type=verdict.get("failure_type", "unknown"),
            layer1_details=None,
            monitor_review_path=os.path.join(
                self.output_dir, "monitor",
                f"phase-{phase_meta['index']:02d}-review.md"),
            rca_summary=None,
            running_summary_path=os.path.join(
                self.output_dir, "monitor", "running-summary.md"),
            kernel_status_path=os.path.join(
                self.output_dir, "monitor", "kernel-status.jsonl"),
        )
        escalation.write_escalation_request(self.output_dir, phase_key, esc_context)
        return {"action": "escalation_pending", "context": esc_context}

    def _determine_response(self, v, phase_key, phase_meta, state, rca_result,
                            escalation_result, phase_reruns, phase_list):
        """Determine response action for a FAIL verdict via response policy."""
        try:
            from .response_policy import determine_response
        except ImportError:
            from response_policy import determine_response

        # Check for escalation pending first
        if escalation_result and escalation_result.get("action") == "escalation_pending":
            return {"action": "escalation_pending",
                    "context": escalation_result.get("context"),
                    "phase_reruns": phase_reruns}

        state.total_reruns += 1
        phase_reruns += 1

        response = determine_response(
            verdict=v,
            failure_type=rca_result.get("failure_type", "unknown") if rca_result else "unknown",
            phase_key=phase_key,
            phase_meta=phase_meta,
            runner_state=state,
            rca_result=rca_result,
            escalation_result=escalation_result,
            rerun_limits=self.rerun_limits,
            phase_reruns=phase_reruns,
            phase_list=phase_list,
        )

        action = response.get("action", "retry")

        # Track human extensions when response policy grants a budget extension
        reason = response.get("reason", "")
        if action == "retry" and "budget extended" in reason:
            state.human_extensions.setdefault(phase_key, 0)
            state.human_extensions[phase_key] += 1

        if action == "redirect":
            target = response.get("target")
            ft = target or phase_meta.get("fallback_target")
            if ft:
                already_used = any(
                    f["phase_key"] == phase_key and f["fallback_target"] == ft
                    for f in state.fallbacks_used
                )
                if not already_used:
                    state.fallbacks_used.append({
                        "phase_key": phase_key,
                        "fallback_target": ft,
                    })
                fb_idx = phase_list.index(ft) if ft in phase_list else 0
                return {"action": "redirect", "target": ft, "target_idx": fb_idx,
                        "phase_reruns": phase_reruns}
            # No valid target, fall through to abort
            action = "abort"

        if action == "abort":
            state.blockers_emitted.append({
                "phase": phase_key,
                "terminal_action": response.get("reason", "budget_exhausted"),
                "blocker_classifications": rca_result.get("blocker_classifications", []) if rca_result else [],
            })
            self.write_pipeline_blockers(state)
            return {"action": "abort",
                    "error_type": response.get("reason", "budget_exhausted"),
                    "message": response.get("message", f"Aborted at {phase_key}"),
                    "phase_reruns": phase_reruns}

        if action == "continue":
            # NOTE: The legacy `allow_partial_report` auto-skip path was
            # removed. After a critical FAIL we never silently jump to
            # report-generate; the runner pauses with
            # ``awaiting_user_instruction`` from the FAIL branch instead.
            return {"action": "continue", "phase_reruns": phase_reruns}

        # Default: retry
        return {"action": "retry", "phase_reruns": phase_reruns}

    def _invoke_phase_rca(self, phase_key, phase_meta, verdict, verdict_severity, rca_fn, state):
        """Run per-phase RCA if configured.

        Non-shadow runs should fail closed earlier in run() when rca_fn is
        missing for RCA-required phases. The branch below remains as a
        defensive fallback for shadow/test paths.
        """
        rca_artifact = phase_meta.get("rca_artifact")
        if not rca_artifact:
            return None

        if rca_fn is None:
            logger.warning(
                "rca_fn not wired; skipping RCA for phase %s "
                "(rca_artifact=%s). This fallback is intended only for "
                "shadow/test paths; real runs should fail closed at startup.",
                phase_key,
                rca_artifact.get("output", "?"),
            )
            state.rca_skipped[phase_key] = "rca_fn_not_wired"
            return None

        rca_manifest = {
            "phase": phase_key,
            "rca_artifact": rca_artifact,
            "failure_type": verdict.get("failure_type", "unknown"),
            "verdict_severity": verdict_severity,
        }
        return rca_fn(phase_key, rca_manifest)

    def _resume_from_escalation(self, state, phase_list):
        """Check for pending escalation and resume if response exists.

        Returns (should_resume, phase_idx, phase_reruns) or (False, 0, 0).
        """
        if state.status != "escalation_pending":
            return False, 0, 0

        esc_ctx = getattr(state, "escalation_context", None)
        if not esc_ctx:
            return False, 0, 0

        phase_key = esc_ctx.get("phase")
        if phase_key not in phase_list:
            return False, 0, 0

        try:
            from . import escalation
        except ImportError:
            import escalation
        response = escalation.read_escalation_response(self.output_dir, phase_key)
        if response is None:
            if escalation.is_escalation_stale(esc_ctx):
                logger.warning(f"Stale escalation for {phase_key} -- auto-aborting")
                self.write_runner_failure("manual_intervention_required",
                                         f"Escalation timed out for {phase_key}", phase_key)
                state.status = "failed"
                state.write_progress()
                return False, 0, 0
            return False, 0, 0

        state.status = "running"
        phase_idx = phase_list.index(phase_key)
        phase_reruns = state.retry_counts.get(phase_key, 0)
        return True, phase_idx, phase_reruns

    def _record_rca_fingerprint(self, state, phase_key, attempt, rca_result):
        """Compute fingerprint, archive prior RCA, append to rca_history.
        Returns the fingerprint (or None if not derivable)."""
        if not rca_result:
            return None
        # Archive the prior file BEFORE this attempt's RCA may have overwritten it.
        # In practice the rca_fn dispatcher writes the file, so the file on disk now
        # is the *current* attempt's RCA — archive it under attempt={attempt}.
        self.archive_prior_rca(phase_key, attempt)
        rcc = rca_result.get("root_cause_class")
        signals = rca_result.get("key_signal_names", []) or []
        fp = rca_result.get("fingerprint") or self.compute_rca_fingerprint(rcc, signals)
        state.rca_history.append({
            "phase": phase_key,
            "attempt": attempt,
            "fingerprint": fp,
            "root_cause_class": rcc,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        # Per-phase latest fingerprint surfaces in progress.json so the
        # outer dispatcher can render repeated-failure context without
        # re-reading the full history file.
        state.rca_fingerprints[phase_key] = fp
        # Persist
        history_path = os.path.join(self.output_dir, "results", "rca_history.json")
        atomic_write_json(history_path, {
            "schema_version": SCHEMA_VERSION,
            "history": list(state.rca_history),
        })
        return fp

    @staticmethod
    def _hash_handoff(content):
        """Stable hash of handoff content used to detect repeated
        attempts that did not change the retry plan."""
        return hashlib.sha256((content or "").encode()).hexdigest()

    @staticmethod
    def _parse_scalar_value(raw_value):
        value = str(raw_value).strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        value = value.strip("`").strip()
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in ("null", "none", "n/a"):
            return None
        try:
            if any(ch in value for ch in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("\"'")

    def extract_result_scalars(self, result_path):
        """Extract flat `field: value` scalars from phase result markdown.

        Phase docs require registry-consumed values as flat scalar lines.
        This parser intentionally accepts a few common Markdown forms:

        - `field: value`
        - `- field: value`
        - `- `field`: value`
        - ``field: value``
        """
        scalars = {}
        if not result_path or not os.path.isfile(result_path):
            return scalars
        try:
            with open(result_path) as f:
                lines = f.readlines()
        except OSError:
            return scalars

        for line in lines:
            text = line.strip()
            if not text or text.startswith("|"):
                continue
            if text.startswith("-"):
                text = text[1:].strip()
            if text.startswith("*"):
                text = text[1:].strip()
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            key = key.strip().strip("`").strip()
            if not key.replace("_", "").isalnum() or not key:
                continue
            scalars[key] = self._parse_scalar_value(value)
        return scalars

    def load_monitor_context(self, phase_key):
        phase_meta = self.phases[phase_key]
        idx = phase_meta["index"]
        path = os.path.join(self.output_dir, "monitor", f"phase-{idx:02d}-context.json")
        if not os.path.isfile(path):
            return {}
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def build_predicate_context(self, phase_key, verdict, result_path=None):
        """Build structured predicate context from all machine sources.

        Precedence, from lowest to highest:
        1. monitor/phase-NN-context.json (artifact scalars)
        2. phase result flat scalars
        3. monitor callback verdict dictionary
        """
        context = {}
        context.update(self.load_monitor_context(phase_key))
        context.update(self.extract_result_scalars(result_path))
        if isinstance(verdict, Mapping):
            context.update(verdict)
        return context

    def evaluate_structured_predicates(self, phase_key, verdict, current_verdict,
                                       result_path=None, v2=False):
        """Evaluate registry structured predicates for both V1 and V2 paths.

        The V1 path previously trusted the monitor verdict directly.
        This method makes `detection_rules_structured` an active floor
        across all critical phases by reading the phase result scalars
        and monitor context in addition to the monitor callback payload.
        """
        try:
            from .predicate_engine import (
                evaluate_predicates, evaluate_predicates_v2, VERDICT_RANK)
        except ImportError:
            from predicate_engine import (
                evaluate_predicates, evaluate_predicates_v2, VERDICT_RANK)

        phase_meta = self.phases[phase_key]
        quality = phase_meta.get("quality", {})
        if v2:
            rules = quality.get(
                "detection_rules_structured_v2",
                quality.get("detection_rules_structured", []),
            )
        else:
            rules = quality.get("detection_rules_structured", [])
        if not rules:
            return current_verdict

        context = self.build_predicate_context(phase_key, verdict, result_path)

        thresholds = {}
        cross_thresh = self.config.get("CROSS_KERNEL_INTERACTION_THRESHOLD")
        if cross_thresh is not None:
            thresholds["CROSS_KERNEL_THRESHOLD"] = float(cross_thresh)
        elif os.path.isfile(os.path.join(self.output_dir, "results", "benchmark_noise.json")):
            try:
                noise = load_json(os.path.join(self.output_dir, "results", "benchmark_noise.json"))
                if "stddev_pct" in noise:
                    thresholds["CROSS_KERNEL_THRESHOLD"] = 3.0 * noise["stddev_pct"]
            except (json.JSONDecodeError, OSError):
                pass
        if "CROSS_KERNEL_THRESHOLD" not in thresholds:
            fallback_pct = float(self.config.get("CROSS_KERNEL_FALLBACK_PCT", 15.0))
            thresholds["CROSS_KERNEL_THRESHOLD"] = fallback_pct

        if v2:
            l1_verdict, details, categories = evaluate_predicates_v2(
                rules, context, thresholds)
        else:
            l1_verdict, details = evaluate_predicates(rules, context, thresholds)
            categories = []

        predicate_result = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase_key,
            "verdict": l1_verdict,
            "rules_evaluated": len(details),
            "rules_triggered": sum(1 for d in details if d.get("triggered")),
            "problem_categories": categories,
            "details": details,
            "context_fields": sorted(context.keys()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        monitor_dir = os.path.join(self.output_dir, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        idx = phase_meta["index"]
        atomic_write_json(
            os.path.join(monitor_dir, f"phase-{idx:02d}-predicate.json"),
            predicate_result,
        )
        return (
            l1_verdict
            if VERDICT_RANK.get(l1_verdict, 0) >= VERDICT_RANK.get(current_verdict, 0)
            else current_verdict
        )

    def _maybe_dispatch_systemic_rca(self, state, phase_key, attempt, current_fp,
                                     systemic_rca_fn, phase_list):
        """If the current RCA fingerprint matches the prior attempt's, dispatch
        the systemic RCA agent and act on its terminal_action_systemic.

        Returns one of:
          None                         -- no systemic dispatch (no match, or fn missing)
          {"action": "continue"}       -- resume per-phase retry path
          {"action": "fallback", "target": <phase>, "target_idx": <idx>}
          {"action": "accept_finding"} -- enter diagnostic mode, route to report-generate
        """
        # repeated_rca_fingerprint compares current to the *last* in history.
        # The current attempt was just appended, so we look at the prior entry.
        prior_entries = [h for h in state.rca_history if h.get("phase") == phase_key]
        if len(prior_entries) < 2 or not current_fp:
            return None
        if prior_entries[-2].get("fingerprint") != current_fp:
            return None
        if systemic_rca_fn is None:
            logger.warning(
                "Repeated RCA fingerprint detected for phase %s (attempt %d) "
                "but systemic_rca_fn is not wired. Pausing for user instruction.",
                phase_key, attempt)
            return {"action": "accept_finding", "reason": "systemic_rca_fn_missing"}

        manifest = {
            "phase": phase_key,
            "attempt": attempt,
            "fingerprint": current_fp,
            "rca_history": list(state.rca_history),
        }
        sys_rca = systemic_rca_fn(phase_key, manifest) or {}
        action = sys_rca.get("terminal_action_systemic", "continue")
        if action == "fallback":
            target = sys_rca.get("suggested_fallback_target")
            if target and target in phase_list:
                return {"action": "fallback", "target": target,
                        "target_idx": phase_list.index(target)}
            logger.warning("systemic RCA returned fallback but target invalid: %r", target)
            return {"action": "continue"}
        if action == "accept_finding":
            self.enter_diagnostic_budget_mode(
                state,
                sys_rca.get("summary", "systemic RCA accept_finding"))
            return {"action": "accept_finding"}
        return {"action": "continue"}

    def run(self, dispatch_fn=None, monitor_fn=None, rca_fn=None,
            systemic_rca_fn=None):
        """Execute the dispatch loop.

        dispatch_fn: callable(phase_key, handoff_path) -> verdict_dict
            Spawns phase agent. If None (shadow mode), simulates PASS.
        monitor_fn: callable(phase_key, result_path, summary_path, checks) -> verdict_dict
            Spawns monitor agent. If None, pass-through is only allowed in
            shadow/test paths (dispatch_fn is None or runner.shadow=True).
        rca_fn: callable(phase_key, manifest_dict) -> rca_dict
            Spawns RCA/analysis agent. In non-shadow runs with dispatch_fn
            wired, this callback must be present whenever the resolved
            phase list includes critical phases with non-null rca_artifact.
        systemic_rca_fn: callable(phase_key, manifest_dict) -> systemic_rca_dict
            Spawns the systemic RCA agent when two consecutive per-phase RCAs
            share a fingerprint. If None, the orchestrator logs a warning and
            falls back to the normal per-phase retry path.
        """
        mode = self.resolve_mode()
        phase_list, dep_overrides = self.resolve_phase_list(mode)

        # Check for escalation resume
        progress_path = os.path.join(self.output_dir, "progress.json")
        if self.v2_monitor and os.path.isfile(progress_path):
            progress = load_json(progress_path)
            if progress.get("status") == "escalation_pending":
                state = RunnerState.from_progress(self.output_dir, progress)
                should_resume, resume_idx, resume_reruns = self._resume_from_escalation(
                    state, phase_list)
                if should_resume:
                    # Resume dispatch from the escalated phase
                    phase_idx = resume_idx
                    state.write_progress()
                else:
                    if state.status == "failed":
                        state.write_parity_manifest()
                        return state
                    # No response yet -- return state as-is
                    return state
            else:
                state = RunnerState(self.output_dir, mode, phase_list)
                phase_idx = 0
        else:
            state = RunnerState(self.output_dir, mode, phase_list)
            phase_idx = 0

        # Auto-derive sticky scalars from filesystem (e.g. phase_split_inputs_ready)
        # so callers never have to set them manually.
        derived_sticky = self.derive_sticky_inputs()
        if derived_sticky:
            sticky_path = os.path.join(self.output_dir, "monitor", "derived-sticky.json")
            atomic_write_json(sticky_path, derived_sticky)

        state.write_progress()

        if dispatch_fn is not None and monitor_fn is None and not self.shadow:
            message = (
                "monitor_fn not wired for non-shadow run. "
                "Independent monitor reviews are required; dispatch-verdict "
                "pass-through is only allowed in shadow/test paths."
            )
            logger.error(message)
            self.write_runner_failure("monitor_error", message)
            state.status = "failed"
            state.write_progress()
            state.write_parity_manifest()
            return state

        rca_required_phases = self.phases_requiring_rca(phase_list)
        if dispatch_fn is not None and rca_fn is None and not self.shadow and rca_required_phases:
            message = (
                "rca_fn not wired for non-shadow run. "
                "Critical phases require independent RCA callbacks: "
                f"{', '.join(rca_required_phases)}. "
                "RCA pass-through is only allowed in shadow/test paths."
            )
            logger.error(message)
            self.write_runner_failure("rca_error", message)
            state.status = "failed"
            state.write_progress()
            state.write_parity_manifest()
            return state

        while phase_idx < len(phase_list):
            phase_key = phase_list[phase_idx]
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
            advance_phase = True  # set False on redirect to suppress phase_idx += 1
            pending_retry_feedback = None
            while True:
                attempt = phase_reruns + 1
                context, context_meta = self.resolve_context_with_meta(phase_key)
                feedback = pending_retry_feedback
                if phase_reruns > 0 and not feedback:
                    feedback = f"Attempt {attempt} after {phase_reruns} prior failure(s)."

                handoff = self.build_handoff(
                    phase_key,
                    context,
                    attempt,
                    prior_feedback=feedback,
                    dep_overrides=dep_overrides,
                    context_meta=context_meta,
                )
                handoff_path = self.write_handoff(phase_key, handoff)
                state.last_changed_handoff_hash[phase_key] = self._hash_handoff(handoff)

                if dispatch_fn:
                    dispatch_verdict = dispatch_fn(phase_key, handoff_path)
                else:
                    dispatch_verdict = {"verdict": "PASS", "attempt": attempt}

                result_path = os.path.join(
                    self.output_dir, "agent-results",
                    f"phase-{self.phases[phase_key]['index']:02d}-result.md",
                )
                if monitor_fn:
                    summary_path = os.path.join(self.output_dir, "monitor", "running-summary.md")
                    phase_meta = self.phases[phase_key]
                    checks = phase_meta.get("quality", {}).get("checks", []) if phase_meta.get("critical") else []
                    verdict = monitor_fn(phase_key, result_path, summary_path, checks)
                else:
                    verdict = dispatch_verdict

                normalized_verdict = self.normalize_monitor_verdict(
                    verdict.get("verdict", "PASS")
                )
                state.verdict_sequence.append({
                    "phase": phase_key,
                    "attempt": attempt,
                    "verdict": normalized_verdict,
                })

                v = normalized_verdict

                if self.v2_monitor:
                    # --- V2 path ---
                    v2_verdict = self._evaluate_v2(
                        phase_key, verdict, v, result_path=result_path)
                    v = self.normalize_monitor_verdict(v2_verdict)
                    phase_meta = self.phases[phase_key]

                    if v == "PASS":
                        state.phases_completed.append(phase_key)
                        state.retry_counts[phase_key] = phase_reruns
                        break

                    # FAIL path: RCA first
                    rca_result = self._invoke_phase_rca(
                        phase_key,
                        phase_meta,
                        verdict,
                        "FAIL",
                        rca_fn,
                        state,
                    )

                    # Auto-systemic-RCA: record fingerprint, dispatch systemic
                    # RCA when the previous attempt's fingerprint matches.
                    current_fp = self._record_rca_fingerprint(
                        state, phase_key, attempt, rca_result)
                    sys_decision = self._maybe_dispatch_systemic_rca(
                        state, phase_key, attempt, current_fp,
                        systemic_rca_fn, phase_list)
                    if sys_decision is not None:
                        if sys_decision["action"] == "accept_finding":
                            # Systemic RCA accepted the finding. Pause
                            # for explicit user instruction instead of
                            # auto-skipping to report-generate. The user
                            # can request a partial report through the
                            # standard user-decision options.
                            state.systemic_rca_triggered[phase_key] = True
                            review_path = os.path.join(
                                self.output_dir, "monitor",
                                f"phase-{self.phases[phase_key]['index']:02d}-review.md",
                            )
                            self.enter_awaiting_user_instruction(
                                state,
                                phase_key=phase_key,
                                reason="systemic_rca_accept_finding",
                                rca_result=rca_result,
                                review_path=review_path,
                            )
                            state.write_parity_manifest()
                            return state
                        if sys_decision["action"] == "fallback":
                            target = sys_decision["target"]
                            already_used = any(
                                f["phase_key"] == phase_key and f["fallback_target"] == target
                                for f in state.fallbacks_used
                            )
                            if not already_used:
                                state.fallbacks_used.append({
                                    "phase_key": phase_key,
                                    "fallback_target": target,
                                })
                            phase_idx = sys_decision["target_idx"]
                            state.phases_completed = state.phases_completed[:phase_idx]
                            state.systemic_rca_triggered[phase_key] = True
                            advance_phase = False
                            break
                        # action == "continue" -> fall through to normal retry path

                    escalation_result = self._maybe_escalate(phase_key, v, verdict)
                    response = self._determine_response(
                        v, phase_key, phase_meta, state, rca_result,
                        escalation_result, phase_reruns, phase_list)
                    phase_reruns = response.get("phase_reruns", phase_reruns)

                    if response["action"] == "continue":
                        state.phases_completed.append(phase_key)
                        state.retry_counts[phase_key] = phase_reruns
                        break
                    elif response["action"] == "retry":
                        pending_retry_feedback = self.build_retry_feedback(
                            attempt, verdict, rca_result)
                        continue
                    elif response["action"] == "redirect":
                        phase_idx = response.get("target_idx", phase_list.index(response["target"]))
                        state.phases_completed = state.phases_completed[:phase_idx]
                        advance_phase = False
                        break  # outer while re-dispatches from target
                    elif response["action"] == "escalation_pending":
                        state.status = "escalation_pending"
                        state.escalation_context = response.get("context")
                        state.write_progress()
                        return state
                    else:  # abort
                        # Critical-phase aborts (RCA stop_with_blocker,
                        # budget exhausted with no fallback, etc.) pause
                        # for explicit user instruction. Non-critical
                        # phase aborts still hard-fail because they
                        # cannot be retried meaningfully.
                        review_path = os.path.join(
                            self.output_dir, "monitor",
                            f"phase-{self.phases[phase_key]['index']:02d}-review.md",
                        )
                        if phase_meta.get("critical"):
                            self.enter_awaiting_user_instruction(
                                state,
                                phase_key=phase_key,
                                reason=response.get("reason", "abort"),
                                rca_result=rca_result,
                                review_path=review_path,
                            )
                            state.write_parity_manifest()
                            return state
                        self.write_runner_failure(
                            response.get("error_type", "budget_exhausted"),
                            response.get("message", f"Aborted at {phase_key}"),
                            phase_key)
                        state.status = "failed"
                        state.write_progress()
                        state.write_parity_manifest()
                        return state
                else:
                    # --- V1 path (unchanged) ---
                    v = self.evaluate_structured_predicates(
                        phase_key, verdict, v, result_path=result_path, v2=False)
                    if v == "PASS":
                        state.phases_completed.append(phase_key)
                        state.retry_counts[phase_key] = phase_reruns
                        break

                    # FAIL path: RCA first (does not increment counters)
                    phase_meta = self.phases[phase_key]
                    rca_result = self._invoke_phase_rca(
                        phase_key,
                        phase_meta,
                        verdict,
                        "FAIL",
                        rca_fn,
                        state,
                    )

                    # Auto-systemic-RCA dispatch (V1 path mirrors V2).
                    current_fp = self._record_rca_fingerprint(
                        state, phase_key, attempt, rca_result)
                    sys_decision = self._maybe_dispatch_systemic_rca(
                        state, phase_key, attempt, current_fp,
                        systemic_rca_fn, phase_list)
                    if sys_decision is not None:
                        if sys_decision["action"] == "accept_finding":
                            # V1: systemic RCA accepted the finding.
                            # Pause for user instruction; do not auto-
                            # skip to a partial report.
                            state.systemic_rca_triggered[phase_key] = True
                            review_path = os.path.join(
                                self.output_dir, "monitor",
                                f"phase-{self.phases[phase_key]['index']:02d}-review.md",
                            )
                            self.enter_awaiting_user_instruction(
                                state,
                                phase_key=phase_key,
                                reason="systemic_rca_accept_finding",
                                rca_result=rca_result,
                                review_path=review_path,
                            )
                            state.write_parity_manifest()
                            return state
                        if sys_decision["action"] == "fallback":
                            target = sys_decision["target"]
                            already_used = any(
                                f["phase_key"] == phase_key and f["fallback_target"] == target
                                for f in state.fallbacks_used
                            )
                            if not already_used:
                                state.fallbacks_used.append({
                                    "phase_key": phase_key,
                                    "fallback_target": target,
                                })
                            phase_idx = sys_decision["target_idx"]
                            state.phases_completed = state.phases_completed[:phase_idx]
                            state.systemic_rca_triggered[phase_key] = True
                            advance_phase = False
                            break
                        # continue -> fall through

                    state.total_reruns += 1
                    phase_reruns += 1

                    review_path = os.path.join(
                        self.output_dir, "monitor",
                        f"phase-{self.phases[phase_key]['index']:02d}-review.md",
                    )

                    # Budget caps are bypassed only in extended (user-set) budget_mode.
                    if (self.budget_caps_enforced(state)
                            and self.rerun_budgets_exhausted(state, phase_reruns)):
                        ft = phase_meta.get("fallback_target")
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
                            phase_idx = fb_idx
                            phase_reruns = 0
                            advance_phase = False
                            break
                        else:
                            # Budget exhausted with no fallback. Pause for
                            # explicit user instruction; never auto-skip
                            # to a partial report.
                            self.enter_awaiting_user_instruction(
                                state,
                                phase_key=phase_key,
                                reason="budget_exhausted",
                                rca_result=rca_result,
                                review_path=review_path,
                            )
                            state.write_parity_manifest()
                            return state

                    # RCA terminal_action check
                    if rca_result and rca_result.get("terminal_action") == "stop_with_blocker":
                        # RCA explicitly recommends stop. Pause for user
                        # instruction instead of auto-skipping to a
                        # partial report. ``allow_partial_report`` no
                        # longer triggers any auto-skip path.
                        self.enter_awaiting_user_instruction(
                            state,
                            phase_key=phase_key,
                            reason="rca_stop_with_blocker",
                            rca_result=rca_result,
                            review_path=review_path,
                        )
                        state.write_parity_manifest()
                        return state

                    pending_retry_feedback = self.build_retry_feedback(
                        attempt, verdict, rca_result)

            state.write_progress()
            state.write_parity_manifest()
            if advance_phase:
                phase_idx += 1

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
    args = parser.parse_args()

    config = load_json(args.config)
    registry = load_json(args.registry)
    runner = DeterministicRunner(config, registry, args.output_dir)
    # CLI mode is always shadow (no real dispatch/monitor/rca) — callbacks
    # are provided by the LLM orchestrator layer, not this entry point.
    state = runner.run()

    print(f"Runner finished: status={state.status}, "
          f"phases={len(state.phases_completed)}, "
          f"reruns={state.total_reruns}")
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
