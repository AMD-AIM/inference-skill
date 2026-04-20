"""Verdict-to-action response policy engine.

Maps (verdict, context) -> response action using explicit priority rules.
Evaluation order is first-match, highest priority wins.

Priority order:
1. Safety stop: RCA stop_with_blocker -> abort (non-overridable)
2. Human override: escalation response -> follow human's choice
3. Budget constraint: active finite caps exhausted -> redirect/abort
4. RCA recommendation: retry or fallback per RCA
5. Default: retry if retries are uncapped or budget remains
"""
import logging

logger = logging.getLogger(__name__)

MAX_HUMAN_EXTENSIONS = 3  # Default cap per phase


def _limit_exceeded(current_value, limit_value):
    """Positive limits are enforced; non-positive values mean uncapped retries."""
    return limit_value is not None and limit_value > 0 and current_value > limit_value


def determine_response(verdict, failure_type, phase_key, phase_meta,
                        runner_state, rca_result=None, escalation_result=None,
                        rerun_limits=None, phase_reruns=0, phase_list=None):
    """Map (verdict, context) -> response action.

    Args:
        verdict: "PASS", "WARN", or "FAIL"
        failure_type: Category string (e.g. "infrastructure", "logic")
        phase_key: Phase identifier
        phase_meta: Phase metadata from registry (fallback_target, terminal_policy, etc.)
        runner_state: RunnerState object with budget tracking
        rca_result: RCA analysis result dict or None
        escalation_result: Human response dict or None
        rerun_limits: {"max_per_phase": int, "max_total": int} or None.
            Non-positive limits mean uncapped retries.
        phase_reruns: Current phase retry count
        phase_list: Ordered list of phase keys (for partial report skip)

    Returns:
        {"action": "continue"|"retry"|"redirect"|"abort",
         "target": str|None, "reason": str}
    """
    if rerun_limits is None:
        rerun_limits = {}
    max_per_phase = rerun_limits.get("max_per_phase", 0)
    max_total = rerun_limits.get("max_total", 0)

    # Non-FAIL verdicts
    if verdict == "PASS":
        return {"action": "continue", "target": None, "reason": "Phase passed"}

    if verdict == "WARN":
        return {"action": "continue", "target": None, "reason": "Phase passed with warnings"}

    # --- FAIL verdict: evaluate priority rules ---

    # 1. SAFETY STOP: RCA stop_with_blocker (non-overridable)
    if rca_result and rca_result.get("terminal_action") == "stop_with_blocker":
        return {"action": "abort", "target": None,
                "reason": f"RCA safety stop: {rca_result.get('analysis', 'blocker detected')}"}

    # 2. HUMAN OVERRIDE
    if escalation_result and escalation_result.get("action") not in (None, "escalation_pending"):
        human_action = escalation_result.get("action", "abort")

        if human_action == "retry":
            if _limit_exceeded(phase_reruns, max_per_phase):
                extensions = getattr(runner_state, 'human_extensions', {})
                ext_count = extensions.get(phase_key, 0)
                max_ext = getattr(runner_state, 'max_human_extensions', MAX_HUMAN_EXTENSIONS)

                if ext_count >= max_ext:
                    return {"action": "abort", "target": None,
                            "reason": f"Human extension cap ({max_ext}) exceeded for {phase_key}"}

                logger.warning(f"Human override: extending budget by 1 for {phase_key} (extension {ext_count + 1}/{max_ext})")
                return {"action": "retry", "target": None,
                        "reason": f"Human override: budget extended ({ext_count + 1}/{max_ext})"}

            return {"action": "retry", "target": None,
                    "reason": f"Human override: retry (notes: {escalation_result.get('notes', '')})"}

        elif human_action == "fallback":
            fallback = phase_meta.get("fallback_target")
            if fallback:
                return {"action": "redirect", "target": fallback,
                        "reason": f"Human override: fallback to {fallback}"}
            return {"action": "abort", "target": None,
                    "reason": "Human requested fallback but no fallback target available"}

        elif human_action == "abort":
            return {"action": "abort", "target": None,
                    "reason": f"Human override: abort (notes: {escalation_result.get('notes', '')})"}

        elif human_action == "manual_fix":
            return {"action": "retry", "target": None,
                    "reason": "Human override: manual fix applied, retrying"}

    # 3. BUDGET CONSTRAINT
    if (_limit_exceeded(phase_reruns, max_per_phase)
            or _limit_exceeded(runner_state.total_reruns, max_total)):
        fallback = phase_meta.get("fallback_target")
        if fallback:
            return {"action": "redirect", "target": fallback,
                    "reason": f"Budget exhausted, falling back to {fallback}"}

        allow_partial = phase_meta.get("terminal_policy") == "allow_partial_report"
        if allow_partial:
            return {"action": "continue", "target": None,
                    "reason": "Budget exhausted, allowing partial report"}

        return {"action": "abort", "target": None,
                "reason": "budget_exhausted",
                "message": f"Budget exhausted for {phase_key}"}

    # 4. RCA RECOMMENDATION
    if rca_result:
        rca_action = rca_result.get("terminal_action", "retry")
        if rca_action == "retry":
            return {"action": "retry", "target": None,
                    "reason": f"RCA recommends retry: {rca_result.get('analysis', '')}"}
        elif rca_action == "fallback":
            fallback = phase_meta.get("fallback_target")
            if fallback:
                return {"action": "redirect", "target": fallback,
                        "reason": f"RCA recommends fallback to {fallback}"}

    # 5. DEFAULT: retry if retries are uncapped or budget remains
    return {"action": "retry", "target": None,
            "reason": "Default: retrying (budget available or uncapped)"}
