"""Human-in-the-loop escalation via signal-and-resume pattern.

runner.py cannot call AskUserQuestion directly (it's real Python).
Instead, it writes an escalation request file and returns with
status="escalation_pending". Claude reads the request, presents to user,
writes a response file, and re-invokes run().

Staged timeout: warn at ESCALATION_WARN_SECONDS, abort at ESCALATION_ABORT_SECONDS.
"""
import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ESCALATION_WARN_SECONDS = 300      # 5 min -- first notification
ESCALATION_ABORT_SECONDS = 900     # 15 min -- abort if no response
ESCALATION_DEFAULT_ACTION = "abort"
MAX_HUMAN_EXTENSIONS = 3           # Max budget extensions per phase


def build_escalation_context(phase_key, verdict, failure_type, layer1_details=None,
                              monitor_review_path=None, rca_summary=None,
                              running_summary_path=None, kernel_status_path=None,
                              suggested_actions=None):
    """Build payload for rapid human comprehension.

    All references to logs/reports are file paths -- human reads the files.

    Returns:
        dict with escalation context.
    """
    context = {
        "phase": phase_key,
        "verdict": verdict,
        "failure_type": failure_type,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "suggested_actions": suggested_actions or ["retry", "fallback", "abort"],
        "timeline": {
            "warn_seconds": ESCALATION_WARN_SECONDS,
            "abort_seconds": ESCALATION_ABORT_SECONDS,
        },
    }
    if layer1_details is not None:
        context["predicate_results"] = layer1_details
    if monitor_review_path:
        context["monitor_review_path"] = monitor_review_path
    if rca_summary:
        context["rca_summary"] = rca_summary
    if running_summary_path:
        context["running_summary_path"] = running_summary_path
    if kernel_status_path:
        context["kernel_status_path"] = kernel_status_path

    return context


def write_escalation_request(output_dir, phase_key, context):
    """Write escalation request file atomically.

    Returns:
        Path to the written request file.
    """
    monitor_dir = os.path.join(output_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    phase_num = context.get("phase_index", phase_key)
    path = os.path.join(monitor_dir, f"escalation-request-phase-{phase_key}.json")

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(context, f, indent=2)
    os.replace(tmp_path, path)

    logger.info(f"Escalation request written: {path}")
    return path


def read_escalation_response(output_dir, phase_key):
    """Read escalation response file if it exists.

    Returns:
        dict with {"action": "retry"|"fallback"|"abort"|"manual_fix", "notes": "..."}
        or None if no response file exists.
    """
    path = os.path.join(output_dir, "monitor", f"escalation-response-phase-{phase_key}.json")
    if not os.path.isfile(path):
        return None

    with open(path) as f:
        return json.load(f)


def is_escalation_stale(context, abort_seconds=None):
    """Check if an escalation request has exceeded its timeout.

    Args:
        context: Escalation context dict with 'requested_at' field.
        abort_seconds: Override for ESCALATION_ABORT_SECONDS.

    Returns:
        True if the escalation is stale and should be auto-aborted.
    """
    if abort_seconds is None:
        abort_seconds = ESCALATION_ABORT_SECONDS

    requested_at = context.get("requested_at")
    if not requested_at:
        return False

    try:
        req_time = datetime.fromisoformat(requested_at)
        now = datetime.now(timezone.utc)
        elapsed = (now - req_time).total_seconds()
        return elapsed > abort_seconds
    except (ValueError, TypeError):
        return False
