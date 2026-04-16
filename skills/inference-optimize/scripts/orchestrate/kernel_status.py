"""Structured per-kernel status tracking via append-only JSONL.

Replaces sentinel files with a single append-only JSONL file that is
corruption-tolerant (truncated last line is skipped) and supports
redirect-aware replay via run_attempt field.
"""
import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def append_kernel_event(jsonl_path, kernel_id, event_type, run_attempt, **kwargs):
    """Append a kernel status event atomically.

    Write protocol: open O_WRONLY|O_APPEND|O_CREAT, single os.write, os.fsync.

    Args:
        jsonl_path: Path to kernel-status.jsonl
        kernel_id: Kernel identifier
        event_type: One of: started, completed, failed, skipped
        run_attempt: Current run attempt number (from state.total_reruns)
        **kwargs: Additional fields (speedup, error, duration_seconds, etc.)
    """
    event = {
        "kernel_id": kernel_id,
        "event_type": event_type,
        "run_attempt": run_attempt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    line = json.dumps(event, separators=(",", ":")) + "\n"
    encoded = line.encode("utf-8")

    fd = os.open(jsonl_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def replay_kernel_status(jsonl_path):
    """Replay events to build per-kernel state.

    Each event has {kernel_id, event_type, run_attempt, ...}.
    Last-writer-wins: for each (kernel_id, event_type),
    the event with highest run_attempt wins.

    Truncated last line is silently skipped.

    Returns:
        dict mapping kernel_id -> {event_type -> event_dict}
    """
    state = {}

    if not os.path.isfile(jsonl_path):
        return state

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Truncated last line -- skip silently
                logger.debug(f"Skipping malformed JSONL line {line_num}")
                continue

            kid = event.get("kernel_id")
            etype = event.get("event_type")
            attempt = event.get("run_attempt", 0)

            if kid is None or etype is None:
                continue

            if kid not in state:
                state[kid] = {}

            existing = state[kid].get(etype)
            if existing is None or attempt >= existing.get("run_attempt", 0):
                state[kid][etype] = event

    return state


def get_kernel_summary(jsonl_path):
    """Get a summary of all kernel statuses.

    Returns:
        dict with keys: total, completed, failed, skipped,
                       best_speedup, kernel_details
    """
    state = replay_kernel_status(jsonl_path)

    summary = {
        "total": len(state),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "best_speedup": 0.0,
        "kernel_details": {},
    }

    for kid, events in state.items():
        # Determine final status from latest event type
        if "completed" in events:
            summary["completed"] += 1
            speedup = events["completed"].get("speedup", 0.0)
            if speedup > summary["best_speedup"]:
                summary["best_speedup"] = speedup
            summary["kernel_details"][kid] = {"status": "completed", "speedup": speedup}
        elif "failed" in events:
            summary["failed"] += 1
            summary["kernel_details"][kid] = {"status": "failed", "error": events["failed"].get("error", "")}
        elif "skipped" in events:
            summary["skipped"] += 1
            summary["kernel_details"][kid] = {"status": "skipped"}
        elif "started" in events:
            summary["kernel_details"][kid] = {"status": "in_progress"}

    return summary
