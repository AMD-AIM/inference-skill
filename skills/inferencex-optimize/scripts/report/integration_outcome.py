"""Single source of truth for E2E integration health constants and gating logic.

Imported by validate_optimization.py, generate_optimization_summary.py, and
tests/e2e_optimize_test.py so the runtime contract and tests cannot drift.
"""

SCHEMA_VERSION = "1.0"

SPEEDUP_PASS_THRESHOLD = 1.0
SPEEDUP_WARN_THRESHOLD = 0.97
SEVERE_TTFT_REGRESSION_PCT = 20.0


def performance_gate(speedup, ttft_regression_pct=None):
    """Return the tri-state gate string and whether TTFT caused an upgrade.

    Returns (gate, ttft_upgraded):
        gate            -- "pass" | "warn" | "fail"
        ttft_upgraded   -- True when the gate was raised from warn to fail
                           because of a severe TTFT regression
    """
    if speedup is None:
        return "fail", False

    if speedup >= SPEEDUP_PASS_THRESHOLD:
        base = "pass"
    elif speedup >= SPEEDUP_WARN_THRESHOLD:
        base = "warn"
    else:
        base = "fail"

    ttft_upgraded = False
    if (
        base == "warn"
        and ttft_regression_pct is not None
        and ttft_regression_pct > SEVERE_TTFT_REGRESSION_PCT
    ):
        base = "fail"
        ttft_upgraded = True

    return base, ttft_upgraded


def derive_fields(speedup, artifacts_valid, ttft_regression_pct=None):
    """Compute every gating field from raw inputs.

    Returns a dict with: performance_gate, performance_valid, validated,
    ttft_upgraded.
    """
    gate, ttft_upgraded = performance_gate(speedup, ttft_regression_pct)
    perf_valid = gate == "pass"
    return {
        "performance_gate": gate,
        "performance_valid": perf_valid,
        "validated": bool(artifacts_valid and perf_valid),
        "ttft_upgraded": ttft_upgraded,
    }


def pipeline_status(blocker_list, integration_gate=None, *, integration_expected=False):
    """Derive pipeline_status from blockers *and* integration health.

    blocker_list         -- list of blocker dicts (may be empty)
    integration_gate     -- "pass" | "warn" | "fail" | None (no integration ran)
    integration_expected -- True when a results dir was provided, meaning
                            integration *should* have produced a comparison file.
                            When True and integration_gate is None the pipeline
                            is treated as incomplete rather than clean.
    """
    early_phases = {"benchmark", "profile-analyze"}

    if blocker_list:
        has_early = any(b.get("phase") in early_phases for b in blocker_list)
        if has_early:
            return "pipeline incomplete"
        return "completed with blockers"

    if integration_gate == "fail":
        return "completed with blockers"
    if integration_gate == "warn":
        return "completed with warnings"
    if integration_gate is None and integration_expected:
        return "pipeline incomplete"
    return "completed"
