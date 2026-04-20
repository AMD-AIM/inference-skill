#!/usr/bin/env python3
"""Contract tests for monitor-related intake guidance."""

import pathlib

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
INTAKE_PATH = SKILL_ROOT / "INTAKE.md"
SKILL_PATH = SKILL_ROOT / "SKILL.md"


def _read(path: pathlib.Path) -> str:
    return path.read_text()


class TestMonitorIntakeContract:
    """Keep monitor intake requirements explicit across docs."""

    def test_round1_includes_monitor_workflow(self):
        intake = _read(INTAKE_PATH)
        assert "`Monitor workflow`" in intake
        assert "mode=monitor" in intake

    def test_strict_monitoring_is_fixed_for_monitor_modes(self):
        intake = _read(INTAKE_PATH)
        required_snippets = [
            "If mode is `optimize`, `optimize-only`, or `monitor`, set `MONITOR_LEVEL=strict` automatically before final confirmation.",
            "If mode is `optimize`, `optimize-only`, or `monitor`, set:",
            "- `MONITOR_LEVEL=strict`",
            "This setting is mandatory for skill-guided runs.",
        ]
        for snippet in required_snippets:
            assert snippet in intake, f"Missing contract snippet in INTAKE.md: {snippet}"

    def test_monitor_followup_disallows_monitor_level_prompt(self):
        intake = _read(INTAKE_PATH)
        assert "Do not ask the user to choose between standard/strict/minimal for monitor workflow." in intake

    def test_final_confirmation_summary_mentions_fixed_monitoring_policy(self):
        intake = _read(INTAKE_PATH)
        assert "- monitoring policy: strict (fixed for `optimize`, `optimize-only`, and `monitor`)" in intake

    def test_skill_entrypoint_mentions_fixed_strict_monitoring(self):
        skill = _read(SKILL_PATH)
        assert "always set `MONITOR_LEVEL=strict` during intake before confirmation (no monitor-level question)." in skill
