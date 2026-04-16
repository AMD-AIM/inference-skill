#!/usr/bin/env python3
"""Validate handoff documents before dispatching phase agents.

Catches malformed handoffs before retry budget is burned. The runner calls
this before each phase dispatch; the legacy orchestrator can call it too.

Usage:
    python3 validate_handoff.py --handoff-path <path> --phase <phase_key> --phase-index <N>
"""

import argparse
import json
import os
import re
import sys

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

REQUIRED_FRONTMATTER = {"phase", "phase_index"}
REQUIRED_SECTIONS = {"## Context", "## Instructions"}

RERUN_REQUIRED_SECTIONS = {"## Prior Attempt Feedback"}


def parse_frontmatter(content):
    """Extract YAML-like frontmatter as a dict of key: value pairs."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return None, ["Missing YAML frontmatter (expected --- ... --- block)"]

    errors = []
    fields = {}
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            errors.append(f"Malformed frontmatter line: {line!r}")
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"').strip("'")

    return fields, errors


def validate_handoff(path, expected_phase, expected_index, is_rerun=False):
    """Validate a handoff document. Returns (valid, errors)."""
    errors = []

    if not os.path.isfile(path):
        return False, [f"Handoff file not found: {path}"]

    with open(path) as f:
        content = f.read()

    if not content.strip():
        return False, ["Handoff file is empty"]

    fields, fm_errors = parse_frontmatter(content)
    errors.extend(fm_errors)

    if fields is None:
        return False, errors

    for req in REQUIRED_FRONTMATTER:
        if req not in fields:
            errors.append(f"Missing required frontmatter field: {req}")

    if "phase" in fields and fields["phase"] != expected_phase:
        errors.append(
            f"Frontmatter phase={fields['phase']!r} does not match "
            f"expected={expected_phase!r}"
        )

    if "phase_index" in fields:
        try:
            idx = int(fields["phase_index"])
            if idx != expected_index:
                errors.append(
                    f"Frontmatter phase_index={idx} does not match "
                    f"expected={expected_index}"
                )
        except ValueError:
            errors.append(f"phase_index is not an integer: {fields['phase_index']!r}")

    for section in REQUIRED_SECTIONS:
        if section not in content:
            errors.append(f"Missing required section: {section}")

    if is_rerun:
        for section in RERUN_REQUIRED_SECTIONS:
            if section not in content:
                errors.append(f"Rerun handoff missing required section: {section}")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate a handoff document")
    parser.add_argument("--handoff-path", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--phase-index", type=int, required=True)
    parser.add_argument("--is-rerun", action="store_true", default=False)
    args = parser.parse_args()

    valid, errors = validate_handoff(
        args.handoff_path, args.phase, args.phase_index, args.is_rerun,
    )

    if valid:
        print("HANDOFF VALID")
        return 0

    print("HANDOFF INVALID:")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
