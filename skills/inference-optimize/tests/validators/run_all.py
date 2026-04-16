#!/usr/bin/env python3
"""Unified validator runner. Imports all per-phase validators and produces
test_report.json with unified output.

Usage:
    python3 run_all.py --output-dir <path>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validators import collect_validators


def main():
    parser = argparse.ArgumentParser(description="Run all phase validators")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    validators = collect_validators()
    all_results = []
    summary = {"total": 0, "passed": 0, "failed": 0}

    for phase_key, validate_fn in sorted(validators.items()):
        results = validate_fn(args.output_dir)
        for r in results:
            all_results.append({
                "phase": r.phase,
                "name": r.name,
                "passed": r.passed,
                "detail": r.detail,
                "severity": r.severity,
            })
            summary["total"] += 1
            if r.passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1

    report = {"summary": summary, "results": all_results}
    report_path = os.path.join(args.output_dir, "test_report.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validators: {summary['passed']}/{summary['total']} passed, "
          f"{summary['failed']} failed")
    print(f"Report: {report_path}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
