#!/usr/bin/env python3
"""Structured predicate evaluator for phase verdicts.

Evaluates typed predicates from detection_rules_structured against phase
context data (Key Findings scalars and artifact fields). Replaces prose-only
detection_rules for mechanical verdict determination.

Usage:
    from predicate_engine import evaluate_predicates

    rules = [
        {"field": "performance_gate", "op": "eq", "value": "fail", "verdict": "FAIL"},
        {"field": "e2e_speedup", "op": "lt", "value": 0.97, "verdict": "FAIL"},
    ]
    verdict, details = evaluate_predicates(rules, context_data)
"""

from typing import Any


OPERATORS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: _num(a) < _num(b),
    "le": lambda a, b: _num(a) <= _num(b),
    "gt": lambda a, b: _num(a) > _num(b),
    "ge": lambda a, b: _num(a) >= _num(b),
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "exists": lambda a, b: a is not None,
    "not_exists": lambda a, b: a is None,
}


def _num(v):
    """Coerce to float for comparison, pass through if already numeric."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _check_condition(condition_str: str, context: dict) -> bool:
    """Evaluate a simple condition string like 'performance_gate == warn'."""
    parts = condition_str.strip().split()
    if len(parts) != 3:
        return True
    field, op_str, value = parts
    actual = context.get(field)
    if actual is None:
        return False
    if op_str == "==":
        return str(actual) == value
    if op_str == "!=":
        return str(actual) != value
    return True


def evaluate_predicates(
    rules: list[dict],
    context: dict[str, Any],
) -> tuple[str, list[dict]]:
    """Evaluate structured predicates against context data.

    Returns (verdict, details):
        verdict  -- "PASS" | "WARN" | "FAIL"
        details  -- list of per-rule evaluation results
    """
    details = []
    worst_verdict = "PASS"
    verdict_rank = {"PASS": 0, "WARN": 1, "FAIL": 2}

    for rule in rules:
        field = rule["field"]
        op = rule["op"]
        expected = rule.get("value")
        rule_verdict = rule.get("verdict", "FAIL")
        condition = rule.get("condition")

        actual = context.get(field)

        if condition and not _check_condition(condition, context):
            details.append({
                "field": field, "op": op, "expected": expected, "actual": actual,
                "triggered": False, "reason": f"condition not met: {condition}",
            })
            continue

        if op not in OPERATORS:
            details.append({
                "field": field, "op": op, "expected": expected, "actual": actual,
                "triggered": False, "reason": f"unknown operator: {op}",
            })
            continue

        if actual is None and op not in ("exists", "not_exists"):
            details.append({
                "field": field, "op": op, "expected": expected, "actual": None,
                "triggered": False, "reason": "field not present in context",
            })
            continue

        triggered = OPERATORS[op](actual, expected)
        detail = {
            "field": field, "op": op, "expected": expected, "actual": actual,
            "triggered": triggered, "verdict": rule_verdict if triggered else "PASS",
        }
        details.append(detail)

        if triggered and verdict_rank.get(rule_verdict, 0) > verdict_rank.get(worst_verdict, 0):
            worst_verdict = rule_verdict

    return worst_verdict, details
