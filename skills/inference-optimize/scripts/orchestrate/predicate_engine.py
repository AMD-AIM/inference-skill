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


# Legacy WARN is treated as FAIL to enforce hard-fail monitor semantics.
VERDICT_RANK = {"PASS": 0, "FAIL": 1, "WARN": 1}

# Valid problem categories for the 9-category failure taxonomy
VALID_CATEGORIES = {
    "infrastructure", "logic", "data_quality", "performance_regression",
    "effort_waste", "cross_kernel_interference", "geak_false_claim",
    "baseline_drift", "stale_artifact",
}


def _normalize_verdict(verdict: str) -> str:
    """Normalize legacy WARN verdicts to hard FAIL."""
    if verdict == "WARN":
        return "FAIL"
    return verdict


def _resolve_value(value, thresholds):
    """Resolve value-level $ref in rule values.

    If value is a dict with {"$ref": "$NAME"}, resolve from thresholds dict.
    Otherwise return as-is. Composes with all existing operators.
    """
    if isinstance(value, dict) and "$ref" in value:
        ref_name = value["$ref"]
        # Strip leading $ if present
        key = ref_name.lstrip("$")
        if thresholds and key in thresholds:
            return thresholds[key]
        return None  # Unresolved ref
    return value


def evaluate_predicates(
    rules: list[dict],
    context: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> tuple[str, list[dict]]:
    """Evaluate structured predicates against context data.

    Returns (verdict, details):
        verdict  -- "PASS" | "FAIL" (legacy WARN is normalized to FAIL)
        details  -- list of per-rule evaluation results
    """
    details = []
    worst_verdict = "PASS"

    for rule in rules:
        field = rule["field"]
        op = rule["op"]
        raw_expected = rule.get("value")
        expected = _resolve_value(raw_expected, thresholds) if thresholds is not None else raw_expected
        rule_verdict = _normalize_verdict(rule.get("verdict", "FAIL"))
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

        if expected is None and op not in ("exists", "not_exists"):
            details.append({
                "field": field, "op": op, "expected": None, "actual": actual,
                "triggered": False, "reason": "unresolved $ref threshold",
            })
            continue

        triggered = OPERATORS[op](actual, expected)
        detail = {
            "field": field, "op": op, "expected": expected, "actual": actual,
            "triggered": triggered, "verdict": rule_verdict if triggered else "PASS",
        }
        details.append(detail)

        if triggered and VERDICT_RANK.get(rule_verdict, 0) > VERDICT_RANK.get(worst_verdict, 0):
            worst_verdict = rule_verdict

    return worst_verdict, details


def evaluate_predicates_v2(
    rules: list[dict],
    context: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> tuple[str, list[dict], list[str]]:
    """V2 predicate evaluation with problem category enrichment.

    Calls evaluate_predicates internally, then enriches details with
    problem_category from the optional 'category' field on rules.

    Returns (verdict, details, problem_categories):
        verdict           -- "PASS" | "FAIL" (legacy WARN is normalized to FAIL)
        details           -- list of per-rule evaluation results with category
        problem_categories -- deduplicated list of triggered categories
    """
    verdict, details = evaluate_predicates(rules, context, thresholds)

    # Enrich details with problem categories
    triggered_categories = []
    for i, detail in enumerate(details):
        if i < len(rules):
            category = rules[i].get("category")
            if category:
                detail["category"] = category
                if detail.get("triggered"):
                    triggered_categories.append(category)

    # Deduplicate while preserving order
    seen = set()
    unique_categories = []
    for cat in triggered_categories:
        if cat not in seen:
            seen.add(cat)
            unique_categories.append(cat)

    return verdict, details, unique_categories
