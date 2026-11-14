"""Strict semantic-output validation and honest rubric coverage accounting."""

from __future__ import annotations

import json
import math
import re
from collections import deque
from typing import Iterable, Mapping


class EvaluationError(ValueError):
    """Malformed/unavailable evaluation, distinct from scientific disagreement."""


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError("Duplicate JSON field")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise EvaluationError("Nonfinite JSON number")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise EvaluationError("Nonfinite JSON number")
    return parsed


def parse_json_object(text: str, *, max_bytes: int = 65536) -> dict:
    """Accept one full object, optionally in one complete JSON fence."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > max_bytes:
        raise EvaluationError("Evaluation output is not bounded text")
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text)
    if fenced:
        text = fenced.group(1)
    try:
        result = json.loads(text, object_pairs_hook=_unique_object,
                            parse_constant=_bad_constant, parse_float=_finite_float)
    except (ValueError, TypeError, RecursionError) as error:
        if isinstance(error, EvaluationError):
            raise
        raise EvaluationError("Evaluation output is not one valid JSON object") from error
    if not isinstance(result, dict):
        raise EvaluationError("Evaluation output must be an object")
    return result


def weighted_coverage(criteria: Iterable[Mapping]) -> dict:
    """Include unknowns and gate dependent passes on successful prerequisites.

    ``depends_on`` names other criterion ids and must form an acyclic graph.
    A reported pass with an unmet prerequisite becomes ``not_run`` in coverage;
    it cannot supply positive evidence for a result whose method was not valid.
    The blocked ids are retained so callers can explain that adjustment.
    """
    applicable = passed = unknown = 0.0
    counts = {key: 0 for key in ("pass", "fail", "inconclusive", "not_run", "error", "not_applicable")}
    critical_blocked = False
    records = {}
    for criterion in criteria:
        if not isinstance(criterion, Mapping):
            raise EvaluationError("Criteria must be objects")
        ident = criterion.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in records:
            raise EvaluationError("Criterion ids must be nonempty and unique")
        weight = criterion.get("weight", 1.0)
        try:
            valid_weight = (type(weight) in {int, float} and math.isfinite(weight) and weight > 0)
        except OverflowError:
            valid_weight = False
        if not valid_weight:
            raise EvaluationError("Criterion weight must be positive and finite")
        verdict = criterion.get("verdict")
        if not isinstance(verdict, str) or verdict not in counts:
            raise EvaluationError("Unknown criterion verdict")
        if type(criterion.get("critical", False)) is not bool:
            raise EvaluationError("critical must be a boolean")
        dependencies = criterion.get("depends_on", [])
        if (not isinstance(dependencies, list)
                or any(not isinstance(dep, str) or not dep.strip() for dep in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise EvaluationError("Criterion dependencies must be a unique array of ids")
        if verdict == "not_applicable":
            if not isinstance(criterion.get("reason"), str) or not criterion["reason"].strip():
                raise EvaluationError("Not-applicable criteria require a reason")
        records[ident] = dict(criterion, weight=weight, depends_on=dependencies)

    pending = {}
    dependents = {ident: [] for ident in records}
    for ident, criterion in records.items():
        dependencies = criterion["depends_on"]
        if any(dep not in records for dep in dependencies):
            raise EvaluationError("Criterion dependency does not exist")
        pending[ident] = len(dependencies)
        for dep in dependencies:
            dependents[dep].append(ident)
    ready = deque(ident for ident, degree in pending.items() if degree == 0)
    effective = {}
    blocked = {}
    while ready:
        ident = ready.popleft()
        criterion = records[ident]
        unmet = [dep for dep in criterion["depends_on"] if effective[dep] != "pass"]
        verdict = criterion["verdict"]
        if unmet and verdict != "not_applicable":
            blocked[ident] = unmet
            if verdict == "pass":
                verdict = "not_run"
        effective[ident] = verdict
        for child in dependents[ident]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if len(effective) != len(records):
        raise EvaluationError("Criterion dependencies must be acyclic")

    for ident, criterion in records.items():
        verdict = effective[ident]
        weight = criterion["weight"]
        counts[verdict] += 1
        if verdict == "not_applicable":
            continue
        applicable += weight
        if verdict == "pass":
            passed += weight
        elif verdict != "fail":
            unknown += weight
        if criterion.get("critical") and verdict != "pass":
            critical_blocked = True
        if not all(math.isfinite(value) for value in (applicable, passed, unknown)):
            raise EvaluationError("Aggregate criterion weights exceed finite limits")
    return {
        "applicable_weight": applicable, "passed_weight": passed, "unknown_weight": unknown,
        "lower_bound": passed / applicable if applicable else None,
        "upper_bound": (passed + unknown) / applicable if applicable else None,
        "coverage": (applicable - unknown) / applicable if applicable else None,
        "critical_blocked": critical_blocked,
        "blocked_dependencies": blocked,
        "status": ("ungradable" if not applicable else
                   "inconclusive" if unknown else "fail" if critical_blocked or passed < applicable else "pass"),
        "counts": counts,
    }
