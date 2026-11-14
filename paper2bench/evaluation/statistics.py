"""Bounded descriptive summaries and explicit finite-sample mean bounds.

The two-sided Hoeffding interval requires independent observations whose common
range was fixed before data collection. It quantifies the stated variability
source, not uncertainty over an unobserved population of papers or subjects.
Reference: Bartlett, CS281B lecture 13, theorem 2.2:
https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/13.pdf
"""
from __future__ import annotations

import math
import statistics as std_statistics
from fractions import Fraction

from .validation import EvaluationError


def finite_number(value, label="value"):
    try:
        finite = type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise EvaluationError(f"{label} must be a finite number")
    return value


def _reported_number(value):
    """Render a rational without rounding integers or hiding nonzero underflow."""
    try:
        reported = value.numerator if value.denominator == 1 else float(value)
        finite_number(reported, "summary arithmetic")
    except (OverflowError, ValueError) as error:
        raise EvaluationError("Summary arithmetic overflow") from error
    if value and reported == 0:
        raise EvaluationError("Summary arithmetic underflow")
    return reported


def _exact_mean(values):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 10000:
        raise EvaluationError("Between one and 10000 observations are required")
    numbers = [finite_number(value) for value in values]
    return sum((Fraction(value) for value in numbers), Fraction()) / len(numbers)


def paired_difference(value, baseline):
    """Avoid rounding a paired effect into agreement before aggregation.

    Evidence records use finite JSON numbers. Preserve exact integer results;
    reject a nonintegral result that cannot be represented by that contract.
    """
    exact = Fraction(finite_number(value)) - Fraction(finite_number(baseline))
    reported = _reported_number(exact)
    if Fraction(reported) != exact:
        raise EvaluationError("Paired difference cannot be represented without precision loss")
    return reported


def summarize(values, *, bounds=None, confidence=0.95, independence_assumption=False):
    """Describe raw run values; add a bound only under declared assumptions."""
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 10000:
        raise EvaluationError("Between one and 10000 observations are required")
    numbers = [finite_number(value) for value in values]
    if type(independence_assumption) is not bool:
        raise EvaluationError("independence_assumption must be a boolean")
    confidence = finite_number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise EvaluationError("confidence must lie strictly between zero and one")
    try:
        mean = _reported_number(_exact_mean(numbers))
        sd = std_statistics.stdev(numbers) if len(numbers) >= 2 else None
    except EvaluationError:
        raise
    except (OverflowError, ValueError) as error:
        raise EvaluationError("Observation summary overflow") from error
    if not math.isfinite(mean) or (sd is not None and not math.isfinite(sd)):
        raise EvaluationError("Observation summary overflow")
    report = {"n_runs": len(numbers), "values": numbers, "mean": mean,
              "sample_standard_deviation": sd,
              "standard_error": sd / math.sqrt(len(numbers)) if sd is not None else None,
              "confidence_interval": None,
              "uncertainty_status": "not_assessed_without_prespecified_bounds_and_independence"}
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise EvaluationError("Bounds must contain two finite numbers")
        lower, upper = (finite_number(x, "bound") for x in bounds)
        if lower >= upper or not math.isfinite(upper - lower):
            raise EvaluationError("Bounds must define a finite nonzero range")
        if any(not lower <= value <= upper for value in numbers):
            raise EvaluationError("Observed value lies outside prespecified bounds")
        report["bounds"] = [lower, upper]
        if independence_assumption:
            radius = (upper - lower) * math.sqrt(math.log(2 / (1 - confidence)) / (2 * len(numbers)))
            report["confidence_interval"] = [max(lower, mean - radius), min(upper, mean + radius)]
            report["confidence_level"] = confidence
            report["interval_method"] = "two_sided_hoeffding"
            report["uncertainty_status"] = "conditional_on_prespecified_bounds_and_independence"
    return report


def compare_summary(summary, *, rule, reference, tolerance=None):
    """Separate point agreement from interval-based equivalence."""
    if not isinstance(summary, dict) or not isinstance(rule, str):
        raise EvaluationError("Comparison requires a summary object and named rule")
    reference = finite_number(reference, "reference")
    reported_mean = finite_number(summary.get("mean"), "mean")
    # The display mean can round a nonintegral rational. Decisions use the
    # original run values whenever present, rather than that rounded display.
    mean = _exact_mean(summary["values"]) if "values" in summary else Fraction(reported_mean)
    if _reported_number(mean) != reported_mean:
        raise EvaluationError("Summary mean does not match its observations")
    reference_exact = Fraction(reference)
    difference = _reported_number(mean - reference_exact)
    if rule in {"absolute_tolerance", "relative_tolerance", "equivalence_interval"}:
        tolerance = finite_number(tolerance, "tolerance")
        if tolerance < 0:
            raise EvaluationError("Tolerance must be nonnegative")
    if rule == "absolute_tolerance":
        passed = abs(mean - reference_exact) <= Fraction(tolerance)
        interpretation = "point_agreement_with_declared_tolerance"
    elif rule == "relative_tolerance":
        if reference == 0:
            raise EvaluationError("Relative tolerance is undefined for a zero reference")
        passed = abs(mean - reference_exact) <= abs(reference_exact) * Fraction(tolerance)
        interpretation = "point_agreement_with_declared_relative_tolerance"
    elif rule == "at_least":
        passed, interpretation = mean >= reference_exact, "point_threshold"
    elif rule == "at_most":
        passed, interpretation = mean <= reference_exact, "point_threshold"
    elif rule == "equivalence_interval":
        interval = summary.get("confidence_interval")
        if interval is None:
            return {"verdict": "inconclusive", "difference": difference,
                    "reason": "equivalence_requires_valid_prespecified_uncertainty_interval"}
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise EvaluationError("Confidence interval must contain two finite endpoints")
        low, high = (finite_number(value, "interval endpoint") for value in interval)
        if low > high or not Fraction(low) <= mean <= Fraction(high):
            raise EvaluationError("Confidence interval must be ordered and contain its mean")
        if ("values" not in summary or "bounds" not in summary or "confidence_level" not in summary
                or summary.get("interval_method") != "two_sided_hoeffding"
                or summary.get("uncertainty_status") != "conditional_on_prespecified_bounds_and_independence"):
            raise EvaluationError("Equivalence requires a validated prespecified uncertainty summary")
        expected = summarize(summary["values"], bounds=summary["bounds"],
                             confidence=summary["confidence_level"], independence_assumption=True)
        if list(interval) != expected["confidence_interval"]:
            raise EvaluationError("Confidence interval does not match its observations and assumptions")
        margin_low = reference_exact - Fraction(tolerance)
        margin_high = reference_exact + Fraction(tolerance)
        passed = margin_low <= Fraction(low) and Fraction(high) <= margin_high
        # A bound overlapping the margin establishes neither equivalence nor
        # non-equivalence. Only a disjoint interval establishes disagreement.
        if not passed and not (Fraction(high) < margin_low or Fraction(low) > margin_high):
            return {"verdict": "inconclusive", "difference": difference,
                    "reason": "interval_overlaps_equivalence_margin"}
        interpretation = "interval_containment_in_prespecified_equivalence_margin"
    else:
        raise EvaluationError("Unsupported scientific comparison rule")
    return {"verdict": "pass" if passed else "fail", "difference": difference,
            "reference": reference, "rule": rule, "tolerance": tolerance,
            "interpretation": interpretation}
