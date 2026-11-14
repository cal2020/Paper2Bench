"""Trusted, bounded example scorers independent of candidate implementations.

These are computational demos, not validated paper-reproduction protocols.
Never load a scorer or Python plugin named by a candidate task.
"""
import itertools
import math

from .bundles import BundleError

# Installed names remain compatible with alpha curator bundles. The separate
# immutable protocol identities distinguish corrected numerical verdicts from
# historical reports, and are pinned by the workflow before dispatch.
SCORER_PROTOCOLS = {
    "linear_predictions/v1": "linear_predictions/v2",
    "knapsack_optimum/v1": "knapsack_optimum/v2",
}
SCORERS = set(SCORER_PROTOCOLS)


def number(value):
    try:
        finite = type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise BundleError("Expected a finite numeric value")
    return value


def _exact_objectives(values):
    """Use integer arithmetic for exact sums of JSON integers/binary floats.

    All finite binary-float denominators are powers of two, so the largest
    denominator is a bounded common scale. Float rounding or relative tolerance
    must not make a demonstrably suboptimal discrete selection pass.
    """
    ratios = [number(value).as_integer_ratio() for value in values]
    scale = max(denominator for _, denominator in ratios)
    return [numerator * (scale // denominator) for numerator, denominator in ratios], scale


def _reported_objective(value, scale):
    try:
        result = value // scale if value % scale == 0 else value / scale
    except OverflowError as error:
        raise BundleError("Metric overflow") from error
    return number(result)


def score(scorer, inputs, predictions, reference):
    if scorer not in SCORERS:
        raise BundleError("Scorer is not an installed trusted scorer")
    if not isinstance(inputs, dict) or not isinstance(predictions, dict) or not isinstance(reference, dict):
        raise BundleError("Scoring inputs must be typed objects")
    if scorer == "linear_predictions/v1":
        actual, labels = predictions.get("predictions"), reference.get("labels")
        tests = inputs.get("test")
        if (not isinstance(actual, list) or not isinstance(labels, list) or not isinstance(tests, list)
                or not 1 <= len(labels) <= 10000 or len(actual) != len(labels) or len(tests) != len(labels)):
            raise BundleError("Prediction and hidden-label dimensions disagree")
        tolerance = number(reference.get("max_mse", 1e-9))
        if tolerance < 0:
            raise BundleError("Metric tolerance cannot be negative")
        # Compare the exact values represented by JSON integers/binary floats.
        # Subtracting a float from a nearby large integer can round an actual
        # residual to zero, and squaring tiny residuals can underflow to zero.
        scaled, scale = _exact_objectives(actual + labels)
        count = len(labels)
        squared_error = sum((a - b) ** 2 for a, b in zip(scaled[:count], scaled[count:]))
        normalizer = count * scale * scale
        try:
            mse = squared_error / normalizer
        except OverflowError as error:
            raise BundleError("Metric overflow") from error
        if not math.isfinite(mse):
            raise BundleError("Metric overflow")
        if squared_error and mse == 0:
            raise BundleError("Metric underflow")
        tolerance_numerator, tolerance_denominator = tolerance.as_integer_ratio()
        passed = squared_error * tolerance_denominator <= tolerance_numerator * normalizer
        return {"verdict": "pass" if passed else "fail", "metric": "mean_squared_error",
                "value": mse, "units": "target_units_squared", "tolerance": tolerance,
                "samples": len(labels), "authority": "trusted_recomputation"}
    weights, values, capacity = inputs.get("weights"), inputs.get("values"), inputs.get("capacity")
    selected = predictions.get("selected")
    if (not isinstance(weights, list) or not isinstance(values, list) or len(weights) != len(values)
            or not 1 <= len(weights) <= 20 or type(capacity) is not int or not 0 <= capacity <= 100000):
        raise BundleError("Invalid bounded knapsack instance")
    if any(type(w) is not int or w <= 0 for w in weights):
        raise BundleError("Knapsack weights must be positive integers")
    objective_values, objective_scale = _exact_objectives(values)
    if (not isinstance(selected, list) or len(selected) > len(weights)
            or any(type(i) is not int or not 0 <= i < len(weights) for i in selected)
            or len(set(selected)) != len(selected)):
        raise BundleError("Invalid selected item indices")
    # Independent exhaustive oracle; candidate example uses dynamic programming.
    optimum = 0
    for bits in itertools.product((False, True), repeat=len(weights)):
        if sum(w for w, on in zip(weights, bits) if on) <= capacity:
            optimum = max(optimum, sum(v for v, on in zip(objective_values, bits) if on))
    weight = sum(weights[i] for i in selected)
    objective = sum(objective_values[i] for i in selected)
    value = _reported_objective(objective, objective_scale)
    optimum_value = _reported_objective(optimum, objective_scale)
    feasible = weight <= capacity
    return {"verdict": "pass" if feasible and objective == optimum else "fail",
            "metric": "feasible_optimal_objective", "value": value, "optimum": optimum_value,
            "feasible": feasible, "weight": weight, "units": "objective_units",
            "authority": "trusted_exhaustive_check"}
