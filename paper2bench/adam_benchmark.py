"""Installed finite-horizon Appendix A check for Reddi et al. (ICLR 2018).

The candidate uses moment recurrences. This controller oracle instead evaluates
the geometric sum for the prescribed cyclic gradients in closed form. Neither
candidate-reported losses nor candidate-selected parameters are trusted.
This bounded numerical experiment is not an asymptotic convergence proof.
"""
from __future__ import annotations

import math

from .bundles import BundleError, canonical_json, digest

SCORER = "adam_counterexample/v1"
METRIC = "adam_vs_amsgrad_average_regret_gap"
UNITS = "loss_per_round"
TRACE_TOLERANCE = 1e-10
METRIC_TOLERANCE = 1e-10
PAPER_SHA256 = "75a949b0751e81c490e7eee76efadd1eb8868c64f44275d8d1542a8f42cb7c01"
INPUTS = {
    "schema_version": 1,
    "C": 3,
    "beta1": 0.0,
    "beta2": 0.1,
    "alpha": 0.1,
    "epsilon": 0.0,
    "bias_correction": False,
    "initial_position": 1.0,
    "domain": [-1.0, 1.0],
    "steps": 30000,
    "recording": "pre_update",
}
CHECKPOINTS = (3, 30, 300, 3000, 30000)


def _number(value):
    try:
        valid = type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise BundleError("Adam benchmark requires finite numeric values")
    return value


def validate_inputs(inputs):
    """Accept only the installed, source-reviewed finite experiment condition."""
    if not isinstance(inputs, dict) or canonical_json(inputs) != canonical_json(INPUTS):
        raise BundleError("Adam benchmark inputs differ from the installed Appendix A condition")


def _second_moment(t):
    # v_t = (1-b) sum_j b**(t-j) g_j**2. For g_(3k+1)=C and
    # all other g_j=-1, split off the unit-gradient baseline and sum spikes.
    b, c = INPUTS["beta2"], INPUTS["C"]
    phase = (t - 1) % 3
    spikes = (t - 1) // 3 + 1
    spike_sum = b ** phase * (1.0 - b ** (3 * spikes)) / (1.0 - b ** 3)
    return 1.0 - b ** t + (c * c - 1.0) * (1.0 - b) * spike_sum


def oracle_trajectories(inputs):
    """Compute a controller reference without the candidate's EMA recurrence."""
    validate_inputs(inputs)
    positions = {"adam": [], "amsgrad": []}
    x = {"adam": 1.0, "amsgrad": 1.0}
    for t in range(1, inputs["steps"] + 1):
        gradient = 3.0 if t % 3 == 1 else -1.0
        v = _second_moment(t)
        # The largest second moment is at the most recent C-gradient round.
        # Those peaks increase monotonically from the zero initialization.
        latest_spike = t - (t - 1) % 3
        denominators = {"adam": math.sqrt(v), "amsgrad": math.sqrt(_second_moment(latest_spike))}
        for algorithm in ("adam", "amsgrad"):
            positions[algorithm].append(x[algorithm])
            proposed = x[algorithm] - inputs["alpha"] * gradient / math.sqrt(t) / denominators[algorithm]
            x[algorithm] = min(1.0, max(-1.0, proposed))
    return {"schema_version": 1, "recording": "pre_update", **{
        algorithm: {"positions": positions[algorithm], "final_position": x[algorithm]}
        for algorithm in positions}}


def _average_regret(positions, horizon):
    # Horizon is a multiple of 3 and C>2, so the best fixed comparator is -1.
    return math.fsum((3.0 if i % 3 == 0 else -1.0) * (positions[i] + 1.0)
                     for i in range(horizon)) / horizon


def trajectory_statistics(trajectories):
    """Derive every loss from raw pre-update positions, never claimed metrics."""
    checkpoints = []
    for horizon in CHECKPOINTS:
        adam = _average_regret(trajectories["adam"]["positions"], horizon)
        amsgrad = _average_regret(trajectories["amsgrad"]["positions"], horizon)
        checkpoints.append({"horizon": horizon, "adam_average_regret": adam,
                            "amsgrad_average_regret": amsgrad, "gap": adam - amsgrad})
    return checkpoints


def build_reference(inputs):
    """Freeze this reference before dispatching any candidate execution."""
    expected = trajectory_statistics(oracle_trajectories(inputs))[-1]
    return {"protocol": "adam_appendix_a/v1", "inputs_sha256": digest(canonical_json(inputs)),
            "expected_gap": expected["gap"], "tolerance": METRIC_TOLERANCE,
            "trace_tolerance": TRACE_TOLERANCE}


def score_adam(inputs, predictions, reference):
    validate_inputs(inputs)
    if not isinstance(reference, dict) or set(reference) != {
            "protocol", "inputs_sha256", "expected_gap", "tolerance", "trace_tolerance"}:
        raise BundleError("Invalid Adam benchmark reference contract")
    if (reference["protocol"] != "adam_appendix_a/v1"
            or reference["inputs_sha256"] != digest(canonical_json(inputs))
            or _number(reference["tolerance"]) != METRIC_TOLERANCE
            or _number(reference["trace_tolerance"]) != TRACE_TOLERANCE):
        raise BundleError("Adam benchmark reference differs from the installed condition")
    expected_gap = _number(reference["expected_gap"])
    if (not isinstance(predictions, dict)
            or set(predictions) != {"schema_version", "recording", "adam", "amsgrad"}
            or type(predictions["schema_version"]) is not int or predictions["schema_version"] != 1
            or predictions["recording"] != "pre_update"):
        raise BundleError("Adam benchmark requires complete pre-update trajectories")
    for algorithm in ("adam", "amsgrad"):
        trace = predictions[algorithm]
        if (not isinstance(trace, dict) or set(trace) != {"positions", "final_position"}
                or not isinstance(trace["positions"], list)
                or len(trace["positions"]) != inputs["steps"]):
            raise BundleError("Adam benchmark trajectory dimensions disagree")
        for position in trace["positions"] + [trace["final_position"]]:
            if not -1.0 <= _number(position) <= 1.0:
                raise BundleError("Adam benchmark position is outside the projected domain")
    expected = oracle_trajectories(inputs)
    oracle_gap = trajectory_statistics(expected)[-1]["gap"]
    if abs(expected_gap - oracle_gap) > 1e-14:
        raise BundleError("Frozen Adam benchmark reference disagrees with the installed oracle")
    max_error = max(
        max(abs(actual - target) for actual, target in zip(
            predictions[algorithm]["positions"] + [predictions[algorithm]["final_position"]],
            expected[algorithm]["positions"] + [expected[algorithm]["final_position"]]))
        for algorithm in ("adam", "amsgrad"))
    checkpoints = trajectory_statistics(predictions)
    measured = checkpoints[-1]
    valid = max_error <= TRACE_TOLERANCE
    passed = valid and abs(measured["gap"] - expected_gap) <= METRIC_TOLERANCE
    return {"verdict": "pass" if passed else "fail", "metric": METRIC,
            "value": measured["gap"], "units": UNITS, "samples": inputs["steps"],
            "authority": "trusted_recomputation", "observation_valid": valid,
            "failure_reason": None if valid else "incorrect_algorithm_trace",
            "tolerance": METRIC_TOLERANCE, "trace_tolerance": TRACE_TOLERANCE,
            "max_trace_absolute_error": max_error,
            "adam_average_regret": measured["adam_average_regret"],
            "amsgrad_average_regret": measured["amsgrad_average_regret"],
            "adam_position_at_T": predictions["adam"]["positions"][-1],
            "amsgrad_position_at_T": predictions["amsgrad"]["positions"][-1],
            "adam_position_after_T": predictions["adam"]["final_position"],
            "amsgrad_position_after_T": predictions["amsgrad"]["final_position"],
            "checkpoints": checkpoints,
            "scope": "finite Appendix A numerical check; no asymptotic or neural-benchmark claim"}
