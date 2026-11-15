"""Trusted numerical evaluation, independent of candidate execution.

The Wilson interval is a binomial sampling interval over test examples. It
assumes independent examples and does not capture training-seed variability,
cluster dependence, or implementation/protocol fidelity.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

from .models import (
    ClaimSpec,
    Dataset,
    KnapsackDataset,
    KnapsackInstance,
    OptimizationClaim,
    OptimizationPredictions,
    Predictions,
    parse_dataset,
    parse_predictions,
)


def evaluate(
    dataset: Dataset | KnapsackDataset,
    predictions: Predictions | OptimizationPredictions,
    claim: ClaimSpec | OptimizationClaim,
) -> dict[str, Any]:
    """Dispatch to a trusted evaluator with a task-compatible metric and output."""
    dataset = parse_dataset(dataset)
    predictions = parse_predictions(predictions, dataset)
    if isinstance(dataset, KnapsackDataset):
        if not isinstance(claim, OptimizationClaim):
            raise ValueError("knapsack requires an optimality_ratio claim")
        claim = OptimizationClaim.model_validate(claim.model_dump(mode="python"))
        return _evaluate_knapsack(dataset, predictions, claim)
    if not isinstance(claim, ClaimSpec):
        raise ValueError("classification requires an accuracy claim")
    claim = ClaimSpec.model_validate(claim.model_dump(mode="python"))
    return _evaluate_classification(dataset, predictions, claim)


def _evaluate_classification(
    dataset: Dataset, predictions: Predictions, claim: ClaimSpec
) -> dict[str, Any]:
    """Assess accuracy agreement with a declared target, without reproduction claims.

    Test IDs must match exactly. Prediction order is arbitrary; per-item results
    follow dataset order. Labels must belong to the training label vocabulary.
    """
    expected_ids = {example.id for example in dataset.test}
    if set(predictions.ids) != expected_ids:
        raise ValueError("prediction IDs must match test IDs exactly")
    known_labels = {example.y for example in dataset.train}
    if any(label not in known_labels for label in predictions.labels):
        raise ValueError("predictions contain labels outside the training label vocabulary")

    by_id = dict(zip(predictions.ids, predictions.labels, strict=True))
    per_item = [
        {
            "id": example.id,
            "expected": example.y,
            "predicted": by_id[example.id],
            "correct": by_id[example.id] == example.y,
        }
        for example in dataset.test
    ]
    count = len(per_item)
    correct = sum(item["correct"] for item in per_item)
    accuracy = correct / count

    # Using the lower tail avoids rounding (1 + confidence) / 2 to exactly 1.
    z = -NormalDist().inv_cdf((1.0 - claim.confidence) / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / count
    center = (accuracy + z_squared / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(accuracy * (1.0 - accuracy) / count + z_squared / (4.0 * count * count))
        / denominator
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    # Exact boundary cases avoid tiny cancellation residues such as 1e-17.
    if correct == 0:
        lower = 0.0
    if correct == count:
        upper = 1.0

    band_lower = max(0.0, claim.target - claim.tolerance)
    band_upper = min(1.0, claim.target + claim.tolerance)
    minimum_met = count >= claim.min_test_examples
    verdict = "inconclusive"
    if minimum_met:
        if lower >= band_lower and upper <= band_upper:
            verdict = "supported"
        elif upper < band_lower or lower > band_upper:
            verdict = "not_supported"

    return {
        "verdict": verdict,
        "criterion": "accuracy_equivalence",
        "metric": "accuracy",
        "accuracy": accuracy,
        "correct": correct,
        "sample_count": count,
        "confidence": claim.confidence,
        "confidence_interval": {"lower": lower, "upper": upper, "method": "wilson"},
        "target": claim.target,
        "tolerance": claim.tolerance,
        "criterion_band": {"lower": band_lower, "upper": band_upper},
        "min_test_examples": claim.min_test_examples,
        "minimum_sample_count_met": minimum_met,
        "implementation_fidelity": "unverified",
        "protocol_fidelity": "unverified",
        "interpretation": (
            "Numerical accuracy agreement only; this result does not establish that a paper "
            "was reproduced or that a method caused an improvement. The Wilson interval "
            "assumes independent test examples and excludes training-seed variability."
        ),
        "per_item": per_item,
    }


def _knapsack_optimum(instance: KnapsackInstance) -> int:
    """Exact value-only 0-1 DP, independently recomputed from instance inputs.

    Descending capacities prevent reuse of an item. Dataset validation bounds
    total DP work to 20 million states and capacity memory to 10,001 entries.
    No candidate solver, objective, or claimed optimum enters this calculation.
    """
    best = [0] * (instance.capacity + 1)
    for weight, value in zip(instance.weights, instance.values, strict=True):
        for capacity in range(instance.capacity, weight - 1, -1):
            best[capacity] = max(best[capacity], best[capacity - weight] + value)
    return best[instance.capacity]


def _evaluate_knapsack(
    dataset: KnapsackDataset, predictions: OptimizationPredictions, claim: OptimizationClaim
) -> dict[str, Any]:
    expected_ids = {instance.id for instance in dataset.instances}
    if set(predictions.ids) != expected_ids:
        raise ValueError("prediction IDs must match knapsack instance IDs exactly")
    by_id = dict(zip(predictions.ids, predictions.selections, strict=True))
    per_item = []
    for instance in dataset.instances:
        selection = by_id[instance.id]
        optimum = _knapsack_optimum(instance)
        total_weight = objective = None
        reason = None
        if any(index < 0 or index >= len(instance.weights) for index in selection):
            reason = "item_index_out_of_range"
        elif len(set(selection)) != len(selection):
            reason = "duplicate_item_index"
        else:
            total_weight = sum(instance.weights[index] for index in selection)
            objective = sum(instance.values[index] for index in selection)
            if total_weight > instance.capacity:
                reason = "capacity_exceeded"
        feasible = reason is None
        score = (objective / optimum if optimum else 1.0) if feasible else 0.0
        per_item.append(
            {
                "id": instance.id,
                "selection": list(selection),
                "feasible": feasible,
                "infeasibility_reason": reason,
                "total_weight": total_weight,
                "objective": objective,
                "optimal_objective": optimum,
                "score": score,
            }
        )

    count = len(per_item)
    mean_score = math.fsum(item["score"] for item in per_item) / count
    # Two-sided Hoeffding concentration for independent scores bounded in [0,1].
    # log1p avoids loss of precision near confidence=1 and overflow in 2/alpha.
    radius = math.sqrt((math.log(2) - math.log1p(-claim.confidence)) / (2 * count))
    lower = max(0.0, mean_score - radius)
    upper = min(1.0, mean_score + radius)
    band_lower = max(0.0, claim.target - claim.tolerance)
    band_upper = min(1.0, claim.target + claim.tolerance)
    minimum_met = count >= claim.min_test_examples
    verdict = "inconclusive"
    if minimum_met:
        if lower >= band_lower and upper <= band_upper:
            verdict = "supported"
        elif upper < band_lower or lower > band_upper:
            verdict = "not_supported"
    return {
        "verdict": verdict,
        "criterion": "optimality_equivalence",
        "metric": "optimality_ratio",
        "mean_score": mean_score,
        "sample_count": count,
        "feasible_count": sum(item["feasible"] for item in per_item),
        "confidence": claim.confidence,
        "confidence_interval": {"lower": lower, "upper": upper, "method": "hoeffding"},
        "target": claim.target,
        "tolerance": claim.tolerance,
        "criterion_band": {"lower": band_lower, "upper": band_upper},
        "min_test_examples": claim.min_test_examples,
        "minimum_sample_count_met": minimum_met,
        "implementation_fidelity": "unverified",
        "protocol_fidelity": "unverified",
        "interpretation": (
            "Numerical mean optimality agreement only; this result does not establish paper "
            "reproduction or implementation fidelity. Scores use independently recomputed "
            "exact 0-1 knapsack optima. The Hoeffding interval assumes independent problem "
            "instances with scores bounded in [0,1] and excludes solver-seed variability."
        ),
        "per_item": per_item,
    }
