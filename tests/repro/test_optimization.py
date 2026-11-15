"""A true optimization task with trusted feasibility and exact objective checks."""

import itertools
import json
import math
import random

import pytest
from pydantic import ValidationError

from paper2bench.repro.evaluation import evaluate
from paper2bench.repro.models import (
    ClaimSpec,
    Dataset,
    ExperimentSpec,
    KnapsackDataset,
    KnapsackInstance,
    OptimizationClaim,
    OptimizationPredictions,
    Predictions,
    WorkloadSpec,
    canonical_bytes,
    parse_dataset,
    parse_predictions,
)


def instance(**updates):
    return KnapsackInstance(
        **{
            "id": "case",
            "weights": (10, 20, 30),
            "values": (60, 100, 120),
            "capacity": 50,
            **updates,
        }
    )


def repeated_case(count=200, selection=(1, 2)):
    dataset = KnapsackDataset(instances=tuple(instance(id=f"case-{i}") for i in range(count)))
    predictions = OptimizationPredictions(
        ids=tuple(row.id for row in dataset.instances),
        selections=(selection,) * count,
    )
    return dataset, predictions


def test_knapsack_json_contract_is_typed_frozen_and_roundtrippable():
    dataset = parse_dataset(
        {
            "task": "knapsack",
            "instances": [{"id": "case", "weights": [1, 2], "values": [3, 4], "capacity": 2}],
        }
    )
    assert isinstance(dataset, KnapsackDataset)
    assert dataset.instances[0].weights == (1, 2)
    with pytest.raises(ValidationError):
        dataset.instances[0].capacity = 3
    assert parse_dataset(json.loads(canonical_bytes(dataset))) == dataset
    assert parse_dataset({"instances": [dataset.instances[0].model_dump(mode="json")]}) == dataset
    parsed = parse_predictions({"ids": ["case"], "selections": [[1]]}, dataset)
    assert isinstance(parsed, OptimizationPredictions)
    assert parsed.selections == ((1,),)
    assert parse_predictions(parsed, "knapsack") == parsed


@pytest.mark.parametrize(
    "updates",
    [
        {"id": " "},
        {"weights": ()},
        {"values": ()},
        {"weights": (1, 2)},
        {"weights": (0, 20, 30)},
        {"weights": (-1, 20, 30)},
        {"weights": (True, 20, 30)},
        {"weights": (1.0, 20, 30)},
        {"values": (-1, 100, 120)},
        {"values": (False, 100, 120)},
        {"values": ("1", 100, 120)},
        {"capacity": -1},
        {"capacity": 10001},
        {"capacity": True},
        {"capacity": 1.0},
        {"capacity": "50"},
        {"weights": (1,) * 101, "values": (1,) * 101},
        {"objective": 1},
    ],
)
def test_knapsack_instances_reject_malformed_or_unbounded_inputs(updates):
    with pytest.raises(ValidationError):
        instance(**updates)


def test_dataset_ids_case_count_and_total_exact_solver_work_are_bounded():
    with pytest.raises(ValidationError):
        KnapsackDataset(instances=())
    with pytest.raises(ValidationError):
        KnapsackDataset(instances=(instance(), instance()))
    with pytest.raises(ValidationError):
        KnapsackDataset(instances=tuple(instance(id=str(i)) for i in range(1001)))
    # 20 * 100 * (9999 + 1) reaches the exact conservative work boundary.
    at_limit = tuple(
        instance(id=str(i), weights=(1,) * 100, values=(1,) * 100, capacity=9999) for i in range(20)
    )
    assert len(KnapsackDataset(instances=at_limit).instances) == 20
    over_limit = tuple(row.model_copy(update={"capacity": 10000}) for row in at_limit)
    with pytest.raises(ValidationError, match="20000000"):
        KnapsackDataset(instances=over_limit)


@pytest.mark.parametrize(
    "values",
    [
        {"ids": [], "selections": []},
        {"ids": ["case"], "selections": []},
        {"ids": ["case", "case"], "selections": [[], []]},
        {"ids": ["case"], "selections": [[True]]},
        {"ids": ["case"], "selections": [[1.0]]},
        {"ids": ["case"], "selections": [["1"]]},
        {"ids": ["case"], "selections": [[1]], "objective": 999999},
    ],
)
def test_optimization_predictions_reject_structural_errors_and_claimed_objectives(values):
    with pytest.raises(ValidationError):
        parse_predictions(values, "knapsack")


def test_trusted_evaluator_computes_optimum_and_actual_objective_independently():
    dataset = KnapsackDataset(instances=(instance(),))
    score = evaluate(
        dataset,
        OptimizationPredictions(ids=("case",), selections=((0, 1),)),
        OptimizationClaim(target=1),
    )
    item = score["per_item"][0]
    assert item["feasible"] is True
    assert item["total_weight"] == 30
    assert item["objective"] == 160
    assert item["optimal_objective"] == 220
    assert item["score"] == pytest.approx(160 / 220)
    assert score["mean_score"] == item["score"]
    assert "accuracy" not in score


@pytest.mark.parametrize(
    "selection,reason",
    [
        ((-1,), "item_index_out_of_range"),
        ((3,), "item_index_out_of_range"),
        ((1, 1), "duplicate_item_index"),
        ((0, 1, 2), "capacity_exceeded"),
    ],
)
def test_infeasible_selections_are_audited_and_score_zero(selection, reason):
    dataset = KnapsackDataset(instances=(instance(),))
    predictions = OptimizationPredictions(ids=("case",), selections=(selection,))
    score = evaluate(dataset, predictions, OptimizationClaim(target=1))
    item = score["per_item"][0]
    assert item["feasible"] is False
    assert item["infeasibility_reason"] == reason
    assert item["score"] == score["mean_score"] == 0
    assert item["optimal_objective"] == 220
    assert score["feasible_count"] == 0


def test_zero_optimum_scores_feasible_empty_solution_one_but_infeasible_zero():
    dataset = KnapsackDataset(
        instances=(
            instance(id="capacity-zero", capacity=0),
            instance(id="zero-values", values=(0, 0, 0)),
            instance(id="infeasible-zero", values=(0, 0, 0), capacity=0),
        )
    )
    predictions = OptimizationPredictions(
        ids=("capacity-zero", "zero-values", "infeasible-zero"), selections=((), (1,), (0,))
    )
    score = evaluate(dataset, predictions, OptimizationClaim(target=1))
    assert [row["optimal_objective"] for row in score["per_item"]] == [0, 0, 0]
    assert [row["score"] for row in score["per_item"]] == [1, 1, 0]
    assert score["mean_score"] == pytest.approx(2 / 3)


def test_zero_one_constraint_prevents_reusing_an_item_in_exact_solver():
    dataset = KnapsackDataset(instances=(instance(weights=(2,), values=(3,), capacity=4),))
    score = evaluate(
        dataset,
        OptimizationPredictions(ids=("case",), selections=((0,),)),
        OptimizationClaim(target=1),
    )
    assert score["per_item"][0]["optimal_objective"] == 3
    assert score["mean_score"] == 1


def test_exact_solver_matches_independent_exhaustive_oracle():
    rng = random.Random(138)
    rows, selections, expected = [], [], []
    for index in range(30):
        row = instance(
            id=str(index),
            weights=tuple(rng.randint(1, 8) for _ in range(6)),
            values=tuple(rng.randint(0, 15) for _ in range(6)),
            capacity=rng.randint(0, 12),
        )
        feasible = []
        for mask in itertools.product((False, True), repeat=6):
            chosen = tuple(i for i, selected in enumerate(mask) if selected)
            if sum(row.weights[i] for i in chosen) <= row.capacity:
                feasible.append((sum(row.values[i] for i in chosen), chosen))
        value, chosen = max(feasible)
        rows.append(row)
        selections.append(chosen)
        expected.append(value)
    dataset = KnapsackDataset(instances=tuple(rows))
    predictions = OptimizationPredictions(
        ids=tuple(row.id for row in rows), selections=tuple(selections)
    )
    result = evaluate(dataset, predictions, OptimizationClaim(target=1))
    assert [row["optimal_objective"] for row in result["per_item"]] == expected
    assert result["mean_score"] == 1


def test_known_hoeffding_interval_and_supported_equivalence():
    dataset, predictions = repeated_case()
    score = evaluate(dataset, predictions, OptimizationClaim(target=1))
    expected_radius = math.sqrt(math.log(40) / 400)
    assert score["confidence_interval"]["lower"] == pytest.approx(1 - expected_radius)
    assert score["confidence_interval"]["upper"] == 1
    assert score["confidence_interval"]["method"] == "hoeffding"
    assert score["verdict"] == "supported"
    assert score["criterion"] == "optimality_equivalence"
    assert score["metric"] == "optimality_ratio"
    assert score["sample_count"] == score["feasible_count"] == 200
    assert score["implementation_fidelity"] == score["protocol_fidelity"] == "unverified"
    assert "independent problem" in score["interpretation"]
    assert "does not establish" in score["interpretation"]


def test_hoeffding_disjoint_and_overlapping_intervals_have_distinct_verdicts():
    dataset, predictions = repeated_case(selection=(0, 1))
    disjoint = evaluate(dataset, predictions, OptimizationClaim(target=0.95, tolerance=0.02))
    overlap = evaluate(dataset, predictions, OptimizationClaim(target=0.8, tolerance=0.05))
    assert disjoint["verdict"] == "not_supported"
    assert overlap["verdict"] == "inconclusive"


@pytest.mark.parametrize("selection", [(1, 2), (-1,)])
def test_minimum_sample_count_overrides_apparent_support_and_refutation(selection):
    dataset, predictions = repeated_case(count=1, selection=selection)
    score = evaluate(dataset, predictions, OptimizationClaim(target=1, tolerance=1))
    assert score["minimum_sample_count_met"] is False
    assert score["verdict"] == "inconclusive"


def test_prediction_order_does_not_change_scoring_and_exact_ids_are_required():
    dataset = KnapsackDataset(instances=(instance(id="a"), instance(id="b")))
    claim = OptimizationClaim(target=1)
    normal = OptimizationPredictions(ids=("a", "b"), selections=((1, 2), (0, 1)))
    reordered = OptimizationPredictions(ids=("b", "a"), selections=((0, 1), (1, 2)))
    assert evaluate(dataset, normal, claim) == evaluate(dataset, reordered, claim)
    for invalid in (
        OptimizationPredictions(ids=("a",), selections=((1, 2),)),
        OptimizationPredictions(ids=("a", "unknown"), selections=((1,), (2,))),
    ):
        with pytest.raises(ValueError, match="IDs"):
            evaluate(dataset, invalid, claim)


def test_extreme_valid_confidence_is_finite_and_interval_clipped():
    dataset, predictions = repeated_case(count=1)
    claim = OptimizationClaim(target=1, confidence=math.nextafter(1, 0))
    score = evaluate(dataset, predictions, claim)
    assert score["confidence_interval"]["lower"] == 0
    assert score["confidence_interval"]["upper"] == 1
    canonical_bytes(score)


def test_claim_union_keeps_target_only_classification_compatibility():
    raw = {
        "title": "Fixture",
        "source_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "evidence": [{"start": 0, "end": 1, "quote": "x"}],
        "claim": {"target": 1},
        "workload": {"kind": "builtin_centroid"},
    }
    assert isinstance(ExperimentSpec.model_validate(raw).claim, ClaimSpec)
    raw["claim"] = {"metric": "optimality_ratio", "target": 1}
    raw["workload"] = {"kind": "builtin_knapsack"}
    assert isinstance(ExperimentSpec.model_validate(raw).claim, OptimizationClaim)
    WorkloadSpec(kind="builtin_knapsack")
    with pytest.raises(ValidationError):
        WorkloadSpec(kind="builtin_knapsack", model="unexpected")


def test_task_parser_preserves_classification_and_rejects_task_mixing():
    classification = parse_dataset(
        {
            "train": [{"id": "train", "x": [0], "y": "a"}],
            "test": [{"id": "test", "x": [1], "y": "a"}],
        }
    )
    assert isinstance(classification, Dataset)
    parsed = parse_predictions({"ids": ["test"], "labels": ["a"]}, classification)
    assert isinstance(parsed, Predictions)
    assert parse_predictions(parsed, "classification") == parsed
    with pytest.raises(ValidationError):
        parse_predictions({"ids": ["test"], "selections": [[0]]}, classification)
    with pytest.raises(ValueError):
        parse_predictions(parsed, "unsupported")
    knapsack = KnapsackDataset(instances=(instance(),))
    with pytest.raises(ValueError, match="optimality_ratio"):
        evaluate(
            knapsack, OptimizationPredictions(ids=("case",), selections=((),)), ClaimSpec(target=1)
        )
    with pytest.raises(ValueError, match="accuracy"):
        evaluate(classification, parsed, OptimizationClaim(target=1))


def test_boundary_parsers_revalidate_models_constructed_without_validation():
    dataset = KnapsackDataset(instances=(instance(),))
    invalid = dataset.model_copy(
        update={"instances": (instance().model_copy(update={"capacity": -1}),)}
    )
    with pytest.raises(ValidationError):
        parse_dataset(invalid)
    predictions = OptimizationPredictions(ids=("case",), selections=((0,),))
    invalid_prediction = predictions.model_copy(update={"selections": ((True,),)})
    with pytest.raises(ValidationError):
        parse_predictions(invalid_prediction, dataset)
