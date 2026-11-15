"""Paired candidate-versus-baseline plans, execution and statistical contracts."""

import json
import math

import pytest
from pydantic import ValidationError

from paper2bench.repro.comparisons import (
    ComparisonSpec,
    InvalidComparison,
    freeze_comparison,
    run_comparison,
)
from paper2bench.repro.demo import create_demo
from paper2bench.repro.engine import Engine
from paper2bench.repro.models import Predictions, canonical_bytes
from paper2bench.repro.runtime import BuiltinRunner, ExecutionFailure, ExecutionResult
from paper2bench.repro.storage import ArtifactNotFound


@pytest.fixture
def engine(tmp_path):
    return Engine(tmp_path / "state")


def make_plan(engine, **updates):
    baseline_hash = create_demo(engine)
    candidate = engine.load_spec(baseline_hash).model_dump(mode="json")
    candidate["title"] = "Synthetic candidate"
    candidate_hash = engine.freeze(candidate)
    return ComparisonSpec(
        candidate_spec_hash=candidate_hash,
        baseline_spec_hash=baseline_hash,
        bootstrap_samples=200,
        **updates,
    )


def mock_predictions(monkeypatch, *, candidate_correct, baseline_correct, reverse=False):
    def run(self, dataset, spec, **kwargs):
        successes = candidate_correct if spec.title == "Synthetic candidate" else baseline_correct
        labels = tuple(
            example.y if index < successes else ("b" if example.y == "a" else "a")
            for index, example in enumerate(dataset.test)
        )
        ids = tuple(example.id for example in dataset.test)
        if reverse and spec.title == "Synthetic candidate":
            ids, labels = ids[::-1], labels[::-1]
        return ExecutionResult(Predictions(ids=ids, labels=labels), {"wall_seconds": 0.001})

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", run)


def resized_plan(engine, plan, count):
    baseline = engine.load_spec(plan.baseline_spec_hash)
    dataset = engine.artifacts.get_json(baseline.dataset_sha256)
    source_rows = dataset["test"]
    dataset["test"] = [
        {**source_rows[index % len(source_rows)], "id": f"resized-{index}"}
        for index in range(count)
    ]
    dataset_hash = engine.register_dataset(dataset)
    updated = {}
    for role in ("candidate", "baseline"):
        field = f"{role}_spec_hash"
        spec = engine.load_spec(getattr(plan, field)).model_dump(mode="json")
        spec["dataset_sha256"] = dataset_hash
        updated[field] = engine.freeze(spec)
    return plan.model_copy(update=updated)


def test_identical_builtin_methods_do_not_show_superiority(engine):
    plan = make_plan(engine)
    plan_hash = freeze_comparison(engine, plan)
    assert engine.runs.list_runs() == []
    result = run_comparison(engine, plan_hash, "identical")
    score = result["assessment"]
    assert score["effect"] == 0
    assert score["verdict"] == "not_supported"
    assert score["sample_count"] == 80
    assert score["minimum_sample_count_met"] is True
    assert score["min_test_examples"] == 20
    assert score["confidence_interval"] == {
        "lower": 0.0,
        "upper": 0.0,
        "method": "percentile_bootstrap",
    }
    assert result["implementation_fidelity"] == result["protocol_fidelity"] == "unverified"
    assert result["plan_hash"] == plan_hash
    stored = engine.artifacts.get_json(result["result_hash"])
    assert stored == {key: value for key, value in result.items() if key != "result_hash"}


@pytest.mark.parametrize(
    "candidate_correct,baseline_correct,effect,verdict",
    [
        (80, 0, 1.0, "supported"),
        (0, 80, -1.0, "not_supported"),
        (40, 40, 0.0, "not_supported"),
    ],
)
def test_known_direction_and_paired_prediction_order(
    engine, monkeypatch, candidate_correct, baseline_correct, effect, verdict
):
    plan_hash = freeze_comparison(engine, make_plan(engine))
    mock_predictions(
        monkeypatch,
        candidate_correct=candidate_correct,
        baseline_correct=baseline_correct,
        reverse=True,
    )
    result = run_comparison(engine, plan_hash, "known-direction")
    assert result["assessment"]["effect"] == effect
    assert result["assessment"]["verdict"] == verdict
    assert result["assessment"]["per_item"][0]["id"] == "test-000"


def test_minimum_effect_is_strict_and_part_of_plan_identity(engine, monkeypatch):
    plan = make_plan(engine, minimum_effect=1)
    plan_hash = freeze_comparison(engine, plan)
    relaxed = plan.model_copy(update={"minimum_effect": 0.0})
    assert freeze_comparison(engine, relaxed) != plan_hash
    mock_predictions(monkeypatch, candidate_correct=80, baseline_correct=0)
    score = run_comparison(engine, plan_hash, "meaningful-effect")["assessment"]
    assert score["effect"] == score["minimum_effect"] == 1
    assert score["verdict"] == "not_supported"


def test_overlap_with_minimum_effect_is_inconclusive_and_bootstrap_is_reproducible(
    engine, monkeypatch
):
    plan = make_plan(engine, minimum_effect=0.25, seed=73)
    plan_hash = freeze_comparison(engine, plan)
    mock_predictions(monkeypatch, candidate_correct=60, baseline_correct=40)
    first = run_comparison(engine, plan_hash, "bootstrap")
    second = run_comparison(engine, plan_hash, "bootstrap")
    assert first == second
    score = first["assessment"]
    assert score["effect"] == 0.25
    assert score["confidence_interval"]["lower"] < 0.25
    assert score["confidence_interval"]["upper"] > 0.25
    assert score["verdict"] == "inconclusive"
    assert score["bootstrap_seed"] == 73
    assert score["bootstrap_samples"] == 200


@pytest.mark.parametrize("candidate_correct,baseline_correct", [(1, 0), (0, 1), (1, 1)])
def test_small_degenerate_sample_is_inconclusive_even_with_extreme_interval(
    engine, monkeypatch, candidate_correct, baseline_correct
):
    plan = resized_plan(engine, make_plan(engine), 1)
    mock_predictions(
        monkeypatch, candidate_correct=candidate_correct, baseline_correct=baseline_correct
    )
    result = run_comparison(engine, freeze_comparison(engine, plan), "tiny-sample")
    score = result["assessment"]
    assert score["sample_count"] == 1
    assert score["min_test_examples"] == 20
    assert score["minimum_sample_count_met"] is False
    assert score["verdict"] == "inconclusive"
    assert score["effect"] == candidate_correct - baseline_correct
    assert score["confidence_interval"]["lower"] == score["effect"]
    assert score["confidence_interval"]["upper"] == score["effect"]


def test_explicit_minimum_sample_count_is_frozen_and_inclusive(engine, monkeypatch):
    plan = make_plan(engine, min_test_examples=80)
    mock_predictions(monkeypatch, candidate_correct=80, baseline_correct=0)
    plan_hash = freeze_comparison(engine, plan)
    sufficient = run_comparison(engine, plan_hash, "minimum-boundary")["assessment"]
    stricter = plan.model_copy(update={"min_test_examples": 81})
    stricter_hash = freeze_comparison(engine, stricter)
    assert stricter_hash != plan_hash
    insufficient = run_comparison(engine, stricter_hash, "minimum-boundary")["assessment"]
    assert sufficient["minimum_sample_count_met"] is True
    assert sufficient["verdict"] == "supported"
    assert insufficient["minimum_sample_count_met"] is False
    assert insufficient["verdict"] == "inconclusive"
    assert insufficient["effect"] == sufficient["effect"] == 1


def test_bootstrap_work_cap_rejected_before_registering_or_executing_runs(engine, monkeypatch):
    plan = resized_plan(engine, make_plan(engine), 1001)
    plan = plan.model_copy(update={"bootstrap_samples": 20000})
    assert 1001 * 20000 > 20_000_000

    def forbidden(*args, **kwargs):
        pytest.fail("over-budget bootstrap plans must fail before workload execution")

    monkeypatch.setattr(engine, "execute", forbidden)
    with pytest.raises(InvalidComparison, match="20000000 paired draws"):
        freeze_comparison(engine, plan)
    # Run preflight repeats the bound check even for a manually registered plan.
    reference = engine.artifacts.put(canonical_bytes(plan))
    with pytest.raises(InvalidComparison, match="20000000 paired draws"):
        run_comparison(engine, reference, "over-budget")
    assert engine.runs.list_runs() == []


def test_both_runs_and_plan_exist_before_either_execution(engine, monkeypatch):
    plan_hash = freeze_comparison(engine, make_plan(engine))
    execute = engine.execute
    observed = []

    def checked_execute(run_id):
        assert engine.artifacts.get(plan_hash)
        rows = engine.runs.list_runs()
        assert len(rows) == 2
        observed.append([row["state"] for row in rows])
        return execute(run_id)

    monkeypatch.setattr(engine, "execute", checked_execute)
    run_comparison(engine, plan_hash, "preregister")
    assert observed[0] == ["queued", "queued"]
    assert len(observed) == 2


def test_repeated_invocation_never_reexecutes_completed_candidate_or_baseline(engine, monkeypatch):
    plan_hash = freeze_comparison(engine, make_plan(engine))
    first = run_comparison(engine, plan_hash, "idempotent")

    def forbidden(*args, **kwargs):
        pytest.fail("idempotent completed role runs must never execute candidates twice")

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", forbidden)
    second = run_comparison(engine, plan_hash, "idempotent")
    assert first == second
    assert len(engine.runs.list_runs()) == 2
    assert all(run["attempt"] == 1 for run in engine.runs.list_runs())


def test_role_keys_remain_distinct_for_identical_spec_hashes(engine):
    spec_hash = create_demo(engine)
    plan_hash = freeze_comparison(
        engine,
        ComparisonSpec(
            candidate_spec_hash=spec_hash, baseline_spec_hash=spec_hash, bootstrap_samples=200
        ),
    )
    result = run_comparison(engine, plan_hash, "same-spec")
    assert result["candidate"]["run_id"] != result["baseline"]["run_id"]
    assert len({run["idempotency_key"] for run in engine.runs.list_runs()}) == 2


def test_failed_role_produces_not_tested_without_fabricated_effect(engine, monkeypatch):
    plan_hash = freeze_comparison(engine, make_plan(engine))
    original = BuiltinRunner.run

    def fail_candidate(self, dataset, spec, **kwargs):
        if spec.title == "Synthetic candidate":
            raise ExecutionFailure("candidate failed")
        return original(self, dataset, spec, **kwargs)

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", fail_candidate)
    result = run_comparison(engine, plan_hash, "failure")
    assert result["candidate"]["state"] == "failed"
    assert result["baseline"]["state"] == "completed"
    assert result["assessment"]["verdict"] == "not_tested"
    assert result["assessment"]["effect"] is None
    assert result["assessment"]["confidence_interval"] is None
    assert result["assessment"]["sample_count"] == 0
    assert result["assessment"]["minimum_sample_count_met"] is False


def test_resources_seeds_and_sources_can_differ_and_are_disclosed(engine):
    plan = make_plan(engine)
    candidate = engine.load_spec(plan.candidate_spec_hash).model_dump(mode="json")
    source = "A different source with the same dataset and evaluator."
    candidate["source_sha256"] = engine.artifacts.put(source.encode())
    candidate["evidence"] = [{"start": 0, "end": len(source), "quote": source}]
    candidate["limits"]["timeout_seconds"] = 5
    candidate["seed"] = 88
    candidate["assumptions"] = ["Candidate-specific runtime conditions"]
    plan = plan.model_copy(update={"candidate_spec_hash": engine.freeze(candidate)})
    result = run_comparison(engine, freeze_comparison(engine, plan), "unequal-resources")
    assert result["conditions"]["equal_resource_limits"] is False
    assert result["conditions"]["equal_seeds"] is False
    assert result["candidate"]["resources"]["limits"]["timeout_seconds"] == 5
    assert result["baseline"]["resources"]["limits"]["timeout_seconds"] == 30
    assert result["candidate"]["assumptions"] == ["Candidate-specific runtime conditions"]


def test_different_datasets_are_rejected_even_if_ids_and_labels_match(engine):
    plan = make_plan(engine)
    candidate = engine.load_spec(plan.candidate_spec_hash).model_dump(mode="json")
    dataset = engine.artifacts.get_json(candidate["dataset_sha256"])
    dataset["test"][0]["x"] = [-9]
    candidate["dataset_sha256"] = engine.register_dataset(dataset)
    altered = plan.model_copy(update={"candidate_spec_hash": engine.freeze(candidate)})
    with pytest.raises(InvalidComparison, match="same dataset"):
        freeze_comparison(engine, altered)
    assert engine.runs.list_runs() == []


def test_different_frozen_evaluators_are_rejected(engine):
    plan = make_plan(engine)
    baseline = engine.load_spec(plan.baseline_spec_hash).model_dump(mode="json")
    baseline["engine_identity_sha256"] = "e" * 64
    altered_baseline = engine.artifacts.put_json(baseline)
    altered_plan = plan.model_copy(update={"baseline_spec_hash": altered_baseline})
    with pytest.raises(InvalidComparison, match="same frozen evaluator"):
        freeze_comparison(engine, altered_plan)


def test_second_role_preflight_failure_does_not_execute_first_role(engine, monkeypatch):
    plan = make_plan(engine)
    baseline = engine.load_spec(plan.baseline_spec_hash).model_dump(mode="json")
    source = "A separate baseline source for a preflight failure fixture."
    reference = engine.artifacts.put(source.encode())
    baseline["source_sha256"] = reference
    baseline["evidence"] = [{"start": 0, "end": len(source), "quote": source}]
    plan = plan.model_copy(update={"baseline_spec_hash": engine.freeze(baseline)})
    plan_hash = freeze_comparison(engine, plan)
    (engine.artifacts.root / reference[:2] / reference[2:]).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("both roles must pass preflight before any workload executes")

    monkeypatch.setattr(engine, "execute", forbidden)
    with pytest.raises(ArtifactNotFound):
        run_comparison(engine, plan_hash, "baseline-preflight-failure")
    assert all(run["state"] == "queued" for run in engine.runs.list_runs())


def test_equal_marginal_accuracy_with_different_errors_retains_paired_uncertainty(
    engine, monkeypatch
):
    plan_hash = freeze_comparison(engine, make_plan(engine))

    def opposite_errors(self, dataset, spec, **kwargs):
        labels = []
        for index, example in enumerate(dataset.test):
            correct = (index % 2 == 0) == (spec.title == "Synthetic candidate")
            labels.append(example.y if correct else ("b" if example.y == "a" else "a"))
        return ExecutionResult(
            Predictions(ids=tuple(example.id for example in dataset.test), labels=tuple(labels)), {}
        )

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", opposite_errors)
    score = run_comparison(engine, plan_hash, "discordant-pairs")["assessment"]
    assert score["effect"] == 0
    assert score["candidate_accuracy"] == score["baseline_accuracy"] == 0.5
    assert score["confidence_interval"]["lower"] < 0
    assert score["confidence_interval"]["upper"] > 0
    assert score["verdict"] == "inconclusive"


def test_plan_is_revalidated_canonical_and_immutable(engine):
    plan = make_plan(engine)
    with pytest.raises(ValidationError):
        freeze_comparison(engine, plan.model_copy(update={"confidence": math.nan}))
    plan_hash = freeze_comparison(engine, plan)
    assert engine.artifacts.get(plan_hash) == canonical_bytes(plan)
    pretty = engine.artifacts.put(json.dumps(plan.model_dump(mode="json"), indent=2).encode())
    with pytest.raises(InvalidComparison, match="canonical"):
        run_comparison(engine, pretty, "noncanonical")
    with pytest.raises(ValidationError):
        plan.seed = 2


@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_spec_hash": "A" * 64},
        {"baseline_spec_hash": "../escape"},
        {"minimum_effect": -0.1},
        {"minimum_effect": 1.1},
        {"minimum_effect": True},
        {"minimum_effect": "0.1"},
        {"confidence": 0},
        {"confidence": 1},
        {"confidence": math.inf},
        {"bootstrap_samples": 199},
        {"bootstrap_samples": 20001},
        {"bootstrap_samples": True},
        {"min_test_examples": 0},
        {"min_test_examples": -1},
        {"min_test_examples": True},
        {"min_test_examples": "20"},
        {"seed": -1},
        {"seed": "2"},
        {"extra": "forbidden"},
    ],
)
def test_comparison_scalar_integrity(updates):
    values = {"candidate_spec_hash": "a" * 64, "baseline_spec_hash": "b" * 64}
    with pytest.raises(ValidationError):
        ComparisonSpec(**{**values, **updates})


@pytest.mark.parametrize("key", ["", " \n", True, None])
def test_comparison_requires_nonempty_string_key(engine, key):
    plan_hash = freeze_comparison(engine, make_plan(engine))
    with pytest.raises(InvalidComparison):
        run_comparison(engine, plan_hash, key)
    assert engine.runs.list_runs() == []
