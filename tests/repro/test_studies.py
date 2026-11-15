"""Multi-seed paired study registration, failure retention, and uncertainty."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from paper2bench.repro.demo import create_demo
from paper2bench.repro.engine import Engine
from paper2bench.repro.models import Predictions, canonical_bytes
from paper2bench.repro.runtime import ExecutionFailure, ExecutionResult, ResourceExhausted
from paper2bench.repro.studies import InvalidStudy, StudySpec, freeze_study, run_study


@pytest.fixture
def engine(tmp_path):
    return Engine(tmp_path / "state")


def make_study(engine, **updates):
    baseline = create_demo(engine)
    candidate = engine.load_spec(baseline).model_dump(mode="json")
    candidate["title"] = "Synthetic study candidate"
    candidate_hash = engine.freeze(candidate)
    return StudySpec(
        **{
            "candidate_spec_hash": candidate_hash,
            "baseline_spec_hash": baseline,
            "seeds": (2, 7, 11),
            "bootstrap_samples": 200,
            **updates,
        }
    )


def mock_predictions(monkeypatch, candidate, baseline):
    def run(_self, dataset, spec, **_kwargs):
        rule = candidate if spec.title == "Synthetic study candidate" else baseline
        count = rule[spec.seed] if isinstance(rule, dict) else rule
        labels = tuple(
            row.y if index < count else ("b" if row.y == "a" else "a")
            for index, row in enumerate(dataset.test)
        )
        ids = tuple(row.id for row in dataset.test)
        if spec.title == "Synthetic study candidate":
            ids, labels = ids[::-1], labels[::-1]
        return ExecutionResult(
            Predictions(ids=ids, labels=labels),
            {"wall_seconds": 1.25, "cost_usd": None},
        )

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", run)


def test_plan_resolves_every_seed_and_role_before_registering_jobs(engine):
    study = make_study(engine)
    plan_hash = freeze_study(engine, study)
    plan = engine.artifacts.get_json(plan_hash)
    assert plan["kind"] == "paired_accuracy_study_plan"
    assert plan["study"] == study.model_dump(mode="json")
    assert engine.runs.list_runs() == []
    assert [(trial["role"], trial["seed"]) for trial in plan["trials"]] == [
        (role, seed) for seed in study.seeds for role in ("candidate", "baseline")
    ]
    for trial in plan["trials"]:
        child = engine.load_spec(trial["spec_hash"])
        original = engine.load_spec(getattr(study, f"{trial['role']}_spec_hash"))
        assert child.seed == trial["seed"]
        assert child.model_dump(exclude={"seed"}) == original.model_dump(exclude={"seed"})
    assert freeze_study(engine, study) == plan_hash


def test_all_trials_exist_before_first_execution(engine, monkeypatch):
    study = make_study(engine)
    plan_hash = freeze_study(engine, study)
    execute = engine.execute
    observed = []

    def checked_execute(run_id):
        assert len(engine.runs.list_runs()) == 2 * len(study.seeds)
        observed.append([run["state"] for run in engine.runs.list_runs()])
        return execute(run_id)

    monkeypatch.setattr(engine, "execute", checked_execute)
    run_study(engine, plan_hash, "preregister")
    assert observed[0] == ["queued"] * 6
    assert len(observed) == 6


@pytest.mark.parametrize(
    "candidate,baseline,effect,verdict",
    [(80, 0, 1.0, "supported"), (0, 80, -1.0, "not_supported"), (80, 80, 0.0, "not_supported")],
)
def test_known_direction_and_paired_case_alignment(
    engine, monkeypatch, candidate, baseline, effect, verdict
):
    plan_hash = freeze_study(engine, make_study(engine))
    mock_predictions(monkeypatch, candidate, baseline)
    result = run_study(engine, plan_hash, "direction")
    assessment = result["assessment"]
    assert assessment["verdict"] == verdict
    assert assessment["effect"] == effect
    assert assessment["seed_count"] == 3 and assessment["sample_count"] == 80
    assert assessment["paired_observation_count"] == 240
    assert len(assessment["paired_correctness_differences"]) == 3
    assert assessment["case_order"][0] == "test-000"
    assert assessment["confidence_interval"]["lower"] == effect
    assert assessment["confidence_interval"]["upper"] == effect
    assert result["implementation_fidelity"] == result["protocol_fidelity"] == "unverified"


def test_seed_resampling_preserves_large_between_fit_variability(engine, monkeypatch):
    plan_hash = freeze_study(engine, make_study(engine))
    # Within each seed all cases agree; uncertainty must still include variability
    # between fits. Flattening seed/case cells would produce a falsely narrow CI.
    mock_predictions(monkeypatch, {2: 80, 7: 0, 11: 0}, 0)
    result = run_study(engine, plan_hash, "seed-variation")
    score = result["assessment"]
    assert score["effect"] == pytest.approx(1 / 3)
    assert [trial["effect"] for trial in score["between_seed_effects"]] == [1.0, 0.0, 0.0]
    assert score["between_seed_effect_standard_deviation"] == pytest.approx(1 / math.sqrt(3))
    assert score["confidence_interval"]["lower"] == 0.0
    assert score["confidence_interval"]["upper"] > 0.9
    assert score["verdict"] == "inconclusive"


def test_case_resampling_contributes_uncertainty_with_identical_seed_effects(engine, monkeypatch):
    plan_hash = freeze_study(engine, make_study(engine))
    mock_predictions(monkeypatch, 40, 0)
    score = run_study(engine, plan_hash, "case-variation")["assessment"]
    assert score["effect"] == 0.5
    assert score["between_seed_effect_standard_deviation"] == 0
    assert score["confidence_interval"]["lower"] < 0.5
    assert score["confidence_interval"]["upper"] > 0.5
    assert score["confidence_interval"]["method"] == "two_way_crossed_percentile_bootstrap"


def test_minimum_effect_is_strict_and_minimum_case_count_is_prespecified(engine, monkeypatch):
    study = make_study(engine, minimum_effect=1.0)
    mock_predictions(monkeypatch, 80, 0)
    result = run_study(engine, freeze_study(engine, study), "effect-boundary")
    assert result["assessment"]["verdict"] == "not_supported"
    strict = make_study(engine, min_test_examples=81)
    strict_result = run_study(engine, freeze_study(engine, strict), "minimum-cases")
    assert strict_result["assessment"]["sample_count"] == 80
    assert strict_result["assessment"]["minimum_sample_count_met"] is False
    assert strict_result["assessment"]["verdict"] == "inconclusive"


def test_idempotent_rerun_uses_same_trials_and_result_artifact(engine, monkeypatch):
    plan_hash = freeze_study(engine, make_study(engine))
    mock_predictions(monkeypatch, 50, 40)
    first = run_study(engine, plan_hash, "stable-study")

    def forbidden(*_args, **_kwargs):
        pytest.fail("completed study trials cannot execute twice")

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", forbidden)
    second = run_study(engine, plan_hash, "stable-study")
    assert first == second
    assert len(engine.runs.list_runs()) == 6
    assert all(run["attempt"] == 1 for run in engine.runs.list_runs())
    assert engine.artifacts.get_json(first["result_hash"]) == {
        key: value for key, value in first.items() if key != "result_hash"
    }


def test_identical_original_spec_still_has_distinct_role_and_seed_runs(engine):
    spec_hash = create_demo(engine)
    study = StudySpec(
        candidate_spec_hash=spec_hash,
        baseline_spec_hash=spec_hash,
        seeds=(0, 1),
        bootstrap_samples=200,
    )
    result = run_study(engine, freeze_study(engine, study), "same-spec")
    assert len({trial["run_id"] for trial in result["trials"]}) == 4
    assert result["assessment"]["effect"] == 0
    assert result["assessment"]["between_seed_effect_standard_deviation"] == 0


@pytest.mark.parametrize("exception", [ExecutionFailure, ResourceExhausted])
def test_any_failed_trial_blocks_primary_verdict_without_dropping_other_trials(
    engine, monkeypatch, exception
):
    plan_hash = freeze_study(engine, make_study(engine))
    mock_predictions(monkeypatch, 80, 0)
    original = __import__("paper2bench.repro.engine", fromlist=["BuiltinRunner"]).BuiltinRunner.run

    def one_failure(self, dataset, spec, **kwargs):
        if spec.title == "Synthetic study candidate" and spec.seed == 7:
            raise exception("private untrusted detail")
        return original(self, dataset, spec, **kwargs)

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", one_failure)
    result = run_study(engine, plan_hash, "one-failure")
    assert result["assessment"]["verdict"] == "not_tested"
    assert result["assessment"]["effect"] is None
    assert len(result["trials"]) == 6
    states = [trial["state"] for trial in result["trials"]]
    assert states.count("completed") == 5
    assert ("failed" if exception is ExecutionFailure else "resource_exhausted") in states
    assert sum(item["trial_count"] for item in result["resources"].values()) == 6
    assert "private untrusted detail" not in json.dumps(result)
    assert all(trial["attempts"] for trial in result["trials"])


def test_active_trial_is_not_selected_out_of_the_primary_analysis(engine, monkeypatch):
    plan_hash = freeze_study(engine, make_study(engine))
    original_submit = engine.submit

    def leased_submit(spec_hash, key):
        run = original_submit(spec_hash, key)
        if len(engine.runs.list_runs()) == 1:
            engine.runs.claim(run["run_id"], "another-worker", 300)
        return run

    monkeypatch.setattr(engine, "submit", leased_submit)
    result = run_study(engine, plan_hash, "active-worker")
    assert result["assessment"]["verdict"] == "not_tested"
    assert any(trial["state"] == "running" for trial in result["trials"])
    assert len(result["trials"]) == 6


def test_resource_aggregation_keeps_unknown_costs_visible(engine, monkeypatch):
    plan_hash = freeze_study(engine, make_study(engine))
    mock_predictions(monkeypatch, 80, 0)
    result = run_study(engine, plan_hash, "resources")
    for role in ("candidate", "baseline"):
        usage = result["resources"][role]
        assert usage["trial_count"] == usage["completed_trials"] == 3
        assert usage["wall_seconds_known_total"] == 3.75
        assert usage["wall_seconds_unknown_trials"] == 0
        assert usage["cost_usd_unknown_trials"] == 3
        assert usage["cost_usd_known_total"] == 0
        assert "not treated as zero" in usage["cost_interpretation"]


def test_bootstrap_cap_fails_before_jobs_and_is_rechecked_for_registered_plan(engine, monkeypatch):
    valid_plan = engine.artifacts.get_json(freeze_study(engine, make_study(engine)))
    study = make_study(engine, seeds=tuple(range(50)), bootstrap_samples=20000)
    assert 80 * 50 * 20000 > 20_000_000

    def forbidden(*_args, **_kwargs):
        pytest.fail("over-budget study must never submit or execute a trial")

    monkeypatch.setattr(engine, "submit", forbidden)
    with pytest.raises(InvalidStudy, match="20000000 paired draws"):
        freeze_study(engine, study)
    valid_plan["study"] = study.model_dump(mode="json")
    manually_registered = engine.artifacts.put_json(valid_plan)
    with pytest.raises(InvalidStudy, match="20000000 paired draws"):
        run_study(engine, manually_registered, "over-budget")
    assert engine.runs.list_runs() == []


def test_different_dataset_rejected_before_trials(engine):
    study = make_study(engine)
    candidate = engine.load_spec(study.candidate_spec_hash).model_dump(mode="json")
    data = engine.artifacts.get_json(candidate["dataset_sha256"])
    data["test"][0]["x"] = [-3.0]
    candidate["dataset_sha256"] = engine.register_dataset(data)
    different = study.model_copy(update={"candidate_spec_hash": engine.freeze(candidate)})
    with pytest.raises(InvalidStudy, match="same dataset"):
        freeze_study(engine, different)
    assert engine.runs.list_runs() == []


def test_model_seed_control_is_rejected_without_a_provider_request(engine):
    study = make_study(engine)
    candidate = engine.load_spec(study.candidate_spec_hash).model_dump(mode="json")
    profile = engine.register_profile(
        {"name": "unused", "base_url": "http://127.0.0.1:9999/v1", "location": "local"}
    )
    candidate["workload"] = {
        "kind": "model_classification",
        "provider_profile": profile,
        "model": "unused-model",
        "prompt": "Classify",
    }
    candidate["policy"]["max_provider_calls"] = 80
    unsupported = study.model_copy(update={"candidate_spec_hash": engine.freeze(candidate)})
    with pytest.raises(InvalidStudy, match="seed control is unsupported"):
        freeze_study(engine, unsupported)
    assert engine.runs.list_runs() == []


@pytest.mark.parametrize("tamper", ["schedule", "seed", "bootstrap", "dataset"])
def test_tampered_plan_is_rejected_before_submission(engine, tamper):
    plan_hash = freeze_study(engine, make_study(engine))
    plan = engine.artifacts.get_json(plan_hash)
    if tamper == "schedule":
        plan["trials"] = plan["trials"][::-1]
    elif tamper == "seed":
        plan["trials"][0]["spec_hash"] = plan["trials"][2]["spec_hash"]
    elif tamper == "bootstrap":
        plan["bootstrap_seed"] += 1
    else:
        plan["dataset_sha256"] = "f" * 64
    altered = engine.artifacts.put_json(plan)
    with pytest.raises(InvalidStudy):
        run_study(engine, altered, "tampered")
    assert engine.runs.list_runs() == []


def test_noncanonical_plan_is_rejected(engine):
    plan_hash = freeze_study(engine, make_study(engine))
    plan = engine.artifacts.get_json(plan_hash)
    noncanonical = engine.artifacts.put(json.dumps(plan, indent=2).encode())
    assert engine.artifacts.get(noncanonical) != canonical_bytes(plan)
    with pytest.raises(InvalidStudy, match="canonical"):
        run_study(engine, noncanonical, "noncanonical")


@pytest.mark.parametrize(
    "updates",
    [
        {"seeds": (0,)},
        {"seeds": (0, 0)},
        {"seeds": (True, 1)},
        {"seeds": (-1, 1)},
        {"seeds": tuple(range(51))},
        {"seeds": (0, 2**63)},
        {"minimum_effect": -0.1},
        {"minimum_effect": 1.1},
        {"minimum_effect": True},
        {"confidence": math.nan},
        {"confidence": 0},
        {"confidence": 1},
        {"bootstrap_samples": 199},
        {"bootstrap_samples": 20001},
        {"bootstrap_samples": True},
        {"min_test_examples": 0},
        {"extra": "forbidden"},
    ],
)
def test_study_contract_rejects_invalid_values(updates):
    values = {"candidate_spec_hash": "a" * 64, "baseline_spec_hash": "b" * 64, "seeds": (0, 1)}
    with pytest.raises(ValidationError):
        StudySpec(**{**values, **updates})


@pytest.mark.parametrize("key", ["", " \n", None, True])
def test_invalid_idempotency_keys_do_not_register_trials(engine, key):
    plan_hash = freeze_study(engine, make_study(engine))
    with pytest.raises(InvalidStudy):
        run_study(engine, plan_hash, key)
    assert engine.runs.list_runs() == []
