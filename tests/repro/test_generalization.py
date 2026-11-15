from __future__ import annotations

import pytest

from paper2bench.repro.bundles import export_bundle, replay_bundle
from paper2bench.repro.comparisons import ComparisonSpec, freeze_comparison, run_comparison
from paper2bench.repro.demo import create_optimization_demo
from paper2bench.repro.engine import Engine, InvalidExperiment
from paper2bench.repro.models import ExperimentSpec
from paper2bench.repro.runtime import ExecutionFailure, _decode_predictions


def test_optimization_executes_and_replays_without_classification_contract(tmp_path):
    engine = Engine(tmp_path / "state")
    spec_hash = create_optimization_demo(engine)
    run = engine.run(spec_hash, "optimization")
    assert run["state"] == "completed"
    result = engine.result(run["run_id"])["result"]
    assert result["assessment"]["mean_score"] == 1.0
    assert result["assessment"]["feasible_count"] == 80
    assert result["assessment"]["criterion"] == "optimality_equivalence"
    assert "Accuracy:" not in engine.report(run["run_id"])
    archive = export_bundle(engine, run["run_id"], tmp_path / "optimization.zip")
    assert replay_bundle(archive)["assessment"] == result["assessment"]


def test_optimization_comparison_uses_paired_scores_not_labels(tmp_path):
    engine = Engine(tmp_path)
    spec_hash = create_optimization_demo(engine)
    plan = freeze_comparison(
        engine, ComparisonSpec(candidate_spec_hash=spec_hash, baseline_spec_hash=spec_hash)
    )
    result = run_comparison(engine, plan, "paired-knapsack")
    assert result["assessment"]["criterion"] == "paired_optimality_superiority"
    assert result["assessment"]["effect"] == 0
    assert result["assessment"]["candidate_mean_score"] == 1.0


def test_wrong_task_metric_fails_before_run_creation(tmp_path):
    engine = Engine(tmp_path)
    original = engine.load_spec(create_optimization_demo(engine))
    raw = original.model_dump(mode="json")
    raw["claim"] = {"target": 1.0, "metric": "accuracy"}
    with pytest.raises(InvalidExperiment, match="different tasks"):
        engine.freeze(ExperimentSpec.model_validate(raw))
    assert engine.runs.list_runs() == []


def test_local_only_cannot_be_placed_remotely(tmp_path):
    engine = Engine(tmp_path)
    raw = engine.load_spec(create_optimization_demo(engine)).model_dump(mode="json")
    raw["policy"]["placement"] = "remote"
    with pytest.raises(InvalidExperiment, match="local-only"):
        engine.freeze(raw)


def test_legacy_classification_bundle_remains_replayable():
    from pathlib import Path

    result = replay_bundle(Path(__file__).parent / "fixtures" / "classification-v1.zip")
    assert result["scores"] == "recomputed"
    assert result["assessment"]["accuracy"] == 1.0


def test_surrogate_predictions_rejected_before_receipt_serialization():
    with pytest.raises(ExecutionFailure):
        _decode_predictions(b'{"ids":["case"],"labels":["\\ud800"]}')
