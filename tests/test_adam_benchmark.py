"""Adversarial contracts for the finite paper-derived Adam experiment."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from paper2bench.adam_benchmark import (
    INPUTS, METRIC, SCORER, UNITS, build_reference, oracle_trajectories, score_adam,
)
from paper2bench.bundles import BundleError, atomic_json, canonical_json, digest, read_json, validate_task
from paper2bench.executors import ExecutionResult
from paper2bench.methods import SCORER_PROTOCOLS, score
from paper2bench.scientific_runs import _metric_record, execute_scientific

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/papers/adam-counterexample"


def load_example(name):
    spec = importlib.util.spec_from_file_location("adam_example_" + name, EXAMPLE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate():
    implementation = load_example("solution")
    return {"schema_version": 1, "recording": "pre_update",
            "adam": implementation.run_algorithm(INPUTS, amsgrad=False),
            "amsgrad": implementation.run_algorithm(INPUTS, amsgrad=True)}


@pytest.fixture(scope="module")
def reference():
    return build_reference(INPUTS)


def test_candidate_recurrence_matches_independent_geometric_sum_oracle(candidate, reference):
    observed = score(SCORER, INPUTS, candidate, reference)
    assert observed["verdict"] == "pass" and observed["observation_valid"] is True
    assert observed["max_trace_absolute_error"] < 1e-12
    assert observed["metric"] == METRIC and observed["units"] == UNITS
    assert observed["adam_average_regret"] >= 2 / 3
    assert observed["amsgrad_average_regret"] < observed["adam_average_regret"]
    assert [point["horizon"] for point in observed["checkpoints"]] == [3, 30, 300, 3000, 30000]
    assert len(json.dumps(candidate, separators=(",", ":")).encode()) < 2 * 1024 * 1024
    assert SCORER_PROTOCOLS[SCORER] == "adam_counterexample/v1"


def test_oracle_first_update_and_pre_update_indexing():
    expected = oracle_trajectories(INPUTS)
    for algorithm in ("adam", "amsgrad"):
        positions = expected[algorithm]["positions"]
        assert positions[0] == 1
        assert positions[1] == pytest.approx(1 - 0.1 * 3 / math.sqrt(0.9 * 9), abs=1e-15)
        assert len(positions) == 30000
        assert positions[-1] != expected[algorithm]["final_position"] or positions[-1] == 1


@pytest.mark.parametrize("mutation", ["short", "long", "nan", "infinity", "bool", "out_of_bounds", "wrong_recording", "claimed_metric"])
def test_rejects_malformed_or_unbounded_output(candidate, reference, mutation):
    output = deepcopy(candidate)
    if mutation == "short":
        output["adam"]["positions"].pop()
    elif mutation == "long":
        output["adam"]["positions"].append(0)
    elif mutation in {"nan", "infinity", "bool", "out_of_bounds"}:
        output["adam"]["positions"][0] = {"nan": float("nan"), "infinity": float("inf"),
                                             "bool": True, "out_of_bounds": 1.000001}[mutation]
    elif mutation == "wrong_recording":
        output["recording"] = "post_update"
    else:
        output["value"] = reference["expected_gap"]
    with pytest.raises(BundleError):
        score_adam(INPUTS, output, reference)


def test_matching_paired_scalar_does_not_validate_fabricated_trajectories(candidate, reference):
    output = deepcopy(candidate)
    # The same loss change in both trajectories cancels in their paired gap.
    for algorithm in ("adam", "amsgrad"):
        output[algorithm]["positions"][1] -= 0.01
    observed = score_adam(INPUTS, output, reference)
    assert observed["value"] == pytest.approx(reference["expected_gap"], abs=1e-14)
    assert observed["verdict"] == "fail" and observed["observation_valid"] is False
    assert observed["failure_reason"] == "incorrect_algorithm_trace"


def test_final_update_is_checked_even_though_it_does_not_enter_regret(candidate, reference):
    output = deepcopy(candidate)
    output["amsgrad"]["final_position"] += 0.001
    observed = score_adam(INPUTS, output, reference)
    assert observed["value"] == reference["expected_gap"]
    assert observed["observation_valid"] is False


@pytest.mark.parametrize("field,value", [("steps", 3), ("alpha", 0.2), ("beta2", 0.9),
                                         ("epsilon", 1e-8), ("bias_correction", True),
                                         ("initial_position", 0)])
def test_candidate_cannot_select_an_easier_condition(candidate, reference, field, value):
    inputs = deepcopy(INPUTS)
    inputs[field] = value
    with pytest.raises(BundleError, match="installed Appendix A condition"):
        score_adam(inputs, candidate, reference)


@pytest.mark.parametrize("field,value", [("expected_gap", 0), ("trace_tolerance", 1),
                                         ("tolerance", 1), ("inputs_sha256", "a" * 64)])
def test_frozen_reference_is_bound_to_oracle_and_inputs(candidate, reference, field, value):
    forged = dict(reference, **{field: value})
    with pytest.raises(BundleError):
        score_adam(INPUTS, candidate, forged)


def test_scientific_observation_requires_trace_validity_without_changing_mse_semantics():
    child = {"id": "child"}
    report = {"execution": "pass", "predictions_sha256": "a" * 64,
              "metric": {"metric": METRIC, "units": UNITS, "value": 0.615,
                         "observation_valid": False, "verdict": "fail"}}
    observation = _metric_record(0, child, report, SCORER)
    assert observation["status"] == "failed" and observation["value"] is None
    report["metric"] = {"metric": "mean_squared_error", "units": "target_units_squared",
                        "value": 1.0, "samples": 3, "verdict": "fail"}
    observation = _metric_record(0, child, report, "linear_predictions/v1")
    assert observation["status"] == "succeeded" and observation["value"] == 1.0


def test_builder_freezes_reference_without_running_candidate(tmp_path):
    replacement = tmp_path / "do_not_execute.py"
    replacement.write_text("raise RuntimeError('must not execute while freezing')\n")
    task_path = load_example("build_task").build(tmp_path / "task", solution=replacement)
    task = validate_task(task_path, require_executable=True)
    curator = read_json(task_path.parent / "curator_manifest.json")
    assert task["files"]["solution.py"] == digest(replacement.read_bytes())
    assert curator["task_manifest_sha256"] == digest(canonical_json(task))
    assert curator["reference"] == build_reference(INPUTS)
    assert "protocol" not in curator


def reviewed_test_task(tmp_path):
    task_path = load_example("build_task").build(tmp_path / "task")
    manifest = read_json(task_path)
    curator_path = task_path.parent / "curator_manifest.json"
    curator = read_json(curator_path)
    paper_hash, code_hash = manifest["source"]["paper_sha256"], manifest["files"]["solution.py"]
    curator["protocol"] = {
        "schema_version": 1, "profile": "reproduction/v1", "protocol_id": "test-only-adam-runtime-contract",
        "task_manifest_sha256": digest(canonical_json(manifest)), "submission_sha256": code_hash,
        "source": {"paper_sha256": paper_hash, "citations": [
            {"id": "appendix-a", "locator": "Appendix A pp10-11", "claim": "Test fixture for the reviewed numerical contract"}]},
        "datasets": {"input.json": manifest["files"]["input.json"]},
        "plan": {"seed_schedule": [0, 1, 2], "stopping_rule": "fixed_schedule",
                 "variability_source": "deterministic repeatability; not independent population samples",
                 "independence_assumption": False},
        "criteria": [
            {"id": "execution", "kind": "execution", "dimension": "execution", "critical": True, "depends_on": []},
            {"id": "fidelity", "kind": "review", "dimension": "implementation_fidelity", "critical": True,
             "depends_on": ["execution"], "review_id": "test-review"},
            {"id": "agreement", "kind": "metric", "dimension": "empirical_agreement", "critical": True,
             "depends_on": ["execution", "fidelity"], "metric_id": METRIC, "units": UNITS,
             "rule": "absolute_tolerance", "reference": curator["reference"]["expected_gap"],
             "tolerance": curator["reference"]["tolerance"], "min_samples_per_run": 1}],
        "reviews": [{"id": "test-review", "reviewer": "test-only runtime fixture; no real fidelity approval",
                     "status": "pass", "submission_sha256": code_hash, "paper_sha256": paper_hash,
                     "source_citations": ["appendix-a"], "evidence_refs": ["test-only-review"]}]}
    atomic_json(curator_path, curator)
    return task_path, curator_path


@pytest.mark.parametrize("fabricated", [False, True])
def test_full_scientific_runner_uses_installed_trace_scorer(tmp_path, candidate, fabricated):
    task, curator = reviewed_test_task(tmp_path)

    class Worker:
        calls = []

        def preflight(self):
            return {"image_id": "sha256:" + "1" * 64}

        def execute(self, request):
            self.calls.append(request.seed)
            assert not (request.workspace / "curator_manifest.json").exists()
            output = deepcopy(candidate)
            if fabricated:
                output["amsgrad"]["final_position"] += 0.001
            # Keep the complete raw output below the same bounded JSON reader.
            (request.workspace / "predictions.json").write_text(json.dumps(output, separators=(",", ":")))
            return ExecutionResult("succeeded", 0, "", "", 0.01, evidence={"cleanup_confirmed": True})

    worker = Worker()
    worker.calls = []
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert worker.calls == [0, 1, 2]
    assert report["pass"] is (not fabricated)
    # Invalid traces are missing scientific observations, so agreement remains
    # inconclusive instead of becoming a contradictory faithful measurement.
    assert report["scientific_reproduction"] == ("inconclusive" if fabricated else "pass")
    assert len(report["child_run_ids"]) == 3
