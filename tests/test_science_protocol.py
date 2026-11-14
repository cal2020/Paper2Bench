"""Scientific contracts tested with declared synthetic protocol fixtures.

Hash fixtures are identities for this test, not evidence of a published result.
"""
from copy import deepcopy
import hashlib
import json
import subprocess
import sys

import pytest

from paper2bench.evaluation.science import evaluate_protocol, freeze_protocol
from paper2bench.evaluation.statistics import compare_summary, summarize
from paper2bench.evaluation.validation import EvaluationError

CODE = "c" * 64
PAPER = "a" * 64
DATA = "b" * 64


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def fixture(profile="reproduction/v1"):
    task = {"schema_version": 1, "entrypoint": "main.py", "inputs": "data.json",
            "files": {"main.py": CODE, "data.json": DATA}, "source": {"paper_sha256": PAPER}}
    dimension = "empirical_agreement" if profile == "reproduction/v1" else "transfer"
    protocol = {"schema_version": 1, "profile": profile, "protocol_id": "synthetic-contract-test",
                "task_manifest_sha256": canonical_sha(task), "submission_sha256": CODE,
                "source": {"paper_sha256": PAPER, "citations": [{"id": "method", "locator": "fixture method",
                                                                             "claim": "Synthetic method and result used only to test contracts"}]},
                "datasets": {"data.json": DATA},
                "plan": {"seed_schedule": [0, 1, 2], "stopping_rule": "fixed_schedule",
                         "variability_source": "independent initialization seeds", "independence_assumption": True},
                "criteria": [
                    {"id": "execution", "kind": "execution", "dimension": "execution", "critical": True, "depends_on": []},
                    {"id": "fidelity", "kind": "review", "dimension": "implementation_fidelity", "critical": True,
                     "depends_on": ["execution"], "review_id": "review"},
                    {"id": "agreement", "kind": "metric", "dimension": dimension, "critical": True,
                     "depends_on": ["execution", "fidelity"], "metric_id": "accuracy", "units": "fraction",
                     "rule": "absolute_tolerance", "reference": 0.8, "tolerance": 0.02, "bounds": [0, 1]}],
                "reviews": [{"id": "review", "reviewer": "independent fixture curator", "status": "pass",
                             "submission_sha256": CODE, "paper_sha256": PAPER,
                             "source_citations": ["method"], "evidence_refs": ["curator/review.json"]}]}
    if profile == "transfer/v1":
        protocol["transformation"] = {"parent_task_sha256": "e" * 64, "changed_factors": ["data domain"],
                                      "expected_outcome_source": "variant_protocol"}
    return task, protocol


def envelope(task, protocol, values=(0.79, 0.8, 0.81)):
    frozen = freeze_protocol(protocol, task_manifest=task, submission_sha256=CODE)
    return {"origin": "controller", "protocol_sha256": frozen.sha256, "protocol_locked_before_execution": True,
            "task_manifest_sha256": protocol["task_manifest_sha256"], "submission_sha256": CODE,
            "datasets": protocol["datasets"], "execution": {"status": "succeeded", "clean_run": True,
                "run_id": "run-123", "evidence_refs": ["controller/execution.json"]},
            "metrics": [{"id": "accuracy", "seed": seed, "status": "succeeded", "value": value,
                         "units": "fraction", "n": 100, "artifact_sha256": "f" * 64,
                         "evidence_refs": [f"controller/metrics-{seed}.json"], "origin": "trusted_recomputation"}
                        for seed, value in enumerate(values)]}


def evaluate(task, protocol, evidence):
    return evaluate_protocol(protocol, task_manifest=task, submission_sha256=CODE, measurement_evidence=evidence)


def test_complete_locked_faithful_experiment_passes_scoped_protocol():
    task, protocol = fixture()
    report = evaluate(task, protocol, envelope(task, protocol))
    assert report["scientific_reproduction"] == "pass"
    assert report["dimensions"] == {"execution": "pass", "implementation_fidelity": "pass",
                                    "empirical_agreement": "pass", "transfer": "not_assessed"}
    assert report["criteria"][-1]["summary"]["values"] == [0.79, 0.8, 0.81]
    assert report["criteria"][-1]["comparison"]["interpretation"] == "point_agreement_with_declared_tolerance"
    assert report["coverage"]["coverage"] == 1


def test_faithful_contradiction_is_distinct_from_method_or_execution_failure():
    task, protocol = fixture()
    report = evaluate(task, protocol, envelope(task, protocol, (0.2, 0.3, 0.25)))
    assert report["scientific_reproduction"] == "fail"
    assert report["contradictory_faithful_result"] is True
    assert report["dimensions"]["implementation_fidelity"] == "pass"


def test_candidate_authority_and_changed_identities_are_rejected():
    task, protocol = fixture()
    evidence = envelope(task, protocol)
    evidence["origin"] = "candidate"
    with pytest.raises(EvaluationError, match="Candidate"):
        evaluate(task, protocol, evidence)
    evidence = envelope(task, protocol)
    evidence["metrics"][0]["origin"] = "candidate"
    with pytest.raises(EvaluationError, match="Candidate"):
        evaluate(task, protocol, evidence)
    for field in ("task_manifest_sha256", "submission_sha256", "protocol_sha256"):
        evidence = envelope(task, protocol)
        evidence[field] = "0" * 64
        with pytest.raises(EvaluationError, match="match"):
            evaluate(task, protocol, evidence)


def test_review_applies_only_to_exact_submission_and_paper():
    task, protocol = fixture()
    protocol["reviews"][0]["submission_sha256"] = "0" * 64
    with pytest.raises(EvaluationError, match="Curator review"):
        freeze_protocol(protocol, task_manifest=task, submission_sha256=CODE)


@pytest.mark.parametrize("mutation", ["missing_review", "missing_seed", "failed_seed", "no_lock", "no_evidence"])
def test_missing_and_failed_requirements_remain_inconclusive(mutation):
    task, protocol = fixture()
    if mutation == "missing_review":
        protocol["reviews"] = []
    evidence = envelope(task, protocol)
    if mutation == "missing_seed":
        evidence["metrics"].pop()
    elif mutation == "failed_seed":
        evidence["metrics"][-1].update(status="failed", value=None)
    elif mutation == "no_lock":
        evidence["protocol_locked_before_execution"] = False
    elif mutation == "no_evidence":
        evidence = {}
    report = evaluate(task, protocol, evidence)
    assert not report["pass"] and report["scientific_reproduction"] == "inconclusive"
    assert report["coverage"]["unknown_weight"] > 0


def test_wrong_units_cannot_pass_numeric_match():
    task, protocol = fixture()
    evidence = envelope(task, protocol)
    evidence["metrics"][0]["units"] = "percent"
    report = evaluate(task, protocol, evidence)
    assert report["dimensions"]["empirical_agreement"] == "fail"
    assert report["criteria"][-1]["reason"] == "metric_units_mismatch"


def test_failed_fidelity_gate_blocks_matching_metric():
    task, protocol = fixture()
    protocol["reviews"][0]["status"] = "fail"
    report = evaluate(task, protocol, envelope(task, protocol))
    assert report["criteria"][-1]["verdict"] == "inconclusive"
    assert report["criteria"][-1]["reason"] == "required_dependency_did_not_pass"
    assert not report["pass"]


def test_duplicate_or_out_of_schedule_observations_cannot_be_cherry_picked():
    task, protocol = fixture()
    for bad in ("duplicate", "extra_seed"):
        evidence = envelope(task, protocol)
        if bad == "duplicate":
            evidence["metrics"].append(deepcopy(evidence["metrics"][0]))
        else:
            evidence["metrics"][0]["seed"] = 100
        with pytest.raises(EvaluationError):
            evaluate(task, protocol, evidence)


@pytest.mark.parametrize("bad", ["cycle", "unknown_dependency", "missing_gate", "ungated_metric", "review_hash", "dataset_hash"])
def test_invalid_frozen_protocol_fails_before_execution(bad):
    task, protocol = fixture()
    if bad == "cycle":
        protocol["criteria"][0]["depends_on"] = ["agreement"]
    elif bad == "unknown_dependency":
        protocol["criteria"][0]["depends_on"] = ["missing"]
    elif bad == "missing_gate":
        protocol["criteria"][0]["critical"] = False
    elif bad == "ungated_metric":
        protocol["criteria"][-1]["depends_on"] = []
    elif bad == "review_hash":
        protocol["reviews"][0]["paper_sha256"] = "0" * 64
    else:
        protocol["datasets"]["data.json"] = "0" * 64
    with pytest.raises(EvaluationError):
        freeze_protocol(protocol, task_manifest=task, submission_sha256=CODE)


def test_frozen_protocol_is_immune_to_original_dict_mutation():
    task, protocol = fixture()
    evidence = envelope(task, protocol)
    frozen = freeze_protocol(protocol, task_manifest=task, submission_sha256=CODE)
    protocol["criteria"][-1]["reference"] = 0.1
    assert evaluate(task, frozen, evidence)["pass"]


def test_transfer_rules_can_reverse_original_effect():
    task, protocol = fixture("transfer/v1")
    protocol["criteria"][-1].update(reference=0.2)
    report = evaluate(task, protocol, envelope(task, protocol, (0.2, 0.2, 0.2)))
    assert report["transfer_verdict"] == "pass"
    assert report["scientific_reproduction"] == "not_assessed"
    protocol["transformation"]["expected_outcome_source"] = "original_paper_answer"
    with pytest.raises(EvaluationError, match="changed protocol"):
        freeze_protocol(protocol, task_manifest=task, submission_sha256=CODE)


def test_paired_effect_uses_same_seed_and_units():
    task, protocol = fixture()
    protocol["criteria"][-1].update(baseline_id="baseline", reference=0.1, tolerance=1e-10, bounds=[-1, 1])
    evidence = envelope(task, protocol)
    for record in list(evidence["metrics"]):
        evidence["metrics"].append(dict(record, id="baseline", value=record["value"] - 0.1))
    report = evaluate(task, protocol, evidence)
    assert report["pass"]
    assert report["criteria"][-1]["summary"]["mean"] == pytest.approx(0.1)


def test_equivalence_requires_interval_containment_not_nonsignificance():
    summary = summarize([0.8] * 100, bounds=[0, 1], independence_assumption=True)
    assert compare_summary(summary, rule="equivalence_interval", reference=0.8, tolerance=0.01)["verdict"] == "inconclusive"
    assert compare_summary(summary, rule="equivalence_interval", reference=0.8, tolerance=0.2)["verdict"] == "pass"
    no_bound = summarize([0.8, 0.8])
    assert compare_summary(no_bound, rule="equivalence_interval", reference=0.8, tolerance=0.2)["verdict"] == "inconclusive"


def test_summary_states_small_sample_uncertainty_and_bounds_assumptions():
    summary = summarize([0.5])
    assert summary["sample_standard_deviation"] is None and summary["confidence_interval"] is None
    summary = summarize([0.5, 0.5], bounds=[0, 1], independence_assumption=False)
    assert summary["confidence_interval"] is None
    with pytest.raises(EvaluationError):
        summarize([1.5], bounds=[0, 1], independence_assumption=True)


@pytest.mark.parametrize("field,value", [("profile", []), ("schema_version", True)])
def test_wrong_protocol_types_have_typed_errors(field, value):
    task, protocol = fixture()
    protocol[field] = value
    with pytest.raises(EvaluationError):
        freeze_protocol(protocol, task_manifest=task, submission_sha256=CODE)


@pytest.mark.parametrize("rule,reference,values,tolerance", [
    ("absolute_tolerance", 10**16, [10**16 + 1], 0),
    ("relative_tolerance", 10**16, [10**16 + 1], 0),
    ("at_most", 10**16, [10**16 + 1], None),
    ("at_least", 10**16 + 1, [1e16], None),
    ("absolute_tolerance", 10**16, [10**16, 10**16 + 1], 0),
])
def test_point_comparisons_do_not_round_observations_into_a_pass(rule, reference, values, tolerance):
    result = compare_summary(summarize(values), rule=rule, reference=reference, tolerance=tolerance)
    assert result["verdict"] == "fail"
    assert result["difference"] != 0


def test_complete_protocol_cannot_pass_a_rounded_large_integer_mean():
    task, protocol = fixture()
    protocol["criteria"][-1].update(reference=10**16, tolerance=0)
    protocol["criteria"][-1].pop("bounds")
    report = evaluate(task, protocol, envelope(task, protocol, [10**16 + 1] * 3))
    assert report["scientific_reproduction"] == "fail"
    assert report["criteria"][-1]["comparison"]["difference"] == 1


def test_paired_subtraction_cannot_round_a_real_effect_into_a_pass():
    task, protocol = fixture()
    protocol["criteria"][-1].update(baseline_id="baseline", reference=10**16, tolerance=0)
    protocol["criteria"][-1].pop("bounds")
    evidence = envelope(task, protocol, [10**16 + 1] * 3)
    evidence["metrics"].extend([dict(record, id="baseline", value=0.0)
                                for record in list(evidence["metrics"])])
    report = evaluate(task, protocol, evidence)
    assert report["scientific_reproduction"] == "fail"
    assert report["criteria"][-1]["comparison"]["difference"] == 1


def test_unrepresentable_paired_difference_is_an_error():
    task, protocol = fixture()
    protocol["criteria"][-1].update(baseline_id="baseline", reference=10**16, tolerance=0)
    protocol["criteria"][-1].pop("bounds")
    evidence = envelope(task, protocol, [10**16 + 1] * 3)
    evidence["metrics"].extend([dict(record, id="baseline", value=0.5)
                                for record in list(evidence["metrics"])])
    with pytest.raises(EvaluationError, match="precision loss"):
        evaluate(task, protocol, evidence)


@pytest.mark.parametrize("interval", [[2, 1], [float("nan"), 2], [0], [0, 1, 2], [2, 3]])
def test_malformed_uncertainty_interval_cannot_establish_equivalence(interval):
    with pytest.raises(EvaluationError):
        compare_summary({"mean": 1.5, "confidence_interval": interval},
                        rule="equivalence_interval", reference=1.5, tolerance=0)


def test_equivalence_rejects_an_interval_narrowed_after_observing_results():
    summary = summarize([0.8] * 100, bounds=[0, 1], independence_assumption=True)
    summary["confidence_interval"] = [0.8, 0.8]
    with pytest.raises(EvaluationError, match="does not match"):
        compare_summary(summary, rule="equivalence_interval", reference=0.8, tolerance=0)


def test_huge_integer_has_typed_scientific_validation_error():
    with pytest.raises(EvaluationError):
        summarize([10**1000])


def test_summary_mean_underflow_cannot_be_reported_as_exact_zero():
    with pytest.raises(EvaluationError, match="underflow"):
        summarize([0, 5e-324])


def test_shared_dependency_paths_have_bounded_validation_work():
    task, protocol = fixture()
    previous = ["execution", "fidelity"]
    for i in range(55):
        identity = f"review-{i}"
        protocol["criteria"].insert(-1, {"id": identity, "kind": "review",
            "dimension": "implementation_fidelity", "critical": False,
            "depends_on": previous[-2:], "review_id": "review"})
        previous.append(identity)
    protocol["criteria"][-1]["depends_on"] = previous[-2:]
    # A shared DAG has exponentially many paths but only 58 bounded nodes.
    # Keep a future path-expansion regression bounded rather than hanging CI.
    program = ("import json,sys; from paper2bench.evaluation.science import freeze_protocol; "
               "p=json.load(sys.stdin); print(freeze_protocol(p['protocol'], "
               "task_manifest=p['task'], submission_sha256=p['submission']).sha256)")
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True,
                            input=json.dumps({"protocol": protocol, "task": task, "submission": CODE}),
                            timeout=3, check=False)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) == 64
