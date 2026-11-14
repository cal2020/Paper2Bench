"""Recompute the public paper receipt using only repository-published evidence.

These checks need neither private authoring outputs nor Docker/model access.
They verify the recorded artifact chain and finite result, not a new experiment.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

import pytest

from paper2bench.adam_benchmark import PAPER_SHA256, trajectory_statistics
from paper2bench.bundles import canonical_json, digest, read_json, validate_task
from paper2bench.evaluation.science import freeze_protocol
from paper2bench.methods import SCORER_PROTOCOLS, score

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/papers/adam-counterexample"
PUBLIC = ROOT / "docs/results/adam-counterexample.json"
PREDICTIONS = ROOT / "docs/results/adam-predictions.json"


@pytest.fixture(scope="module")
def evidence():
    return read_json(PUBLIC)


def _assert_sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _assert_condition_chain(example, public):
    task = validate_task(example / "task_manifest.json", require_executable=True)
    curator = read_json(example / "curator_manifest.json")
    protocol = read_json(example / "protocol.json")
    curation = read_json(example / "curation.json")
    execution, authoring = public["execution"], public["authoring"]
    task_hash = digest(canonical_json(task))
    assert task_hash == curator["task_manifest_sha256"] == protocol["task_manifest_sha256"] == execution["task_sha256"]
    assert curator["protocol"] == protocol
    assert digest(canonical_json(protocol)) == execution["protocol_sha256"]
    code_hash = digest((example / task["entrypoint"]).read_bytes())
    assert code_hash == task["files"][task["entrypoint"]] == protocol["submission_sha256"]
    assert code_hash == execution["submission_sha256"] == curation["candidate_sha256"]
    assert freeze_protocol(protocol, task_manifest=task, submission_sha256=code_hash).sha256 == execution["protocol_sha256"]
    input_hash = digest((example / task["inputs"]).read_bytes())
    assert input_hash == task["files"][task["inputs"]] == protocol["datasets"][task["inputs"]]
    assert read_json(example / task["inputs"]) == execution["parameters"]
    assert digest(canonical_json(execution["parameters"])) == curator["reference"]["inputs_sha256"]
    assert digest(canonical_json(curation)) == task["source"]["curation_sha256"] == authoring["curation_record_sha256"]
    assert task["source"]["paper_sha256"] == protocol["source"]["paper_sha256"] == PAPER_SHA256
    assert public["paper"]["pdf_sha256"] == curation["source_paper_sha256"] == PAPER_SHA256
    assert task["source"]["authoring_run_id"] == curation["authoring_run_id"] == authoring["run_id"]
    source_review = (example / curation["review_evidence"]).read_text(encoding="utf-8")
    for reviewed_hash in (code_hash, input_hash, PAPER_SHA256, curation["draft_config_sha256"],
                          curation["draft_instruction_sha256"], curation["draft_plan_sha256"]):
        assert reviewed_hash in source_review
    for review in protocol["reviews"]:
        assert review["status"] == "pass" and review["submission_sha256"] == code_hash
        assert review["paper_sha256"] == PAPER_SHA256
        assert set(review["source_citations"]) <= {citation["id"] for citation in protocol["source"]["citations"]}
    assert execution["scorer_protocol"] == SCORER_PROTOCOLS[curator["scorer"]]
    return task, curator, protocol, curation


def _assert_prediction_metrics(public, raw, curator):
    execution = public["execution"]
    children = execution["children"]
    expected_hash = digest(raw)
    assert all(child["predictions_sha256"] == expected_hash for child in children)
    predictions = json.loads(raw)
    measured = score(curator["scorer"], execution["parameters"], predictions, curator["reference"])
    assert measured["verdict"] == "pass" and measured["observation_valid"] is True
    assert all(child["metric"] == measured for child in children)
    assert execution["steps"] == measured["samples"] == execution["parameters"]["steps"]
    assert execution["agreement"]["verdict"] == "pass"
    assert execution["agreement"]["reference"] == curator["reference"]["expected_gap"]
    assert execution["agreement"]["tolerance"] == curator["reference"]["tolerance"]
    assert abs(measured["value"] - execution["agreement"]["reference"]) <= execution["agreement"]["tolerance"]
    assert [metric["value"] for metric in public["metrics"]] == [
        measured["adam_average_regret"], measured["amsgrad_average_regret"], measured["value"], len(children)]
    expected_points = [{"step": point["horizon"],
                        "adam_position": predictions["adam"]["positions"][point["horizon"] - 1],
                        "amsgrad_position": predictions["amsgrad"]["positions"][point["horizon"] - 1],
                        "adam_average_regret": point["adam_average_regret"],
                        "amsgrad_average_regret": point["amsgrad_average_regret"]}
                       for point in trajectory_statistics(predictions)]
    assert public["checkpoints"] == expected_points
    return measured


def test_browser_script_embeds_exact_published_receipt(evidence):
    script = (ROOT / "docs/paper-result.js").read_text(encoding="utf-8")
    prefix, separator, encoded = script.partition("window.P2B_REAL_PAPER = ")
    assert separator and prefix.strip().startswith("// Recorded evidence;")
    assert encoded.endswith(";\n")
    payload = encoded[:-2]
    assert payload == PUBLIC.read_text(encoding="utf-8").rstrip()
    assert json.loads(payload) == evidence


def test_public_task_protocol_curation_and_review_form_one_hash_chain(evidence):
    _assert_condition_chain(EXAMPLE, evidence)


def test_recorded_authoring_provenance_and_draft_curation_are_explicit(evidence):
    curation = read_json(EXAMPLE / "curation.json")
    author = evidence["authoring"]
    assert author["model_calls"] == 4  # Recorded model calls; render is a local stage.
    for field in ("input_tokens", "output_tokens", "context_tokens"):
        assert type(author[field]) is int and author[field] > 0
    for public_key, curation_key in (("model", "model"), ("provider", "provider"),
                                     ("model_digest", "model_digest"),
                                     ("source_commit", "authoring_source_commit"),
                                     ("pipeline_sha256", "authoring_pipeline_sha256")):
        assert author[public_key] == curation[curation_key]
    assert author["provider"] == "ollama" and author["model"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", author["model_digest"])
    assert re.fullmatch(r"[0-9a-f]{40}", author["source_commit"])
    _assert_sha(author["pipeline_sha256"])
    stages = {stage["label"].lower(): stage for stage in author["stages"]}
    assert set(stages) == {"parse", "classify", "extract", "render", "plan"}
    assert len(author["stages"]) == len(stages)
    assert all(stage["status"] == "succeeded" for stage in stages.values())
    assert {name: stage["receipt_sha256"] for name, stage in stages.items()} == curation["authoring_stage_receipts"]
    for stage in stages.values():
        _assert_sha(stage["receipt_sha256"])
    assert curation["draft_status"] == "needs_review"
    assert curation["executed_model_draft"] is False
    assert author["curation"]["executed_model_draft"] is False
    assert author["curation"]["required"] is True and author["curation"]["corrections"]


def test_published_raw_trace_recomputes_all_metrics_and_every_seed(evidence):
    _, curator, protocol, _ = _assert_condition_chain(EXAMPLE, evidence)
    measured = _assert_prediction_metrics(evidence, PREDICTIONS.read_bytes(), curator)
    execution = evidence["execution"]
    children = execution["children"]
    assert execution["seed_schedule"] == protocol["plan"]["seed_schedule"] == [child["seed"] for child in children]
    assert len(children) == len({child["run_id"] for child in children}) == 3
    assert protocol["plan"]["independence_assumption"] is False
    assert execution["cleanup_confirmed"] is True and execution["network"] == "none"
    for child in children:
        assert child["status"] == "succeeded" and child["cleanup_confirmed"] is True
        assert child["isolation"] == "single_user_linux_container" and child["network"] == "none"
        assert child["image_id"] == execution["image_id"]
        assert 0 < child["duration_seconds"] <= 30
        _assert_sha(child["report_sha256"])
    _assert_sha(execution["scientific_report_sha256"])
    assert measured["adam_average_regret"] >= 2 / 3


def test_rewritten_prediction_hash_cannot_hide_a_metric_preserving_fabrication(evidence):
    output = json.loads(PREDICTIONS.read_bytes())
    # The same loss shift cancels in the paired scalar, but the algorithm is wrong.
    for method in ("adam", "amsgrad"):
        output[method]["positions"][1] -= 0.01
    altered = json.dumps(output, separators=(",", ":")).encode()
    public = deepcopy(evidence)
    for child in public["execution"]["children"]:
        child["predictions_sha256"] = digest(altered)
    curator = read_json(EXAMPLE / "curator_manifest.json")
    observed = score(curator["scorer"], public["execution"]["parameters"], output, curator["reference"])
    assert observed["value"] == pytest.approx(evidence["metrics"][2]["value"], abs=1e-14)
    with pytest.raises(AssertionError):
        _assert_prediction_metrics(public, altered, curator)


def test_changed_curation_is_rejected_even_with_intact_code_and_input(tmp_path, evidence):
    example = tmp_path / "example"
    shutil.copytree(EXAMPLE, example)
    changed = read_json(example / "curation.json")
    changed["executed_model_draft"] = True
    (example / "curation.json").write_bytes(canonical_json(changed))
    with pytest.raises(AssertionError):
        _assert_condition_chain(example, evidence)
