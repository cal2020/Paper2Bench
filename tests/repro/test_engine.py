"""Whole-core behavior with synthetic data and mock model calls only."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paper2bench.repro.authoring import EvidenceValidationError
from paper2bench.repro.cli import main
from paper2bench.repro.demo import create_demo
from paper2bench.repro.engine import Engine, InvalidExperiment
from paper2bench.repro.models import ExperimentSpec, Predictions
from paper2bench.repro.providers import Generation, OutcomeUnknown, UnsupportedCapability
from paper2bench.repro.runtime import (
    ExecutionFailure,
    ExecutionResult,
    ResourceExhausted,
    RuntimeUnavailable,
)
from paper2bench.repro.storage import ArtifactNotFound, IdempotencyConflict


@pytest.fixture
def engine(tmp_path):
    return Engine(tmp_path / "state")


def model_spec(engine, *, count=4, **updates):
    """A tiny known rule, without relying on a provider or a real paper."""
    base = engine.load_spec(create_demo(engine)).model_dump(mode="json")
    dataset = {
        "train": [
            {"id": "train-negative", "x": [-1], "y": "a"},
            {"id": "train-positive", "x": [1], "y": "b"},
        ],
        "test": [
            {
                "id": f"heldout-{index}",
                "x": [-2 if index % 2 == 0 else 2],
                "y": "a" if index % 2 == 0 else "b",
            }
            for index in range(count)
        ],
    }
    profile_hash = engine.register_profile(
        {
            "name": "mock-local",
            "location": "local",
            "base_url": "http://127.0.0.1:9999/v1",
        }
    )
    base.update(
        {
            "dataset_sha256": engine.register_dataset(dataset),
            "workload": {
                "kind": "model_classification",
                "provider_profile": profile_hash,
                "model": "mock-requested",
                "prompt": "Classify using the training examples.",
            },
            "policy": {"local_only": True, "max_provider_calls": count, "max_output_tokens": 16},
        }
    )
    base.update(updates)
    return ExperimentSpec.model_validate(base)


def assessment(engine, run):
    return engine.result(run["run_id"])["result"]["assessment"]


def test_builtin_demo_finishes_with_trusted_score_and_unverified_fidelity(engine):
    spec_hash = create_demo(engine)
    run = engine.run(spec_hash, "whole-demo")
    assert run["state"] == "completed"
    assert run["attempt"] == 1
    record = engine.result(run["run_id"])
    result = record["result"]
    assert result["spec_hash"] == spec_hash
    assert result["execution_state"] == "completed"
    assert len(result["predictions"]["ids"]) == 80
    assert result["assessment"]["accuracy"] == 1
    assert result["assessment"]["sample_count"] == 80
    assert result["assessment"]["verdict"] == "supported"
    assert result["assessment"]["criterion"] == "accuracy_equivalence"
    assert result["assessment"]["implementation_fidelity"] == "unverified"
    assert result["assessment"]["protocol_fidelity"] == "unverified"
    assert result["provider_calls"] == []
    assert "synthetic" in " ".join(result["assumptions"]).lower()
    assert "Paper reproduction: not established" in engine.report(run["run_id"])
    assert [event["kind"] for event in record["events"]] == ["created", "claimed", "finished"]


def test_builtin_cli_demo_is_usable_and_persists_state(tmp_path, capsys):
    root = tmp_path / "cli-state"
    assert main(["--state", str(root), "demo", "--key", "cli-demo"]) == 0
    output = capsys.readouterr()
    assert "Execution: completed" in output.out
    assert "Accuracy: 1.000000" in output.out
    assert output.err == ""
    retained = Engine(root).runs.list_runs()
    assert len(retained) == 1
    assert retained[0]["state"] == "completed"


def test_idempotent_run_never_reexecutes_completed_work(engine, monkeypatch):
    spec_hash = create_demo(engine)
    first = engine.run(spec_hash, "stable-key")

    def forbidden(*args, **kwargs):
        pytest.fail("a completed idempotent run must never execute again")

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", forbidden)
    second = engine.run(spec_hash, "stable-key")
    reopened = Engine(engine.root).execute(first["run_id"])
    assert second == first == reopened
    assert len(engine.runs.list_runs()) == 1
    assert len(first["attempts"]) == 1


def test_frozen_identity_changes_with_protocol_and_conflicting_key_is_rejected(engine):
    original_hash = create_demo(engine)
    assert create_demo(engine) == original_hash
    original_bytes = engine.artifacts.get(original_hash)
    changed = engine.load_spec(original_hash).model_dump(mode="json")
    changed["claim"]["tolerance"] = 0.2
    changed_hash = engine.freeze(changed)
    assert changed_hash != original_hash
    assert engine.artifacts.get(original_hash) == original_bytes
    engine.submit(original_hash, "one-contract")
    with pytest.raises(IdempotencyConflict):
        engine.submit(changed_hash, "one-contract")


def test_noncanonical_spec_artifact_cannot_be_executed_as_frozen_spec(engine):
    spec = engine.load_spec(create_demo(engine))
    reference = engine.artifacts.put(json.dumps(spec.model_dump(mode="json"), indent=2).encode())
    with pytest.raises(InvalidExperiment, match="canonical"):
        engine.load_spec(reference)


@pytest.mark.parametrize("invalid", ["quote", "unresolved"])
def test_freeze_rejects_source_quote_mismatch_and_unresolved_settings(engine, invalid):
    spec = engine.load_spec(create_demo(engine)).model_dump(mode="json")
    if invalid == "quote":
        quote = spec["evidence"][0]["quote"]
        spec["evidence"][0]["quote"] = "X" + quote[1:]
    else:
        spec["unresolved"] = ["The training split is unknown"]
    with pytest.raises(EvidenceValidationError):
        engine.freeze(spec)
    assert engine.runs.list_runs() == []


def test_freeze_revalidates_model_instances_from_trusted_api_copy(engine):
    spec = engine.load_spec(create_demo(engine))
    # Pydantic model_copy intentionally bypasses validation. A public boundary
    # must not mistake such an instance for a validated frozen specification.
    invalid = spec.model_copy(update={"claim": spec.claim.model_copy(update={"target": 2.0})})
    with pytest.raises((ValidationError, InvalidExperiment)):
        engine.freeze(invalid)


def test_missing_source_fails_before_a_run_is_created(engine):
    spec = engine.load_spec(create_demo(engine)).model_dump(mode="json")
    spec["source_sha256"] = "e" * 64
    with pytest.raises(ArtifactNotFound):
        engine.freeze(spec)
    assert engine.runs.list_runs() == []


def test_missing_artifact_after_submission_is_operational_failure_not_refutation(engine):
    spec_hash = create_demo(engine)
    run = engine.submit(spec_hash, "missing-after-submit")
    reference = engine.load_spec(spec_hash).dataset_sha256
    (engine.artifacts.root / reference[:2] / reference[2:]).unlink()
    executed = engine.execute(run["run_id"])
    assert executed["state"] in {"failed", "invalid"}
    assert assessment(engine, executed)["verdict"] == "not_tested"


@pytest.mark.parametrize("bad_output", ["missing_id", "unknown_label"])
def test_candidate_predictions_are_validated_before_scoring(engine, monkeypatch, bad_output):
    spec_hash = create_demo(engine)

    def malformed(self, dataset, spec, **kwargs):
        ids = tuple(example.id for example in dataset.test)
        labels = tuple(example.y for example in dataset.test)
        if bad_output == "missing_id":
            ids, labels = ids[:-1], labels[:-1]
        else:
            labels = ("not-a-class",) + labels[1:]
        return ExecutionResult(Predictions(ids=ids, labels=labels), {"claimed_accuracy": 1})

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", malformed)
    run = engine.run(spec_hash, "invalid-output")
    assert run["state"] == "invalid"
    assert assessment(engine, run)["verdict"] == "not_tested"
    assert "accuracy" not in assessment(engine, run)


@pytest.mark.parametrize(
    "error,state",
    [
        (ExecutionFailure, "failed"),
        (RuntimeUnavailable, "unsupported"),
        (UnsupportedCapability, "unsupported"),
        (ResourceExhausted, "resource_exhausted"),
        (RuntimeError, "failed"),
    ],
)
def test_operational_failures_never_become_scientific_verdicts(engine, monkeypatch, error, state):
    spec_hash = create_demo(engine)

    def fail(*args, **kwargs):
        raise error("secret-token=do-not-persist provider-private-text")

    monkeypatch.setattr("paper2bench.repro.engine.BuiltinRunner.run", fail)
    run = engine.run(spec_hash, "failed-candidate")
    record = engine.result(run["run_id"])
    assert run["state"] == state
    assert record["result"]["assessment"]["verdict"] == "not_tested"
    assert "accuracy" not in record["result"]["assessment"]
    assert "do-not-persist" not in json.dumps(record)
    assert "provider-private-text" not in json.dumps(record)


def test_mock_model_receives_training_labels_but_no_test_labels(engine, monkeypatch):
    spec = model_spec(engine)
    seen = []

    def generate(self, *, model, messages, max_output_tokens, local_only, total_timeout):
        payload = json.loads(messages[1]["content"])
        assert set(payload["test"]) == {"id", "x"}
        assert "y" not in payload["test"]
        assert all(set(row) == {"id", "x", "y"} for row in payload["train"])
        assert payload["allowed_labels"] == ["a", "b"]
        assert model == "mock-requested"
        assert max_output_tokens == 16 and local_only is True
        assert 0 < total_timeout <= spec.limits.timeout_seconds
        seen.append(payload)
        label = "a" if payload["test"]["x"][0] < 0 else "b"
        return Generation(
            text=json.dumps({"label": label}),
            model="mock-actual-revision",
            request_id=f"request-{len(seen)}",
            fingerprint="mock-fp",
        )

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", generate)
    run = engine.run(engine.freeze(spec), "mock-model")
    assert run["state"] == "completed"
    result = engine.result(run["run_id"])["result"]
    assert result["assessment"]["accuracy"] == 1
    assert len(seen) == 4
    assert len(result["provider_calls"]) == 4
    assert result["measurements"]["cost_usd"] is None
    for receipt in result["provider_calls"]:
        assert receipt["model"] == "mock-actual-revision"
        assert receipt["fingerprint"] == "mock-fp"
        assert "text" not in receipt
        assert set(json.loads(engine.artifacts.get(receipt["output_sha256"]))) == {"label"}


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        "[]",
        '{"label": 1}',
        '{"label":"unknown"}',
        '{"label":"a","score":1}',
        '{"label":"b","label":"a"}',
    ],
)
def test_malformed_model_classifications_fail_without_retry(engine, monkeypatch, text):
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        return Generation(text=text, model="mock")

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", generate)
    run = engine.run(engine.freeze(model_spec(engine)), "malformed-model")
    assert run["state"] == "invalid"
    assert assessment(engine, run)["verdict"] == "not_tested"
    assert len(calls) == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_truncated_or_filtered_model_output_is_never_scored(engine, monkeypatch, finish_reason):
    def generate(*args, **kwargs):
        return Generation(text='{"label":"a"}', model="mock", finish_reason=finish_reason)

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", generate)
    run = engine.run(engine.freeze(model_spec(engine, count=1)), "incomplete-model")
    assert run["state"] == "invalid"
    assert assessment(engine, run)["verdict"] == "not_tested"


def test_unknown_model_timeout_is_terminal_and_not_automatically_repeated(engine, monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise OutcomeUnknown("remote-secret-do-not-persist", request_id="maybe-accepted")

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", timeout)
    spec_hash = engine.freeze(model_spec(engine))
    run = engine.run(spec_hash, "unknown-dispatch")
    assert run["state"] == "outcome_unknown"
    assert assessment(engine, run)["verdict"] == "not_tested"
    assert "remote-secret" not in json.dumps(engine.result(run["run_id"]))
    assert engine.execute(run["run_id"]) == run
    assert engine.run(spec_hash, "unknown-dispatch") == run
    assert len(calls) == 1


def test_crashed_model_lease_cannot_repeat_possibly_billable_requests(engine, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    spec_hash = engine.freeze(model_spec(engine))
    run = engine.submit(spec_hash, "crashed-model")
    previous = engine.runs.claim(run["run_id"], "crashed-worker", lease_seconds=10)
    now[0] = 1011

    def forbidden(*args, **kwargs):
        pytest.fail("an expired model attempt must not repeat unacknowledged requests")

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", forbidden)
    resumed = engine.execute(run["run_id"], worker="replacement")
    assert resumed["state"] == "outcome_unknown"
    assert resumed["attempt"] == 2
    assert [item["state"] for item in resumed["attempts"]] == ["expired", "outcome_unknown"]
    assert assessment(engine, resumed)["verdict"] == "not_tested"
    assert not engine.runs.finish(run["run_id"], previous["token"], "failed")


def test_last_model_response_after_wall_limit_is_not_scored(engine, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr("paper2bench.repro.engine.time.monotonic", lambda: elapsed[0])
    spec = model_spec(engine, count=1, limits={"timeout_seconds": 1})

    def late(*args, **kwargs):
        elapsed[0] = 2.0
        return Generation(text='{"label":"a"}', model="mock")

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", late)
    run = engine.run(engine.freeze(spec), "late-last-response")
    assert run["state"] == "resource_exhausted"
    assert assessment(engine, run)["verdict"] == "not_tested"


def test_model_budget_insufficient_for_test_set_rejected_before_dispatch(engine, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight budget failure cannot dispatch a model request")

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", forbidden)
    spec = model_spec(engine, policy={"local_only": True, "max_provider_calls": 1})
    with pytest.raises(InvalidExperiment, match="budget"):
        engine.freeze(spec)
