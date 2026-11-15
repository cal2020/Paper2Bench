"""Remote workers receive public inputs; all scores remain server decisions."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from paper2bench.repro.engine import Engine
from paper2bench.repro.models import ExperimentSpec, digest
from paper2bench.repro.remote import RemoteConflict, RemoteCoordinator

IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "b" * 64
CODE = b"import json,sys\ndata=json.load(sys.stdin)\nprint(json.dumps({'ids':[], 'labels':[]}))\n"
SOURCE = "PRIVATE PAPER CONTENT. A centroid classifier reports accuracy 1.0."


@pytest.fixture
def setup(tmp_path):
    engine = Engine(tmp_path / "state")
    coordinator = RemoteCoordinator(engine)
    return engine, coordinator


def submit(engine, key="job", *, placement="remote", memory=256, task="classification"):
    if task == "classification":
        dataset = {
            "train": [
                {"id": "train-a", "x": [-1.0], "y": "negative"},
                {"id": "train-b", "x": [1.0], "y": "positive"},
            ],
            "test": [
                {
                    "id": f"test-{i}",
                    "x": [-2.0 if i % 2 == 0 else 2.0],
                    "y": "negative" if i % 2 == 0 else "positive",
                }
                for i in range(80)
            ],
        }
        claim = {"metric": "accuracy", "target": 1.0}
    else:
        dataset = {
            "task": "knapsack",
            "instances": [
                {"id": f"instance-{i}", "weights": [2, 3], "values": [3, 5], "capacity": 3}
                for i in range(20)
            ],
        }
        claim = {"metric": "optimality_ratio", "target": 1.0}
    spec = ExperimentSpec.model_validate(
        {
            "title": "remote task",
            "source_sha256": engine.artifacts.put(SOURCE.encode()),
            "dataset_sha256": engine.register_dataset(dataset),
            "evidence": [{"start": 0, "end": len(SOURCE), "quote": SOURCE}],
            "claim": claim,
            "workload": {
                "kind": "python",
                "implementation_sha256": engine.artifacts.put(CODE),
                "image": IMAGE,
            },
            "limits": {"memory_mb": memory},
            "policy": {"local_only": False, "placement": placement},
        }
    )
    return engine.submit(engine.freeze(spec), key)


def claim(coordinator, worker="worker-a", **changes):
    return coordinator.claim(
        worker, changes.pop("images", (IMAGE,)), changes.pop("max_memory_mb", 1024), **changes
    )


def completed(assignment, **changes):
    if assignment["task"] == "classification":
        output = {
            "ids": [item["id"] for item in assignment["input"]["test"]],
            "labels": [
                "negative" if item["x"][0] < 0 else "positive"
                for item in assignment["input"]["test"]
            ],
        }
    else:
        output = {
            "ids": [item["id"] for item in assignment["input"]["instances"]],
            "selections": [[1] for _ in assignment["input"]["instances"]],
        }
    value = {
        "state": "completed",
        "input_sha256": assignment["input_sha256"],
        "implementation_sha256": assignment["implementation_sha256"],
        "image": assignment["image"],
        "predictions": output,
        "measurements": {
            "runtime": "docker",
            "image": IMAGE,
            "wall_seconds": 0.1,
            "memory_limit_enforced": True,
            "network_used": False,
            "cleanup_confirmed": True,
        },
    }
    value.update(changes)
    return value


def complete(coordinator, assignment, value=None, worker="worker-a"):
    return coordinator.complete(
        worker,
        assignment["run_id"],
        assignment["token"],
        completed(assignment) if value is None else value,
    )


def test_assignment_has_exact_public_contract_and_stable_hashes(setup):
    engine, coordinator = setup
    run = submit(engine)
    assignment = claim(coordinator)
    assert assignment["run_id"] == run["run_id"]
    assert assignment["worker_id"] == "worker-a"
    assert set(assignment) == {
        "schema_version",
        "worker_id",
        "run_id",
        "token",
        "attempt",
        "lease_seconds",
        "spec_hash",
        "implementation_sha256",
        "input_sha256",
        "image",
        "limits",
        "seed",
        "task",
        "code",
        "input",
    }
    assert all(set(row) == {"id", "x"} for row in assignment["input"]["test"])
    assert all(set(row) == {"id", "x", "y"} for row in assignment["input"]["train"])
    assert "PRIVATE PAPER" not in str(assignment)
    assert assignment["implementation_sha256"] == engine.artifacts.put(assignment["code"].encode())
    assert assignment["input_sha256"] == digest(assignment["input"])
    assert engine.artifacts.get_json(assignment["input_sha256"]) == assignment["input"]
    assert claim(coordinator) == assignment
    assert len(engine.runs.get_run(run["run_id"])["attempts"]) == 1


def test_only_explicit_remote_python_jobs_with_advertised_resources_are_claimed(setup):
    engine, coordinator = setup
    local = submit(engine, "local", placement="local")
    remote = submit(engine, "remote", memory=512)
    assert claim(coordinator, images=(OTHER_IMAGE,)) is None
    assert claim(coordinator, max_memory_mb=256) is None
    assignment = claim(coordinator, max_memory_mb=512)
    assert assignment["run_id"] == remote["run_id"]
    assert engine.runs.get_run(local["run_id"])["state"] == "queued"


def test_concurrent_claims_for_one_worker_do_not_allocate_multiple_jobs(setup):
    engine, coordinator = setup
    submit(engine, "first")
    submit(engine, "second")
    with ThreadPoolExecutor(max_workers=6) as pool:
        assignments = list(pool.map(lambda _: claim(coordinator), range(12)))
    assert len({item["run_id"] for item in assignments}) == 1
    assert len({item["token"] for item in assignments}) == 1
    assert sum(run["state"] == "running" for run in engine.runs.list_runs()) == 1
    different = claim(coordinator, worker="worker-b")
    assert different["run_id"] != assignments[0]["run_id"]


def test_server_computes_score_and_marks_remote_provenance_unverified(setup):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    run = complete(coordinator, assignment)
    assert run["state"] == "completed"
    result = engine.result(run["run_id"])["result"]
    assert result["assessment"]["accuracy"] == 1.0
    assert result["assessment"]["implementation_fidelity"] == "unverified"
    assert result["measurements"]["measurement_source"] == "remote_worker_reported"
    assert result["measurements"]["runtime_attestation"] == "not_verified"
    assert result["provider_calls"] == []
    assert engine.artifacts.get_json(result["remote_completion_sha256"]) == completed(assignment)
    assert "not attested" in " ".join(result["limitations"])


def test_successful_and_conflicting_duplicate_completion_after_response_loss(setup):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    original = complete(coordinator, assignment)
    reopened = RemoteCoordinator(Engine(engine.root))
    assert complete(reopened, assignment) == original
    changed = completed(assignment)
    changed["measurements"]["wall_seconds"] = 0.2
    with pytest.raises(RemoteConflict):
        complete(reopened, assignment, changed)
    assert engine.runs.get_run(original["run_id"]) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "c" * 64),
        ("implementation_sha256", "c" * 64),
        ("image", OTHER_IMAGE),
    ],
)
def test_identity_mismatches_rejected_without_publishing(setup, field, value):
    engine, coordinator = setup
    run = submit(engine)
    assignment = claim(coordinator)
    with pytest.raises(RemoteConflict):
        complete(coordinator, assignment, completed(assignment, **{field: value}))
    assert engine.runs.get_run(run["run_id"])["result_hash"] is None
    assert engine.runs.is_active(run["run_id"], assignment["token"])


def test_wrong_owner_cannot_heartbeat_complete_or_read_idempotent_result(setup):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    assert not coordinator.heartbeat("worker-b", assignment["run_id"], assignment["token"])
    with pytest.raises(RemoteConflict):
        complete(coordinator, assignment, worker="worker-b")
    complete(coordinator, assignment)
    with pytest.raises(RemoteConflict):
        complete(coordinator, assignment, worker="worker-b")


def test_expired_lease_resume_fences_late_remote_results(setup, monkeypatch):
    engine, coordinator = setup
    now = [1000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    submit(engine)
    first = claim(coordinator, lease_seconds=5)
    now[0] = 1005
    assert not coordinator.heartbeat("worker-a", first["run_id"], first["token"])
    with pytest.raises(RemoteConflict):
        complete(coordinator, first)
    second = claim(coordinator, worker="worker-b", lease_seconds=10)
    assert second["token"] > first["token"] and second["attempt"] == 2
    with pytest.raises(RemoteConflict):
        complete(coordinator, first)
    run = complete(coordinator, second, worker="worker-b")
    assert [item["state"] for item in run["attempts"]] == ["expired", "completed"]


def test_admin_cancellation_cannot_be_overwritten_by_any_worker_completion(setup):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    cancelled = engine.runs.cancel(assignment["run_id"])
    with pytest.raises(RemoteConflict):
        complete(coordinator, assignment)
    value = completed(assignment, state="cancelled", predictions=None, error="worker_cancelled")
    with pytest.raises(RemoteConflict):
        complete(coordinator, assignment, value)
    assert engine.runs.get_run(assignment["run_id"]) == cancelled


@pytest.mark.parametrize("state", ["failed", "unsupported", "resource_exhausted", "cancelled"])
def test_unsuccessful_remote_work_preserves_not_tested_and_duplicate_receipts(setup, state):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    value = completed(assignment, state=state, predictions=None, error="worker_execution_failed")
    run = complete(coordinator, assignment, value)
    assert run["state"] == state
    assert complete(coordinator, assignment, value) == run
    record = engine.result(run["run_id"])
    result = record["result"] or record["attempt_diagnostics"][-1]["details"]
    assert result["assessment"]["verdict"] == "not_tested"
    assert "accuracy" not in result["assessment"]
    if state == "cancelled":
        assert run["token"] == assignment["token"] + 1


@pytest.mark.parametrize("wrong", ["missing_id", "unknown_label"])
def test_malformed_predictions_become_invalid_not_a_scientific_failure(setup, wrong):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    value = completed(assignment)
    if wrong == "missing_id":
        value["predictions"]["ids"].pop()
        value["predictions"]["labels"].pop()
    else:
        value["predictions"]["labels"][0] = "invented"
    run = complete(coordinator, assignment, value)
    assert run["state"] == "invalid"
    assert engine.result(run["run_id"])["result"]["assessment"]["verdict"] == "not_tested"
    assert complete(coordinator, assignment, value) == run


@pytest.mark.parametrize(
    "field,value",
    [
        ("measurements", {"accuracy": 1.0}),
        ("measurements", {"api_key": "secret"}),
        ("measurements", {"wall_seconds": float("nan")}),
        ("error", "Bearer secret"),
        ("assessment", {"verdict": "supported"}),
    ],
)
def test_untrusted_score_fields_raw_errors_and_secrets_are_rejected(setup, field, value):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    with pytest.raises((ValueError, ValidationError)):
        complete(coordinator, assignment, completed(assignment, **{field: value}))
    assert engine.runs.get_run(assignment["run_id"])["result_hash"] is None


def test_remote_knapsack_outputs_are_scored_by_server_exact_solver(setup):
    engine, coordinator = setup
    submit(engine, task="knapsack")
    assignment = claim(coordinator)
    assert assignment["task"] == "knapsack"
    assert set(assignment["input"]) == {"instances", "seed"}
    assert "optimum" not in str(assignment["input"])
    run = complete(coordinator, assignment)
    result = engine.result(run["run_id"])["result"]
    assert run["state"] == "completed"
    assert result["assessment"]["metric"] == "optimality_ratio"
    assert result["assessment"]["mean_score"] == 1.0


def test_prediction_output_size_is_enforced_on_server(setup):
    engine, coordinator = setup
    run = submit(engine)
    spec = engine.load_spec(run["spec_hash"]).model_dump(mode="json")
    # Register a separate valid frozen job with a deliberately tiny output budget.
    spec["limits"]["output_bytes"] = 10
    engine.runs.cancel(run["run_id"])
    tiny = engine.submit(engine.freeze(spec), "tiny-output")
    assignment = claim(coordinator)
    assert assignment["run_id"] == tiny["run_id"]
    finished = complete(coordinator, assignment)
    assert finished["state"] == "resource_exhausted"
    assert engine.result(tiny["run_id"])["result"]["assessment"]["verdict"] == "not_tested"


def failed_cleanup(assignment, **updates):
    value = completed(
        assignment,
        state="cancelled",
        predictions=None,
        error="lease_lost_or_cancelled",
        measurements={"cleanup_confirmed": False, "container_name": "paper2bench-" + "f" * 32},
    )
    value.update(updates)
    return value


def test_stale_known_owner_failure_attaches_cleanup_without_accepting_completion(setup):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    cancelled = engine.runs.cancel(assignment["run_id"])
    for _ in range(2):
        with pytest.raises(RemoteConflict):
            complete(coordinator, assignment, failed_cleanup(assignment))
    assert engine.runs.get_run(assignment["run_id"]) == cancelled
    diagnostics = engine.result(assignment["run_id"])["attempt_diagnostics"]
    assert len(diagnostics) == 1  # Duplicate stale reports remain conflicts and deduplicate.
    diagnostic = diagnostics[0]
    assert diagnostic["token"] == assignment["token"]
    assert diagnostic["details"]["diagnostic_only"] is True
    assert diagnostic["details"]["assessment"]["verdict"] == "not_tested"
    assert "predictions" not in diagnostic["details"]
    assert diagnostic["details"]["measurements"]["cleanup_confirmed"] is False
    assert diagnostic["details"]["measurements"]["container_name"].endswith("f" * 32)
    assert engine.artifacts.get_json(diagnostic["details"]["remote_completion_sha256"])


def test_expired_failure_cleanup_never_modifies_successor_or_completed_score(setup, monkeypatch):
    engine, coordinator = setup
    now = [2000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    submit(engine)
    first = claim(coordinator, lease_seconds=5)
    now[0] = 2005
    second = claim(coordinator, worker="worker-b")
    active = engine.runs.get_run(first["run_id"])
    with pytest.raises(RemoteConflict):
        complete(coordinator, first, failed_cleanup(first))
    assert engine.runs.get_run(first["run_id"]) == active
    accepted = complete(coordinator, second, worker="worker-b")
    with pytest.raises(RemoteConflict):
        complete(coordinator, first, failed_cleanup(first))
    assert engine.runs.get_run(first["run_id"]) == accepted
    assert engine.result(first["run_id"])["result"]["assessment"]["accuracy"] == 1.0


@pytest.mark.parametrize(
    "mutation",
    ["wrong_worker", "unknown_token", "wrong_input", "wrong_code", "wrong_image", "predictions"],
)
def test_stale_cleanup_requires_known_owner_and_frozen_identity(setup, mutation):
    engine, coordinator = setup
    submit(engine)
    assignment = claim(coordinator)
    engine.runs.cancel(assignment["run_id"])
    value = failed_cleanup(assignment)
    worker, token = "worker-a", assignment["token"]
    if mutation == "wrong_worker":
        worker = "worker-b"
    elif mutation == "unknown_token":
        token += 100
    elif mutation == "wrong_input":
        value["input_sha256"] = "c" * 64
    elif mutation == "wrong_code":
        value["implementation_sha256"] = "c" * 64
    elif mutation == "wrong_image":
        value["image"] = OTHER_IMAGE
    else:
        value["predictions"] = completed(assignment)["predictions"]
    with pytest.raises(ValueError):
        coordinator.complete(worker, assignment["run_id"], token, value)
    assert engine.runs.diagnostics(assignment["run_id"]) == []
