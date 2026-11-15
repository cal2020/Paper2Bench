"""Remote delivery exercises real HTTP serialization with mocked isolated execution."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from paper2bench.repro.engine import Engine
from paper2bench.repro.models import (
    ExperimentSpec,
    Limits,
    OptimizationPredictions,
    Predictions,
    canonical_bytes,
)
from paper2bench.repro.remote import RemoteCoordinator
from paper2bench.repro.remote_client import WorkerClient, WorkerClientError
from paper2bench.repro.runtime import ExecutionCancelled, ExecutionResult, RuntimeUnavailable

IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "b" * 64
RUN_ID = "1" * 32
SPEC_HASH = "2" * 64
SECRET = "worker-secret-marker-8172"
CODE = "import json,sys\np=json.load(sys.stdin)\nprint('{}')\n"
INPUT = {
    "train": [
        {"id": "train-neg", "x": [-1.0], "y": "neg"},
        {"id": "train-pos", "x": [1.0], "y": "pos"},
    ],
    "test": [{"id": "test-1", "x": [1.0]}],
    "seed": 7,
}


def assignment(**changes):
    value = {
        "schema_version": "1",
        "worker_id": "worker-1",
        "run_id": RUN_ID,
        "token": 1,
        "attempt": 1,
        "lease_seconds": 30.0,
        "spec_hash": SPEC_HASH,
        "implementation_sha256": hashlib.sha256(CODE.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(canonical_bytes(INPUT)).hexdigest(),
        "image": IMAGE,
        "limits": Limits().model_dump(mode="json"),
        "seed": 7,
        "task": "classification",
        "code": CODE,
        "input": json.loads(json.dumps(INPUT)),
    }
    value.update(changes)
    return value


class Coordinator:
    def __init__(self, work=None):
        self.work = work or assignment()
        self.requests = []
        self.claims = 0
        self.completions = []
        self.completion_failure = None
        self.heartbeat_active = True
        self.ack_changes = {}

    def handle(self, request):
        assert request.headers["authorization"] == "Bearer " + SECRET
        body = json.loads(request.content)
        assert SECRET not in request.content.decode()
        self.requests.append((request.url.path, body))
        if request.url.path == "/worker/claim":
            self.claims += 1
            assert body == {"worker_id": "worker-1", "images": [IMAGE], "max_memory_mb": 256}
            return httpx.Response(200, json={"assignment": self.work if self.claims == 1 else None})
        if request.url.path == "/worker/heartbeat":
            return httpx.Response(200, json={"active": self.heartbeat_active})
        assert request.url.path == "/worker/complete"
        self.completions.append(body)
        if self.completion_failure == "timeout" and len(self.completions) == 1:
            raise httpx.ReadTimeout("sensitive-server-error-" + SECRET, request=request)
        if self.completion_failure in {409, 410}:
            return httpx.Response(self.completion_failure, text="sensitive-server-error-" + SECRET)
        ack = {
            "id": RUN_ID,
            "run_id": RUN_ID,
            "worker": "remote:worker-1",
            "spec_hash": SPEC_HASH,
            "token": 2 if body["completion"]["state"] == "cancelled" else 1,
            "state": body["completion"]["state"],
        }
        ack.update(self.ack_changes)
        return httpx.Response(200, json=ack)


def client(tmp_path, monkeypatch, coordinator=None, **changes):
    monkeypatch.setenv("P2B_WORKER_TOKEN", SECRET)
    coordinator = coordinator or Coordinator()
    values = {
        "base_url": "http://127.0.0.1:12345",
        "worker_id": "worker-1",
        "token_env": "P2B_WORKER_TOKEN",
        "spool": tmp_path / "spool",
        "images": (IMAGE,),
        "transport": httpx.MockTransport(coordinator.handle),
    }
    values.update(changes)
    return WorkerClient(**values), coordinator


def fake_runtime(monkeypatch):
    calls = []

    def run_input(self, code, input_payload, spec, *, cancelled):
        calls.append((code, input_payload, spec))
        assert code == CODE.encode()
        assert input_payload == INPUT
        assert "y" not in input_payload["test"][0]
        assert not hasattr(spec, "claim")
        assert not hasattr(spec, "policy")
        assert spec.workload.image == IMAGE
        return ExecutionResult(
            predictions=Predictions(ids=("test-1",), labels=("pos",)),
            measurements={
                "runtime": "docker",
                "image": IMAGE,
                "task": "classification",
                "wall_seconds": 0.1,
                "cleanup_confirmed": True,
                "arbitrary_log": SECRET,
            },
        )

    monkeypatch.setattr("paper2bench.repro.remote_client.DockerRunner.run_input", run_input)
    return calls


def test_remote_executes_typed_input_and_submits_predictions_only(tmp_path, monkeypatch):
    calls = fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch)
    result = worker.run_once()
    assert result == {"state": "completed", "run_id": RUN_ID, "run_state": "completed"}
    assert len(calls) == 1
    completion = coordinator.completions[0]["completion"]
    assert completion["predictions"] == {"ids": ["test-1"], "labels": ["pos"]}
    assert "assessment" not in completion and "score" not in completion
    assert "arbitrary_log" not in completion["measurements"]
    assert completion["measurements"]["task"] == "classification"
    assert not (worker.spool / "pending.json").exists()
    assert not any(
        thread.name == "paper2bench-remote-heartbeat" for thread in threading.enumerate()
    )


def test_uncertain_submission_replays_durable_result_without_rerunning(tmp_path, monkeypatch):
    calls = fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch)
    coordinator.completion_failure = "timeout"
    first = worker.run_once()
    assert first["state"] == "pending_submission"
    path = worker.spool / "pending.json"
    raw = path.read_bytes()
    assert SECRET.encode() not in raw
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(worker.spool).st_mode & 0o777 == 0o700
    # A fresh client/process uses the same private spool and replays delivery only.
    restarted, _ = client(tmp_path, monkeypatch, coordinator)
    assert restarted.run_once()["state"] == "completed"
    assert len(calls) == 1
    assert coordinator.claims == 1
    assert len(coordinator.completions) == 2
    assert coordinator.completions[0] == coordinator.completions[1]
    assert not path.exists()


@pytest.mark.parametrize("status", [409, 410])
def test_fenced_submission_retained_without_infinite_retry(tmp_path, monkeypatch, status):
    calls = fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch)
    coordinator.completion_failure = status
    assert worker.run_once()["state"] == "fenced"
    assert not (worker.spool / "pending.json").exists()
    archives = list(worker.spool.glob("fenced-*.json"))
    assert len(archives) == 1
    assert SECRET not in archives[0].read_text()
    assert worker.run_once() == {"state": "idle"}
    assert len(calls) == 1 and len(coordinator.completions) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"code": CODE + "# tampered"},
        {"input_sha256": "f" * 64},
        {"worker_id": "other-worker"},
        {"image": OTHER_IMAGE},
        {"image": "python:latest"},
        {"limits": {"memory_mb": 512}},
        {"token": True},
        {"seed": 99},
        {"task": "knapsack"},
        {"schema_version": "2"},
        {"hidden_answers": ["pos"]},
    ],
)
def test_invalid_or_tampered_assignment_never_reaches_runtime(tmp_path, monkeypatch, change):
    calls = fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch, Coordinator(assignment(**change)))
    assert worker.run_once() == {"state": "error", "error": "invalid_remote_assignment"}
    assert calls == [] and coordinator.completions == []


def test_hidden_test_labels_rejected_even_with_matching_input_hash(tmp_path, monkeypatch):
    calls = fake_runtime(monkeypatch)
    work = assignment()
    work["input"]["test"][0]["y"] = "pos"
    work["input_sha256"] = hashlib.sha256(canonical_bytes(work["input"])).hexdigest()
    worker, _ = client(tmp_path, monkeypatch, Coordinator(work))
    assert worker.run_once()["error"] == "invalid_remote_assignment"
    assert calls == []


def test_lease_loss_cancels_runtime_and_stops_heartbeat_thread(tmp_path, monkeypatch):
    work = assignment(lease_seconds=0.03)
    coordinator = Coordinator(work)
    coordinator.heartbeat_active = False
    cancellation = []

    def run_input(self, code, input_payload, spec, *, cancelled):
        deadline = time.monotonic() + 1
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert cancelled()
        cancellation.append(True)
        raise ExecutionCancelled("untrusted candidate error " + SECRET)

    monkeypatch.setattr("paper2bench.repro.remote_client.DockerRunner.run_input", run_input)
    worker, _ = client(tmp_path, monkeypatch, coordinator)
    result = worker.run_once()
    assert result["state"] == "completed" and result["run_state"] == "cancelled"
    assert cancellation == [True]
    assert coordinator.completions[0]["completion"]["error"] == "lease_lost_or_cancelled"
    assert not any(
        thread.name == "paper2bench-remote-heartbeat" for thread in threading.enumerate()
    )


def test_unavailable_docker_is_reported_without_host_fallback(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeUnavailable("host-sensitive-error " + SECRET)

    monkeypatch.setattr("paper2bench.repro.remote_client.DockerRunner.run_input", unavailable)
    worker, coordinator = client(tmp_path, monkeypatch)
    assert worker.run_once()["run_state"] == "unsupported"
    completion = coordinator.completions[0]["completion"]
    assert completion["error"] == "runtime_unavailable"
    assert completion["predictions"] is None
    assert SECRET not in json.dumps(completion)


@pytest.mark.parametrize(
    "change",
    [
        {"worker": "remote:other"},
        {"run_id": "f" * 32},
        {"spec_hash": "f" * 64},
        {"token": 2},
        {"state": "running"},
    ],
)
def test_wrong_acknowledgement_does_not_discard_pending_result(tmp_path, monkeypatch, change):
    fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch)
    coordinator.ack_changes = change
    result = worker.run_once()
    assert result["state"] == "pending_submission"
    assert result["error"] == "submission_acknowledgement_invalid"
    assert (worker.spool / "pending.json").exists()


def test_pending_submission_integrity_checked_before_network(tmp_path, monkeypatch):
    calls = fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch)
    coordinator.completion_failure = "timeout"
    assert worker.run_once()["state"] == "pending_submission"
    path = worker.spool / "pending.json"
    pending = json.loads(path.read_bytes())
    pending["completion"]["predictions"]["labels"] = ["neg"]
    path.write_bytes(canonical_bytes(pending))
    before = len(coordinator.requests)
    assert worker.run_once() == {"state": "error", "error": "invalid_pending_submission"}
    assert len(coordinator.requests) == before and len(calls) == 1


def test_pending_result_cannot_be_replayed_to_another_coordinator(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    worker, coordinator = client(tmp_path, monkeypatch)
    coordinator.completion_failure = "timeout"
    assert worker.run_once()["state"] == "pending_submission"
    other, _ = client(tmp_path, monkeypatch, coordinator, base_url="https://other.invalid")
    before = len(coordinator.requests)
    assert other.run_once()["error"] == "pending_submission_owner_mismatch"
    assert len(coordinator.requests) == before


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.invalid",
        "http://localhost:12345",
        "http://192.168.1.1",
        "https://user:secret@example.invalid",
        "https://example.invalid?token=secret",
    ],
)
def test_remote_url_policy_rejects_unsafe_configuration(tmp_path, monkeypatch, base_url):
    with pytest.raises(WorkerClientError) as error:
        client(tmp_path, monkeypatch, base_url=base_url)
    assert str(error.value) == "invalid_worker_configuration"


def test_https_cloud_and_ipv6_loopback_supported(tmp_path, monkeypatch):
    worker, _ = client(tmp_path, monkeypatch, base_url="https://coordinator.invalid")
    assert worker.base_url == "https://coordinator.invalid"
    worker, _ = client(tmp_path, monkeypatch, base_url="http://[::1]:12345")
    assert worker.base_url == "http://[::1]:12345"


def test_redirect_refused_without_credential_forwarding(tmp_path, monkeypatch):
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://attacker.invalid"})

    worker, _ = client(tmp_path, monkeypatch, transport=httpx.MockTransport(redirect))
    assert worker.run_once()["state"] == "error"
    assert len(requests) == 1


def test_spool_lock_prevents_concurrent_worker_consumption(tmp_path, monkeypatch):
    worker, coordinator = client(tmp_path, monkeypatch)
    with worker._locked_spool():
        assert worker.run_once() == {"state": "error", "error": "worker_spool_busy"}
    assert coordinator.requests == []


def test_symlink_spool_refused_without_accessing_target(tmp_path, monkeypatch):
    target = tmp_path / "private"
    target.mkdir()
    spool = tmp_path / "spool"
    spool.symlink_to(target)
    worker, coordinator = client(tmp_path, monkeypatch)
    assert worker.run_once()["error"] == "unsafe_spool_directory"
    assert list(target.iterdir()) == [] and coordinator.requests == []


def test_oversized_response_rejected_before_runtime(tmp_path, monkeypatch):
    import paper2bench.repro.remote_client as remote_client

    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(remote_client, "MAX_BODY_BYTES", 1024)

    def oversized(request):
        return httpx.Response(200, content=b"x" * 2048)

    worker, _ = client(tmp_path, monkeypatch, transport=httpx.MockTransport(oversized))
    assert worker.run_once() == {"state": "error", "error": "remote_response_too_large"}
    assert calls == []


def test_duplicate_json_keys_rejected_before_runtime(tmp_path, monkeypatch):
    calls = fake_runtime(monkeypatch)
    worker, _ = client(
        tmp_path,
        monkeypatch,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'{"assignment":null,"assignment":null}')
        ),
    )
    assert worker.run_once() == {"state": "error", "error": "remote_response_invalid"}
    assert calls == []


@pytest.mark.parametrize("worker_id", ["-worker", ".worker", "worker:name", "worker/name"])
def test_worker_id_matches_coordinator_protocol(tmp_path, monkeypatch, worker_id):
    with pytest.raises(WorkerClientError, match="invalid_worker_configuration"):
        client(tmp_path, monkeypatch, worker_id=worker_id)


def test_http_worker_roundtrip_with_real_coordinator_and_trusted_solver_score(
    tmp_path, monkeypatch
):
    engine = Engine(tmp_path / "coordinator")
    coordinator = RemoteCoordinator(engine)
    source = "The solver selects a feasible subset maximizing item value. Reported ratio is 1.0."
    dataset = {
        "task": "knapsack",
        "instances": [
            {"id": f"instance-{i}", "weights": [2, 3], "values": [3, 5], "capacity": 3}
            for i in range(20)
        ],
    }
    spec = ExperimentSpec.model_validate(
        {
            "title": "HTTP remote solver",
            "source_sha256": engine.artifacts.put(source.encode()),
            "dataset_sha256": engine.register_dataset(dataset),
            "evidence": [{"start": 0, "end": len(source), "quote": source}],
            "claim": {"metric": "optimality_ratio", "target": 1.0},
            "workload": {
                "kind": "python",
                "implementation_sha256": engine.artifacts.put(CODE.encode()),
                "image": IMAGE,
            },
            "policy": {"local_only": False, "placement": "remote"},
        }
    )
    run = engine.submit(engine.freeze(spec), "http-solver")
    runtime_calls = []

    def isolated_solver(self, code, input_payload, runtime_spec, *, cancelled):
        runtime_calls.append(input_payload)
        assert runtime_spec.task == "knapsack"
        assert set(input_payload) == {"instances", "seed"}
        assert not hasattr(runtime_spec, "claim")
        return ExecutionResult(
            predictions=OptimizationPredictions(
                ids=tuple(row["id"] for row in input_payload["instances"]),
                selections=tuple((1,) for _ in input_payload["instances"]),
            ),
            measurements={
                "runtime": "docker",
                "image": IMAGE,
                "task": "knapsack",
                "cleanup_confirmed": True,
            },
        )

    monkeypatch.setattr("paper2bench.repro.remote_client.DockerRunner.run_input", isolated_solver)

    def handle(request):
        assert request.headers["authorization"] == "Bearer " + SECRET
        body = json.loads(request.content)
        if request.url.path == "/worker/claim":
            return httpx.Response(
                200,
                json={
                    "assignment": coordinator.claim(
                        body["worker_id"], tuple(body["images"]), body["max_memory_mb"]
                    )
                },
            )
        if request.url.path == "/worker/heartbeat":
            return httpx.Response(200, json={"active": coordinator.heartbeat(**body)})
        assert request.url.path == "/worker/complete"
        return httpx.Response(200, json=coordinator.complete(**body))

    worker, _ = client(tmp_path, monkeypatch, transport=httpx.MockTransport(handle))
    assert worker.run_once() == {
        "state": "completed",
        "run_id": run["run_id"],
        "run_state": "completed",
    }
    result = engine.result(run["run_id"])["result"]
    assert len(runtime_calls) == 1
    assert result["assessment"]["metric"] == "optimality_ratio"
    assert result["assessment"]["mean_score"] == 1.0
    assert result["measurements"]["runtime_attestation"] == "not_verified"


@pytest.mark.parametrize(
    "bad_output,bad_measurements",
    [
        (Predictions(ids=("test-1",), labels=("\ud800",)), {}),
        ({"ids": ["test-1"], "labels": [object()]}, {}),
        ({"ids": ["test-1"]}, {}),
        (None, {}),
        (Predictions(ids=("test-1",), labels=("pos",)), {"image": OTHER_IMAGE}),
        (Predictions(ids=("test-1",), labels=("pos",)), {"task": ["classification"]}),
        (Predictions(ids=("test-1",), labels=("pos",)), {"memory_limit_mb": 512}),
        (Predictions(ids=("test-1",), labels=("pos",)), {"stdout_bytes": 10**100}),
        (Predictions(ids=("test-1",), labels=("pos",)), {"worker_python_version": "\ud800"}),
        (Predictions(ids=("test-1",), labels=("pos",)), {"wall_seconds": float("nan")}),
        (Predictions(ids=("test-1",), labels=("pos",)), None),
    ],
    ids=[
        "surrogate-output",
        "non-json-label",
        "missing-labels",
        "missing-output",
        "wrong-image",
        "invalid-task-type",
        "wrong-memory",
        "oversized-metadata",
        "surrogate-metadata",
        "nonfinite-metadata",
        "missing-metadata",
    ],
)
def test_invalid_runtime_result_is_durably_failed_before_submission(
    tmp_path, monkeypatch, bad_output, bad_measurements
):
    calls = []

    def invalid_result(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(predictions=bad_output, measurements=bad_measurements)

    monkeypatch.setattr("paper2bench.repro.remote_client.DockerRunner.run_input", invalid_result)
    worker, coordinator = client(tmp_path, monkeypatch)
    coordinator.completion_failure = "timeout"
    assert worker.run_once()["state"] == "pending_submission"
    pending = json.loads((worker.spool / "pending.json").read_bytes())
    assert pending["completion"]["state"] == "failed"
    assert pending["completion"]["predictions"] is None
    assert pending["completion"]["error"] in {"candidate_execution_failed", "worker_internal_error"}
    assert worker.run_once() == {"state": "completed", "run_id": RUN_ID, "run_state": "failed"}
    assert len(calls) == 1
    assert coordinator.claims == 1
    assert len(coordinator.completions) == 2
    assert coordinator.completions[0] == coordinator.completions[1]
    assert coordinator.completions[1]["completion"]["predictions"] is None
    assert not (worker.spool / "pending.json").exists()
