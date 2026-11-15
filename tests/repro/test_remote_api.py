from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from paper2bench.repro.api import create_app
from paper2bench.repro.bundles import export_bundle, replay_bundle
from paper2bench.repro.demo import create_demo, create_optimization_demo
from paper2bench.repro.models import ExperimentSpec
from paper2bench.repro.remote_client import WorkerClient

ADMIN = "admin-verification-secret"
WORKER = "worker-verification-secret"
IMAGE = "sha256:" + "a" * 64


def remote_spec(engine, *, task="classification", image=IMAGE):
    if task == "classification":
        frozen = create_demo(engine)
        from paper2bench.repro.demo import CODE

        code = CODE
    else:
        frozen = create_optimization_demo(engine)
        from paper2bench.repro.demo import OPTIMIZATION_CODE

        code = OPTIMIZATION_CODE
    raw = engine.load_spec(frozen).model_dump(mode="json")
    raw["workload"] = {
        "kind": "python",
        "image": image,
        "implementation_sha256": engine.artifacts.put(code.encode()),
    }
    raw["policy"].update(local_only=False, placement="remote")
    return engine.freeze(ExperimentSpec.model_validate(raw))


def bearer(token):
    return {"Authorization": "Bearer " + token}


def test_worker_roles_identity_and_trusted_scoring(tmp_path):
    app = create_app(
        tmp_path, token=ADMIN, worker_tokens={"worker-a": WORKER, "worker-b": "different-secret"}
    )
    engine = app.state.engine
    spec_hash = remote_spec(engine)
    with TestClient(app) as client:
        run = client.post(
            "/runs",
            json={"spec_hash": spec_hash, "idempotency_key": "remote-api"},
            headers=bearer(ADMIN),
        ).json()
        request = {"worker_id": "worker-a", "images": [IMAGE], "max_memory_mb": 256}
        assert client.post("/worker/claim", json=request, headers=bearer(ADMIN)).status_code == 401
        assert client.get("/runs", headers=bearer(WORKER)).status_code == 401
        assert (
            client.post(
                "/worker/claim", json={**request, "worker_id": "worker-b"}, headers=bearer(WORKER)
            ).status_code
            == 403
        )
        response = client.post("/worker/claim", json=request, headers=bearer(WORKER))
        assert response.status_code == 200
        assignment = response.json()["assignment"]
        assert assignment["run_id"] == run["run_id"]
        assert all(set(row) == {"id", "x"} for row in assignment["input"]["test"])
        assert not {
            "claim",
            "evidence",
            "dataset_sha256",
            "provider_profile",
            "source_sha256",
        } & set(assignment)
        assert engine.execute(run["run_id"])["worker"] == "remote:worker-a"
        predictions = {
            "ids": [r["id"] for r in assignment["input"]["test"]],
            "labels": ["a" if r["x"][0] < 0 else "b" for r in assignment["input"]["test"]],
        }
        completion = {
            "state": "completed",
            "input_sha256": assignment["input_sha256"],
            "implementation_sha256": assignment["implementation_sha256"],
            "image": IMAGE,
            "predictions": predictions,
            "measurements": {"runtime": "docker"},
        }
        body = {
            "worker_id": "worker-a",
            "run_id": run["run_id"],
            "token": assignment["token"],
            "completion": completion,
        }
        first = client.post("/worker/complete", json=body, headers=bearer(WORKER))
        assert first.status_code == 200 and first.json()["state"] == "completed"
        again = client.post("/worker/complete", json=body, headers=bearer(WORKER))
        assert again.json()["result_hash"] == first.json()["result_hash"]
        assert (
            client.get(f"/runs/{run['run_id']}/report", headers=bearer(WORKER)).status_code == 401
        )
        assert engine.result(run["run_id"])["result"]["assessment"]["accuracy"] == 1.0
        changed = json.loads(json.dumps(body))
        changed["completion"]["predictions"]["labels"][0] = "b"
        assert (
            client.post("/worker/complete", json=changed, headers=bearer(WORKER)).status_code == 409
        )


def test_worker_configuration_rejects_shared_or_unicode_secrets(tmp_path):
    for workers in ({"worker-a": ADMIN}, {"a": WORKER, "b": WORKER}, {"a": "nonascii-é"}):
        with pytest.raises(RuntimeError):
            create_app(tmp_path, token=ADMIN, worker_tokens=workers)


def test_environment_worker_keys_are_references_not_persisted_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER2BENCH_WORKERS", json.dumps({"worker-a": "P2B_WORKER_SECRET"}))
    monkeypatch.setenv("P2B_WORKER_SECRET", WORKER)
    app = create_app(tmp_path, token=ADMIN)
    with TestClient(app) as client:
        assert client.get("/runs", headers=bearer(WORKER)).status_code == 401
        response = client.post(
            "/worker/claim",
            json={"worker_id": "worker-a", "images": [IMAGE]},
            headers=bearer(WORKER),
        )
        assert response.json() == {"assignment": None}
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert WORKER.encode() not in file.read_bytes()


@pytest.mark.docker
@pytest.mark.parametrize("task", ["classification", "knapsack"])
def test_actual_http_remote_worker_docker_and_replay(tmp_path, monkeypatch, task):
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set staged immutable PAPER2BENCH_DOCKER_IMAGE")
    import uvicorn

    app = create_app(tmp_path / "server", token=ADMIN, worker_tokens={"worker-a": WORKER})
    engine = app.state.engine
    run = engine.submit(remote_spec(engine, task=task, image=image), "real-remote-" + task)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert server.started
        monkeypatch.setenv("P2B_REMOTE_TEST_TOKEN", WORKER)
        client = WorkerClient(
            f"http://127.0.0.1:{port}",
            "worker-a",
            "P2B_REMOTE_TEST_TOKEN",
            tmp_path / "worker-spool",
            (image,),
        )
        outcome = client.run_once()
        assert outcome["state"] == "completed", outcome
        result = engine.result(run["run_id"])["result"]
        assert engine.runs.get_run(run["run_id"])["state"] == "completed"
        assert result["assessment"].get("accuracy", result["assessment"].get("mean_score")) == 1.0
        bundle = export_bundle(engine, run["run_id"], tmp_path / "remote.zip")
        assert replay_bundle(bundle)["scores"] == "recomputed"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive()
