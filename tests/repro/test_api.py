"""Actual HTTP routing against persisted state, without paid model calls."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from paper2bench.repro.api import create_app  # noqa: E402
from paper2bench.repro.demo import PAPER  # noqa: E402
from paper2bench.repro.engine import Engine  # noqa: E402

TOKEN = "test-server-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def service(tmp_path):
    state = tmp_path / "state"
    app = create_app(state, token=TOKEN)
    with TestClient(app) as client:
        yield client, state


def register_fixture(client):
    source_response = client.post("/sources", json={"text": PAPER}, headers=HEADERS)
    assert source_response.status_code == 201
    source = source_response.json()
    data = {
        "train": [
            {"id": "train-a", "x": [-1.0], "y": "a"},
            {"id": "train-b", "x": [1.0], "y": "b"},
        ],
        "test": [
            {"id": f"test-{i}", "x": [-2.0 if i % 2 == 0 else 2.0], "y": "a" if i % 2 == 0 else "b"}
            for i in range(80)
        ],
    }
    dataset_response = client.post("/datasets", json=data, headers=HEADERS)
    assert dataset_response.status_code == 201
    spec = {
        "title": "Synthetic API lifecycle fixture",
        "source_sha256": source["source_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "dataset_sha256": dataset_response.json()["dataset_sha256"],
        "evidence": [{"start": 0, "end": len(PAPER), "quote": PAPER}],
        "claim": {"target": 1.0, "tolerance": 0.1},
        "workload": {"kind": "builtin_centroid"},
    }
    response = client.post("/specs", json=spec, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()["spec_hash"], spec


def test_startup_requires_configured_nonempty_token(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPER2BENCH_SERVER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="authentication token"):
        create_app(tmp_path / "missing")
    for invalid in ("", " ", "token\n", "token with space"):
        with pytest.raises(RuntimeError, match="authentication token"):
            create_app(tmp_path / "invalid", token=invalid)


def test_environment_token_is_used_without_exposing_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER2BENCH_SERVER_TOKEN", "environment-secret-token")
    with TestClient(create_app(tmp_path / "environment")) as client:
        assert client.get("/runs").status_code == 401
        response = client.get("/runs", headers={"Authorization": "Bearer environment-secret-token"})
        assert response.status_code == 200
        assert "environment-secret-token" not in response.text


def test_public_health_has_only_status_and_version(service):
    client, _ = service
    response = client.get("/health")
    assert response.status_code == 200
    from paper2bench.repro import __version__

    assert response.json() == {"status": "ok", "version": __version__}


@pytest.mark.parametrize("path", ["/sources", "/datasets", "/profiles", "/specs", "/runs"])
def test_all_data_writes_require_authentication_before_validation(service, path):
    client, _ = service
    response = client.post(path, content=b"not json")
    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "path", ["/runs", "/runs/missing", "/runs/missing/report", "/openapi.json", "/docs"]
)
def test_all_data_reads_require_authentication(service, path):
    client, _ = service
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("authorization", ["Bearer wrong", "Basic test-server-token", "Bearer"])
def test_incorrect_authentication_does_not_echo_token(service, authorization):
    client, _ = service
    response = client.get("/runs", headers={"Authorization": authorization})
    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_queue_then_independent_worker_completes_actual_experiment(service):
    client, state = service
    spec_hash, _ = register_fixture(client)
    response = client.post(
        "/runs", json={"spec_hash": spec_hash, "idempotency_key": "api-run"}, headers=HEADERS
    )
    assert response.status_code == 202
    queued = response.json()
    assert queued["state"] == "queued" and queued["attempt"] == 0
    run_id = queued["run_id"]
    assert client.get(f"/runs/{run_id}", headers=HEADERS).json()["state"] == "queued"
    pending_report = client.get(f"/runs/{run_id}/report", headers=HEADERS).json()
    assert pending_report["result"] is None

    executed = Engine(state).execute(run_id, worker="independent-api-test-worker")
    assert executed["state"] == "completed"
    assert client.get(f"/runs/{run_id}", headers=HEADERS).json()["state"] == "completed"
    report = client.get(f"/runs/{run_id}/report", headers=HEADERS).json()
    assert report["result"]["assessment"]["accuracy"] == 1.0
    assert report["result"]["assessment"]["implementation_fidelity"] == "unverified"
    assert "Paper reproduction: not established" in report["text"]
    assert len(client.get("/runs", headers=HEADERS).json()["runs"]) == 1


def test_idempotent_submission_and_conflict(service):
    client, _ = service
    first_hash, spec = register_fixture(client)
    body = {"spec_hash": first_hash, "idempotency_key": "same-key"}
    first = client.post("/runs", json=body, headers=HEADERS)
    second = client.post("/runs", json=body, headers=HEADERS)
    assert first.status_code == second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"]
    spec["claim"]["tolerance"] = 0.2
    changed = client.post("/specs", json=spec, headers=HEADERS).json()["spec_hash"]
    conflict = client.post(
        "/runs", json={"spec_hash": changed, "idempotency_key": "same-key"}, headers=HEADERS
    )
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency_conflict"}


def test_cancel_is_persisted_and_does_not_execute(service):
    client, state = service
    spec_hash, _ = register_fixture(client)
    queued = client.post(
        "/runs", json={"spec_hash": spec_hash, "idempotency_key": "cancel"}, headers=HEADERS
    ).json()
    response = client.post(f"/runs/{queued['run_id']}/cancel", headers=HEADERS)
    assert response.status_code == 200 and response.json()["state"] == "cancelled"
    assert Engine(state).execute(queued["run_id"])["state"] == "cancelled"


@pytest.mark.parametrize("path", ["/runs/missing", "/runs/missing/report"])
def test_missing_run_is_404(service, path):
    client, _ = service
    response = client.get(path, headers=HEADERS)
    assert response.status_code == 404
    assert response.json() == {"detail": "resource_not_found"}


def test_missing_frozen_spec_is_404_and_does_not_queue(service):
    client, _ = service
    response = client.post(
        "/runs", json={"spec_hash": "f" * 64, "idempotency_key": "missing"}, headers=HEADERS
    )
    assert response.status_code == 404
    assert client.get("/runs", headers=HEADERS).json() == {"runs": []}


def test_validation_response_never_echoes_submitted_credentials(service):
    client, _ = service
    secret = "SENSITIVE-ACCIDENTAL-CREDENTIAL"
    bad_profile = {
        "name": "bad",
        "location": "cloud",
        "base_url": f"https://user:{secret}@example.invalid/v1",
    }
    for path, body in (
        ("/profiles", bad_profile),
        ("/sources", {"text": 123, secret: secret}),
        ("/runs", {"spec_hash": secret, "idempotency_key": "key"}),
    ):
        response = client.post(path, json=body, headers=HEADERS)
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid_request"}
        assert secret not in response.text


def test_profile_registration_only_stores_configuration(service):
    client, state = service
    profile = {
        "name": "local-test",
        "location": "local",
        "base_url": "http://127.0.0.1:12345/v1",
        "api_key_env": "LOCAL_MODEL_KEY",
    }
    response = client.post("/profiles", json=profile, headers=HEADERS)
    assert response.status_code == 201
    stored = Engine(state).artifacts.get_json(response.json()["provider_profile_sha256"])
    assert stored["api_key_env"] == "LOCAL_MODEL_KEY"
    assert "api_key" not in stored


def test_source_normalization_and_manifest_are_retained(service):
    client, state = service
    original = "\ufeffCafe\u0301\r\nResults"
    response = client.post("/sources", json={"text": original}, headers=HEADERS)
    assert response.status_code == 201
    receipt = response.json()
    artifacts = Engine(state).artifacts
    assert artifacts.get(receipt["source_sha256"]).decode() == "Café\nResults"
    assert artifacts.get(receipt["original_sha256"]).decode() == original
    manifest = artifacts.get_json(receipt["source_manifest_sha256"])
    assert manifest["source_sha256"] == receipt["source_sha256"]


def test_body_limit_applies_before_parsing_and_no_raw_input_is_echoed(service):
    client, _ = service
    payload = json.dumps({"text": "x" * (4 * 1024 * 1024)})
    response = client.post("/sources", content=payload, headers=HEADERS)
    assert response.status_code == 413
    assert response.json() == {"detail": "request_too_large"}


def test_server_paths_shell_and_execution_routes_are_absent(service):
    client, state = service
    response = client.post("/sources", json={"path": "/etc/passwd"}, headers=HEADERS)
    assert response.status_code == 422
    for path in ("/execute", "/draft", "/shell", "/artifacts/anything", "/docs"):
        assert client.post(path, json={}, headers=HEADERS).status_code == 404
    assert not list(state.glob("*.py"))
