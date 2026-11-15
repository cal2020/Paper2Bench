import hashlib
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from paper2bench.repro.storage import (
    ArtifactCorruption,
    ArtifactNotFound,
    ArtifactStore,
    IdempotencyConflict,
    InvalidArtifactHash,
    RunNotFound,
    RunStore,
)

SPEC = hashlib.sha256(b"specification").hexdigest()
RESULT = hashlib.sha256(b"result").hexdigest()


@pytest.fixture
def artifacts(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def runs(tmp_path):
    return RunStore(tmp_path / "runs.sqlite3")


def test_content_addressed_json_and_concurrent_atomic_put(artifacts):
    payload = b"immutable data" * 1000
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(lambda _: artifacts.put(payload), range(32)))
    assert len(set(hashes)) == 1
    assert hashes[0] == hashlib.sha256(payload).hexdigest()
    assert artifacts.get(hashes[0]) == payload
    assert not list(artifacts.root.rglob(".pending-*"))
    left = artifacts.put_json({"b": [1, 2], "a": "text"})
    right = artifacts.put_json({"a": "text", "b": [1, 2]})
    assert left == right
    assert artifacts.get_json(left) == {"a": "text", "b": [1, 2]}


def test_artifact_tamper_fails_closed_and_is_never_replaced(artifacts):
    digest = artifacts.put(b"original")
    path = artifacts.root / digest[:2] / digest[2:]
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactCorruption):
        artifacts.get(digest)
    with pytest.raises(ArtifactCorruption):
        artifacts.put(b"original")
    assert path.read_bytes() == b"tampered"


@pytest.mark.parametrize("reference", ["../escape", "A" * 64, "a" * 63, "a" * 65, "", None])
def test_artifact_references_reject_traversal_and_malformed_hashes(artifacts, reference):
    with pytest.raises(InvalidArtifactHash):
        artifacts.get(reference)


def test_missing_symlink_and_invalid_json_artifacts(artifacts, tmp_path):
    with pytest.raises(ArtifactNotFound):
        artifacts.get("a" * 64)
    digest = artifacts.put(b"correct")
    path = artifacts.root / digest[:2] / digest[2:]
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"correct")
    path.symlink_to(outside)
    with pytest.raises(ArtifactCorruption):
        artifacts.get(digest)
    for invalid in [b"not json", b"NaN", b"Infinity", b"\xff"]:
        with pytest.raises(ArtifactCorruption):
            artifacts.get_json(artifacts.put(invalid))
    with pytest.raises((ValueError, TypeError)):
        artifacts.put_json({"bad": math.nan})


def test_creation_is_transactionally_idempotent_and_conflicts(runs):
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: runs.create_run(SPEC, "same-key"), range(24)))
    assert len({row["id"] for row in rows}) == 1
    assert len(runs.list_runs()) == 1
    assert len(runs.events(rows[0]["id"])) == 1
    with pytest.raises(IdempotencyConflict):
        runs.create_run(RESULT, "same-key")
    assert runs.get_run(rows[0]["id"])["spec_hash"] == SPEC


def test_only_one_worker_claims_and_terminal_publication_is_immutable(runs):
    run_id = runs.create_run(SPEC, "claim-race")["id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda n: runs.claim(run_id, f"worker-{n}"), range(16)))
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    claim = winners[0]
    assert (claim["token"], claim["attempt"]) == (1, 1)
    assert runs.finish(run_id, claim["token"], "completed", RESULT)
    assert not runs.finish(run_id, claim["token"], "failed", error="late")
    assert runs.claim(run_id, "late-worker") is None
    before = runs.get_run(run_id)
    assert runs.cancel(run_id) == before
    assert before["state"] == "completed"
    assert before["result_hash"] == RESULT
    assert before["attempts"][0]["state"] == "completed"


def test_lease_expiry_fences_old_worker_and_preserves_attempts(runs, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    run_id = runs.create_run(SPEC, "expiry")["id"]
    first = runs.claim(run_id, "first", lease_seconds=10)
    now[0] = 1005
    assert runs.heartbeat(run_id, first["token"], lease_seconds=10)
    assert runs.get_run(run_id)["lease_expires_at"] == 1015
    now[0] = 1015  # Expiry boundary itself is already invalid.
    assert not runs.is_active(run_id, first["token"])
    assert not runs.heartbeat(run_id, first["token"])
    assert not runs.finish(run_id, first["token"], "completed", RESULT)
    second = runs.claim(run_id, "second", lease_seconds=20)
    assert (second["token"], second["attempt"]) == (2, 2)
    assert not runs.finish(run_id, first["token"], "completed", RESULT)
    assert not runs.heartbeat(run_id, first["token"])
    assert runs.is_active(run_id, second["token"])
    assert runs.finish(run_id, second["token"], "completed", RESULT)
    record = runs.get_run(run_id)
    assert [item["state"] for item in record["attempts"]] == ["expired", "completed"]
    assert record["attempts"][0]["finished_at"] == 1015
    assert [event["kind"] for event in runs.events(run_id)] == [
        "created",
        "claimed",
        "heartbeat",
        "lease_expired",
        "claimed",
        "finished",
    ]


@pytest.mark.parametrize("claimed", [False, True])
def test_cancel_fences_workers_and_is_idempotent(runs, claimed):
    run_id = runs.create_run(SPEC, f"cancel-{claimed}")["id"]
    lease = runs.claim(run_id, "worker") if claimed else None
    cancelled = runs.cancel(run_id)
    assert cancelled["state"] == "cancelled"
    assert runs.cancel(run_id) == cancelled
    assert runs.claim(run_id, "another") is None
    if lease:
        assert cancelled["token"] > lease["token"]
        assert not runs.finish(run_id, lease["token"], "completed", RESULT)
        assert not runs.heartbeat(run_id, lease["token"])
        assert cancelled["attempts"][0]["state"] == "cancelled"


def test_restart_retains_state_and_unknown_outcome_is_terminal(tmp_path):
    path = tmp_path / "durable.sqlite3"
    store = RunStore(path)
    run_id = store.create_run(SPEC, "restart")["id"]
    lease = store.claim(run_id, "worker")
    reopened = RunStore(path)
    assert reopened.is_active(run_id, lease["token"])
    assert reopened.finish(
        run_id,
        lease["token"],
        "outcome_unknown",
        error="token=shhh Bearer xyz",
    )
    record = RunStore(path).get_run(run_id)
    assert record["state"] == "outcome_unknown"
    assert "shhh" not in record["error"] and "xyz" not in record["error"]
    assert reopened.claim(run_id, "retry") is None


def test_invalid_ownership_and_states_cannot_publish(runs):
    run_id = runs.create_run(SPEC, "invalid")["id"]
    with pytest.raises(RunNotFound):
        runs.get_run("missing")
    for duration in [0, -1, math.nan, math.inf, True]:
        with pytest.raises((ValueError, TypeError)):
            runs.claim(run_id, "worker", duration)
    lease = runs.claim(run_id, "worker")
    assert not runs.is_active(run_id, True)
    assert not runs.finish(run_id, lease["token"] + 1, "completed", RESULT)
    with pytest.raises(ValueError):
        runs.finish(run_id, lease["token"], "queued")
    with pytest.raises(ValueError):
        runs.finish(run_id, lease["token"], "completed")
    assert runs.get_run(run_id)["state"] == "running"


def call_metadata(**changes):
    call = {
        "phase": "dispatched",
        "call_id": "call-1",
        "test_id": "example-1",
        "requested_model": "local-foundation",
        "provider_profile": SPEC,
        "input_sha256": RESULT,
        "ordinal": 0,
    }
    call.update(changes)
    return call


def test_provider_call_ledger_survives_crash_resume_and_cancellation(tmp_path, monkeypatch):
    now = [2000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    path = tmp_path / "ledger.sqlite3"
    store = RunStore(path)
    run_id = store.create_run(SPEC, "call-ledger")["id"]
    first = store.claim(run_id, "worker-1", lease_seconds=5)
    dispatched = call_metadata()
    assert store.record_call(run_id, first["token"], dispatched)
    # Reopening models a crash after the request intent was committed.
    reopened = RunStore(path)
    assert reopened.calls(run_id) == [dispatched]
    now[0] = 2005
    assert not reopened.record_call(run_id, first["token"], call_metadata(phase="completed"))
    second = reopened.claim(run_id, "worker-2")
    unknown = call_metadata(phase="outcome_unknown", error="provider_outcome_unknown")
    assert reopened.record_call(run_id, second["token"], unknown)
    assert not reopened.record_call(run_id, first["token"], call_metadata(phase="completed"))
    reopened.cancel(run_id)
    assert not reopened.record_call(run_id, second["token"], call_metadata(phase="completed"))
    assert RunStore(path).calls(run_id) == [dispatched, unknown]
    assert reopened.get_run(run_id)["state"] == "cancelled"
    call_events = [event for event in reopened.events(run_id) if event["kind"] == "provider_call"]
    assert [event["attempt"] for event in call_events] == [1, 2]


def test_provider_completed_metadata_is_copied_and_remains_after_publication(runs):
    run_id = runs.create_run(SPEC, "call-complete")["id"]
    lease = runs.claim(run_id, "worker")
    completed = call_metadata(
        phase="completed",
        output_sha256=RESULT,
        model="actual-model",
        request_id="request-1",
        usage={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105},
        fingerprint=None,
        finish_reason="stop",
        wall_seconds=0.125,
    )
    assert runs.record_call(run_id, lease["token"], completed)
    completed["usage"]["input_tokens"] = 999
    assert runs.finish(run_id, lease["token"], "completed", RESULT)
    assert runs.calls(run_id)[0]["usage"]["input_tokens"] == 100
    assert not runs.record_call(run_id, lease["token"], call_metadata())


@pytest.mark.parametrize(
    "change",
    [
        {"api_key": "secret"},
        {"messages": [{"content": "raw input"}]},
        {"text": "raw output"},
        {"headers": {"Authorization": "Bearer secret"}},
        {"phase": "invalid"},
        {"call_id": ""},
        {"input_sha256": "../outside"},
        {"usage": {"api_key": "secret"}},
        {"usage": {"input_tokens": True}},
        {"wall_seconds": math.nan},
        {"error": "token=secret"},
        {"ordinal": -1},
        {"test_id": "x" * (64 * 1024 + 1)},
    ],
)
def test_provider_ledger_rejects_raw_payloads_secrets_and_invalid_metadata(runs, change):
    run_id = runs.create_run(SPEC, "bad-ledger")["id"]
    lease = runs.claim(run_id, "worker")
    with pytest.raises((ValueError, TypeError)):
        runs.record_call(run_id, lease["token"], call_metadata(**change))
    assert runs.calls(run_id) == []


def test_cancelled_attempt_keeps_diagnostics_without_resurrecting_ownership(runs):
    run_id = runs.create_run(SPEC, "cancel-cleanup")["id"]
    lease = runs.claim(run_id, "worker")
    cancelled = runs.cancel(run_id)
    assert runs.attach_diagnostic(run_id, lease["token"], RESULT)
    assert runs.diagnostics(run_id) == [
        {
            "attempt": lease["attempt"],
            "token": lease["token"],
            "artifact_sha256": RESULT,
        }
    ]
    assert runs.get_run(run_id) == cancelled
    assert not runs.is_active(run_id, lease["token"])
    assert not runs.finish(run_id, lease["token"], "completed", RESULT)
    event = runs.events(run_id)[-1]
    assert event["kind"] == "attempt_diagnostic"
    assert event["token"] == lease["token"]
    assert event["token"] != cancelled["token"]


def test_expired_attempt_diagnostics_never_overwrite_successor_or_its_score(runs, monkeypatch):
    now = [3000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    run_id = runs.create_run(SPEC, "old-diagnostics")["id"]
    first = runs.claim(run_id, "first", lease_seconds=5)
    now[0] = 3005
    second = runs.claim(run_id, "second", lease_seconds=10)
    before = runs.get_run(run_id)
    assert runs.attach_diagnostic(run_id, first["token"], SPEC)
    assert runs.get_run(run_id) == before
    assert runs.is_active(run_id, second["token"])
    assert runs.finish(run_id, second["token"], "completed", RESULT)
    completed = runs.get_run(run_id)
    assert runs.attach_diagnostic(run_id, first["token"], RESULT)
    assert runs.get_run(run_id) == completed
    assert completed["result_hash"] == RESULT
    assert completed["state"] == "completed"
    assert [item["attempt"] for item in runs.diagnostics(run_id)] == [1, 1]


def test_diagnostic_duplicates_are_transactionally_idempotent_and_persist(tmp_path):
    path = tmp_path / "diagnostics.sqlite3"
    store = RunStore(path)
    run_id = store.create_run(SPEC, "duplicate-diagnostic")["id"]
    lease = store.claim(run_id, "worker")
    with ThreadPoolExecutor(max_workers=8) as pool:
        attached = list(
            pool.map(
                lambda _: store.attach_diagnostic(run_id, lease["token"], RESULT),
                range(24),
            )
        )
    assert all(attached)
    reopened = RunStore(path)
    assert reopened.diagnostics(run_id) == [
        {
            "attempt": 1,
            "token": 1,
            "artifact_sha256": RESULT,
        }
    ]
    assert len([e for e in reopened.events(run_id) if e["kind"] == "attempt_diagnostic"]) == 1


def test_diagnostics_reject_unknown_attempts_and_invalid_references(runs):
    run_id = runs.create_run(SPEC, "unknown-attempt")["id"]
    assert not runs.attach_diagnostic(run_id, 0, RESULT)
    lease = runs.claim(run_id, "worker")
    for token in [lease["token"] + 1, -1, True, None]:
        assert not runs.attach_diagnostic(run_id, token, RESULT)
    with pytest.raises(InvalidArtifactHash):
        runs.attach_diagnostic(run_id, lease["token"], "../outside")
    assert runs.diagnostics(run_id) == []


def test_cancel_if_active_fences_live_attempt_and_keeps_terminal_state(runs):
    run_id = runs.create_run(SPEC, "cas-cancel-live")["id"]
    assert not runs.cancel_if_active(run_id, 0)
    lease = runs.claim(run_id, "worker")
    assert runs.cancel_if_active(run_id, lease["token"])
    cancelled = runs.get_run(run_id)
    assert cancelled["state"] == "cancelled"
    assert cancelled["token"] == lease["token"] + 1
    assert cancelled["lease_expires_at"] is None
    assert cancelled["result_hash"] is None
    assert cancelled["attempts"][0]["state"] == "cancelled"
    assert not runs.cancel_if_active(run_id, lease["token"])
    assert not runs.cancel_if_active(run_id, cancelled["token"])
    assert not runs.finish(run_id, lease["token"], "completed", RESULT)
    assert runs.get_run(run_id) == cancelled
    event = runs.events(run_id)[-1]
    assert event["kind"] == "cancelled" and event["token"] == cancelled["token"]


def test_cancel_if_active_rejects_expiry_and_stale_successor_tokens(runs, monkeypatch):
    now = [4000.0]
    monkeypatch.setattr("paper2bench.repro.storage.time.time", lambda: now[0])
    run_id = runs.create_run(SPEC, "cas-cancel-expired")["id"]
    first = runs.claim(run_id, "first", lease_seconds=5)
    now[0] = 4005
    before = runs.get_run(run_id)
    assert not runs.cancel_if_active(run_id, first["token"], diagnostic_hash=SPEC)
    assert runs.get_run(run_id) == before
    assert runs.diagnostics(run_id) == []
    second = runs.claim(run_id, "second", lease_seconds=10)
    successor = runs.get_run(run_id)
    for bad_token in [first["token"], True, None, str(second["token"])]:
        assert not runs.cancel_if_active(run_id, bad_token, diagnostic_hash=SPEC)
    assert runs.get_run(run_id) == successor
    assert runs.diagnostics(run_id) == []
    assert runs.is_active(run_id, second["token"])
    assert runs.finish(run_id, second["token"], "completed", RESULT)
    completed = runs.get_run(run_id)
    assert not runs.cancel_if_active(run_id, second["token"], diagnostic_hash=SPEC)
    assert runs.get_run(run_id) == completed
    assert runs.diagnostics(run_id) == []


def test_cancel_if_active_attaches_original_token_receipt_atomically(tmp_path):
    path = tmp_path / "cancel-receipt.sqlite3"
    store = RunStore(path)
    run_id = store.create_run(SPEC, "cas-cancel-receipt")["id"]
    lease = store.claim(run_id, "worker")
    assert store.cancel_if_active(run_id, lease["token"], diagnostic_hash=RESULT)
    reopened = RunStore(path)
    assert reopened.get_run(run_id)["state"] == "cancelled"
    assert reopened.get_run(run_id)["result_hash"] is None
    assert reopened.diagnostics(run_id) == [
        {
            "attempt": lease["attempt"],
            "token": lease["token"],
            "artifact_sha256": RESULT,
        }
    ]
    events = reopened.events(run_id)
    assert [event["kind"] for event in events[-2:]] == ["cancelled", "attempt_diagnostic"]
    assert events[-2]["token"] == lease["token"] + 1
    assert events[-1]["token"] == lease["token"]
    assert reopened.attach_diagnostic(run_id, lease["token"], RESULT)
    assert len(reopened.diagnostics(run_id)) == 1


def test_cancel_if_active_rejects_invalid_receipt_without_cancelling(runs):
    run_id = runs.create_run(SPEC, "cas-invalid-receipt")["id"]
    lease = runs.claim(run_id, "worker")
    before = runs.get_run(run_id)
    with pytest.raises(InvalidArtifactHash):
        runs.cancel_if_active(run_id, lease["token"], diagnostic_hash="../outside")
    assert runs.get_run(run_id) == before
    assert runs.diagnostics(run_id) == []


def test_cancel_if_active_concurrent_receipt_has_one_winner(runs):
    run_id = runs.create_run(SPEC, "cas-concurrent")["id"]
    lease = runs.claim(run_id, "worker")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(
                lambda _: runs.cancel_if_active(run_id, lease["token"], diagnostic_hash=RESULT),
                range(24),
            )
        )
    assert sum(outcomes) == 1
    assert runs.get_run(run_id)["token"] == lease["token"] + 1
    assert len(runs.diagnostics(run_id)) == 1
    assert len([event for event in runs.events(run_id) if event["kind"] == "cancelled"]) == 1


def test_cancel_if_active_receipt_failure_rolls_back_cancellation(runs, monkeypatch):
    run_id = runs.create_run(SPEC, "cas-receipt-rollback")["id"]
    lease = runs.claim(run_id, "worker")
    before = runs.get_run(run_id)
    before_events = runs.events(run_id)
    emit = runs._event

    def fail_receipt(connection, row, kind, at, **details):
        if kind == "attempt_diagnostic":
            raise RuntimeError("simulated receipt storage failure")
        emit(connection, row, kind, at, **details)

    monkeypatch.setattr(runs, "_event", fail_receipt)
    with pytest.raises(RuntimeError, match="simulated"):
        runs.cancel_if_active(run_id, lease["token"], diagnostic_hash=RESULT)
    assert runs.get_run(run_id) == before
    assert runs.events(run_id) == before_events
    assert runs.diagnostics(run_id) == []
    assert runs.is_active(run_id, lease["token"])
