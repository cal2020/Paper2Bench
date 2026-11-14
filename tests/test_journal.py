from concurrent.futures import ThreadPoolExecutor

import pytest

from paper2bench.journal import JournalError, RunJournal


def test_concurrent_reservations_cannot_overspend(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({}, limits={"tokens": 10})

    def reserve(i):
        try:
            journal.reserve(run, str(i), {"tokens": 4})
            return True
        except JournalError:
            return False

    with ThreadPoolExecutor(max_workers=5) as pool:
        assert sum(pool.map(reserve, range(5))) == 2
    assert journal.get(run)["budget"]["reserved"] == {"tokens": 8}


def test_unknown_dispatch_is_not_retryable_and_reservation_survives(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({}, limits={"wall_seconds": 10})
    attempt = journal.begin_attempt(run)
    journal.reserve(run, attempt, {"wall_seconds": 10})
    journal.finish(run, attempt, "reconciling", {"reason": "lost_response"})
    with pytest.raises(JournalError):
        journal.begin_attempt(run)
    assert journal.get(run)["budget"]["reserved"]["wall_seconds"] == 10


def test_intent_is_durable_and_cancellation_does_not_release_cost(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({}, limits={"tokens": 5})
    attempt = journal.begin_attempt(run)
    journal.reserve(run, attempt, {"tokens": 5})
    journal.request_cancel(run)
    reopened = RunJournal(tmp_path)
    assert reopened.get(run)["state"] == "cancellation_requested"
    assert reopened.get(run)["attempts"][0]["request_key"] == attempt
    assert reopened.get(run)["budget"]["reserved"]["tokens"] == 5
    assert reopened.finish(run, attempt, "succeeded", {}) == "cancelled"


def test_reservation_idempotency_and_complete_settlement(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({}, limits={"tokens": 5})
    journal.reserve(run, "call", {"tokens": 5})
    journal.reserve(run, "call", {"tokens": 5})
    with pytest.raises(JournalError):
        journal.reserve(run, "call", {"tokens": 4})
    with pytest.raises(JournalError):
        journal.settle(run, "call", {})
    journal.settle(run, "call", {"tokens": 3})
    journal.settle(run, "call", {"tokens": 3})
    assert journal.get(run)["budget"] == {"limits": {"tokens": 5}, "used": {"tokens": 3}, "reserved": {}}


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "blocked"])
def test_cancellation_cannot_become_a_retryable_failure(tmp_path, outcome):
    journal = RunJournal(tmp_path)
    run = journal.create({})
    attempt = journal.begin_attempt(run)
    journal.request_cancel(run)
    assert journal.finish(run, attempt, outcome, {}) == "cancelled"
    assert RunJournal(tmp_path).get(run)["state"] == "cancelled"
    with pytest.raises(JournalError):
        journal.begin_attempt(run)


def test_cancellation_keeps_ambiguous_dispatch_unsettled(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({}, limits={"wall_seconds": 10})
    attempt = journal.begin_attempt(run)
    journal.reserve(run, attempt, {"wall_seconds": 10})
    journal.request_cancel(run)
    assert journal.finish(run, attempt, "reconciling", {}) == "reconciling"
    assert journal.get(run)["budget"]["reserved"] == {"wall_seconds": 10}


def test_result_factory_observes_the_atomically_resolved_terminal_state(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({})
    attempt = journal.begin_attempt(run)
    journal.request_cancel(run)
    journal.finish(run, attempt, "succeeded", lambda state: {"terminal_state": state})
    reopened = RunJournal(tmp_path).get(run)
    assert reopened["attempts"][0]["result"] == {"terminal_state": "cancelled"}


def test_recover_authoring_retains_settled_usage_and_supersedes_active_attempt(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({"kind": "authoring"}, limits={"tokens": 10})
    attempt = journal.begin_attempt(run)
    journal.reserve(run, "known-call", {"tokens": 10})
    journal.settle(run, "known-call", {"tokens": 7})
    assert RunJournal(tmp_path).recover_authoring(run) == "failed"
    recovered = journal.get(run)
    assert recovered["state"] == "failed"
    assert recovered["attempts"][0]["state"] == "failed"
    assert recovered["budget"]["used"] == {"tokens": 7}
    assert recovered["budget"]["reserved"] == {}
    assert journal.begin_attempt(run) != attempt
    assert journal.events(run)[-2]["kind"] == "authoring_recovery"
    with pytest.raises(JournalError):
        journal.finish(run, attempt, "succeeded", {})


def test_recover_authoring_rejects_unknown_model_usage(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({"kind": "authoring"}, limits={"tokens": 10})
    attempt = journal.begin_attempt(run)
    journal.reserve(run, "unknown-call", {"tokens": 10})
    journal.finish(run, attempt, "reconciling", {"reason": "unknown_model_response"})
    before = journal.get(run)
    with pytest.raises(JournalError, match="usage must be reconciled"):
        journal.recover_authoring(run)
    assert journal.get(run) == before


def test_recover_authoring_handles_finalized_interruption_without_rewriting_prior_result(tmp_path):
    journal = RunJournal(tmp_path)
    run = journal.create({"kind": "authoring"})
    attempt = journal.begin_attempt(run)
    journal.finish(run, attempt, "reconciling", {"reason": "interrupted_finalization"})
    journal.recover_authoring(run)
    result = journal.get(run)
    assert result["state"] == "failed"
    assert result["attempts"][0]["state"] == "reconciling"
    assert result["attempts"][0]["result"] == {"reason": "interrupted_finalization"}
    journal.begin_attempt(run)


@pytest.mark.parametrize("kind,state", [
    ("execution", "running"), ("scientific", "running"),
    ("authoring", "pending"), ("authoring", "succeeded"),
    ("authoring", "failed"), ("authoring", "blocked"),
    ("authoring", "cancelled"), ("authoring", "cancellation_requested"),
])
def test_recover_authoring_is_guarded_by_kind_and_state(tmp_path, kind, state):
    journal = RunJournal(tmp_path)
    run = journal.create({"kind": kind})
    if state != "pending":
        attempt = journal.begin_attempt(run)
        if state == "cancellation_requested":
            journal.request_cancel(run)
        elif state != "running":
            journal.finish(run, attempt, state, {})
    before = journal.get(run)
    with pytest.raises(JournalError):
        journal.recover_authoring(run)
    assert journal.get(run) == before
