from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from paper2bench.repro.budgets import (
    BudgetConflict,
    BudgetedProviderClient,
    BudgetExceeded,
    BudgetNotFound,
    BudgetStore,
)
from paper2bench.repro.providers import (
    Generation,
    OutcomeUnknown,
    ProviderProfile,
    ProviderRequestError,
    ProviderUsage,
)


@pytest.fixture
def store(tmp_path):
    return BudgetStore(tmp_path / "budgets.sqlite")


def test_create_is_idempotent_and_cannot_reset_existing_usage(store):
    first = store.create("research", 3, 100)
    assert first == store.create("research", 3, 100)
    store.reserve("research", "call-1", output_tokens=20)
    again = store.create("research", 3, 100)
    assert again["used_calls"] == 1
    assert again["used_output_tokens"] == 20
    with pytest.raises(BudgetConflict):
        store.create("research", 4, 100)
    with pytest.raises(BudgetConflict):
        store.create("research", 3, 101)


def test_missing_budget_and_reservation(store):
    with pytest.raises(BudgetNotFound):
        store.get("missing")
    with pytest.raises(BudgetNotFound):
        store.reserve("missing", "call-1", output_tokens=1)
    store.create("known", 2, 10)
    with pytest.raises(BudgetNotFound):
        store.reconcile("known", "missing", actual_output_tokens=None, outcome="unknown")


@pytest.mark.parametrize("identifier", ["", "../escape", "with space", "\nsecret", "x" * 129, True])
def test_safe_bounded_identifiers(store, identifier):
    with pytest.raises(ValueError):
        store.create(identifier, 1, 10)
    store.create("safe", 1, 10)
    with pytest.raises(ValueError):
        store.reserve("safe", identifier, output_tokens=10)


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "2", 10**12 + 1])
def test_quota_counters_reject_bool_coercion_and_overflow(store, value):
    with pytest.raises(ValueError):
        store.create("bad-calls", value, 100)
    with pytest.raises(ValueError):
        store.create("bad-tokens", 1, value)
    store.create("safe", 2, 100)
    with pytest.raises(ValueError):
        store.reserve("safe", "bad", output_tokens=value)


def test_reserve_idempotence_and_parameter_conflicts(store):
    store.create("campaign", 3, 30)
    first = store.reserve("campaign", "request", calls=2, output_tokens=20, run_id="run-a")
    assert first == store.reserve("campaign", "request", calls=2, output_tokens=20, run_id="run-a")
    for overrides in ({"calls": 1}, {"output_tokens": 19}, {"run_id": "run-b"}):
        arguments = {"calls": 2, "output_tokens": 20, "run_id": "run-a", **overrides}
        with pytest.raises(BudgetConflict):
            store.reserve("campaign", "request", **arguments)
    budget = store.get("campaign")
    assert budget["used_calls"] == 2 and budget["used_output_tokens"] == 20
    assert len(store.events("campaign")) == 2


def test_call_and_token_limits_fail_before_reservation(store):
    store.create("calls", 1, 100)
    store.reserve("calls", "first", output_tokens=1)
    with pytest.raises(BudgetExceeded):
        store.reserve("calls", "second", output_tokens=1)
    store.create("tokens", 100, 10)
    store.reserve("tokens", "first", output_tokens=10)
    with pytest.raises(BudgetExceeded):
        store.reserve("tokens", "second", output_tokens=1)
    assert store.get("calls")["used_calls"] == 1
    assert store.get("tokens")["used_output_tokens"] == 10


def test_zero_quota_and_zero_token_calls(store):
    store.create("closed", 0, 0)
    with pytest.raises(BudgetExceeded):
        store.reserve("closed", "first", output_tokens=0)
    store.create("no-output", 1, 0)
    store.reserve("no-output", "first", output_tokens=0)
    with pytest.raises(ValueError):
        store.reserve("no-output", "zero-calls", calls=0, output_tokens=0)


def test_concurrent_cross_run_reservations_admit_exact_quota(tmp_path):
    path = tmp_path / "shared.sqlite"
    BudgetStore(path).create("shared", 7, 70)

    def reserve(index):
        instance = BudgetStore(path)
        try:
            instance.reserve("shared", f"call-{index}", output_tokens=10, run_id=f"run-{index}")
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(reserve, range(30)))
    assert sum(outcomes) == 7
    budget = BudgetStore(path).get("shared")
    assert budget["used_calls"] == 7
    assert budget["used_output_tokens"] == 70
    assert len(BudgetStore(path).events("shared")) == 8


def test_concurrent_duplicate_reservation_consumes_once(tmp_path):
    path = tmp_path / "shared.sqlite"
    BudgetStore(path).create("shared", 1, 10)

    def reserve(_index):
        return BudgetStore(path).reserve("shared", "same-request", output_tokens=10, run_id="run-a")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve, range(20)))
    assert all(result == results[0] for result in results)
    assert BudgetStore(path).get("shared")["used_calls"] == 1


def test_concurrent_duplicate_reconciliation_refunds_once(tmp_path):
    path = tmp_path / "shared.sqlite"
    first = BudgetStore(path)
    first.create("shared", 1, 10)
    first.reserve("shared", "same-request", output_tokens=10)

    def reconcile(_index):
        return BudgetStore(path).reconcile(
            "shared", "same-request", actual_output_tokens=3, outcome="completed"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reconcile, range(20)))
    assert all(result == results[0] for result in results)
    assert first.get("shared")["used_calls"] == 1
    assert first.get("shared")["used_output_tokens"] == 3
    assert len(first.events("shared")) == 3


def test_unknown_and_unreconciled_crash_reservations_survive_restart(tmp_path):
    path = tmp_path / "durable.sqlite"
    first = BudgetStore(path)
    first.create("campaign", 2, 20)
    first.reserve("campaign", "timed-out", output_tokens=10)
    first.reconcile("campaign", "timed-out", actual_output_tokens=None, outcome="unknown")
    first.reserve("campaign", "crashed-before-reconcile", output_tokens=10)
    restarted = BudgetStore(path)
    assert restarted.get("campaign")["used_calls"] == 2
    assert restarted.get("campaign")["used_output_tokens"] == 20
    with pytest.raises(BudgetExceeded):
        restarted.reserve("campaign", "new-call", output_tokens=1)


def test_known_completion_releases_unused_tokens_but_retains_call(store):
    store.create("campaign", 3, 30)
    store.reserve("campaign", "first", output_tokens=20)
    settled = store.reconcile("campaign", "first", actual_output_tokens=5, outcome="completed")
    assert settled["accounted_calls"] == 1
    assert settled["accounted_output_tokens"] == 5
    assert store.get("campaign")["remaining_output_tokens"] == 25
    assert (
        store.reconcile("campaign", "first", actual_output_tokens=5, outcome="completed") == settled
    )
    assert store.get("campaign")["used_output_tokens"] == 5
    assert len(store.events("campaign")) == 3


def test_unknown_can_be_reconciled_to_completion(store):
    store.create("campaign", 2, 20)
    store.reserve("campaign", "first", output_tokens=10)
    store.reconcile("campaign", "first", actual_output_tokens=None, outcome="unknown")
    store.reconcile("campaign", "first", actual_output_tokens=3, outcome="completed")
    assert store.get("campaign")["used_output_tokens"] == 3
    assert store.get("campaign")["used_calls"] == 1


def test_completed_unknown_usage_retains_maximum_then_can_refine(store):
    store.create("campaign", 2, 20)
    store.reserve("campaign", "first", output_tokens=10)
    store.reconcile("campaign", "first", actual_output_tokens=None, outcome="completed")
    assert store.get("campaign")["used_output_tokens"] == 10
    store.reconcile("campaign", "first", actual_output_tokens=4, outcome="completed")
    assert store.get("campaign")["used_output_tokens"] == 4


def test_definite_nondispatch_releases_both_once(store):
    store.create("campaign", 1, 10)
    store.reserve("campaign", "first", output_tokens=10)
    rejected = store.reconcile("campaign", "first", actual_output_tokens=None, outcome="rejected")
    assert (
        store.reconcile("campaign", "first", actual_output_tokens=0, outcome="rejected") == rejected
    )
    assert store.get("campaign")["used_calls"] == 0
    assert store.get("campaign")["used_output_tokens"] == 0
    store.reserve("campaign", "second", output_tokens=10)


def test_unknown_can_release_only_after_explicit_definite_nondispatch_outcome(store):
    store.create("campaign", 1, 10)
    store.reserve("campaign", "first", output_tokens=10)
    store.reconcile("campaign", "first", actual_output_tokens=None, outcome="unknown")
    store.reconcile("campaign", "first", actual_output_tokens=None, outcome="rejected")
    assert store.get("campaign")["used_calls"] == 0


@pytest.mark.parametrize(
    "terminal,new_outcome,new_actual",
    [
        ("rejected", "completed", 2),
        ("rejected", "unknown", None),
        ("completed", "rejected", None),
        ("completed", "unknown", None),
        ("completed", "completed", 3),
    ],
)
def test_conflicting_terminal_reconciliation_never_changes_counters(
    store, terminal, new_outcome, new_actual
):
    store.create("campaign", 2, 20)
    store.reserve("campaign", "first", output_tokens=10)
    store.reconcile(
        "campaign",
        "first",
        actual_output_tokens=2 if terminal == "completed" else None,
        outcome=terminal,
    )
    before = store.get("campaign")
    with pytest.raises(BudgetConflict):
        store.reconcile("campaign", "first", actual_output_tokens=new_actual, outcome=new_outcome)
    assert store.get("campaign") == before


def test_provider_overage_records_actual_and_blocks_future_dispatch(store):
    store.create("campaign", 10, 100)
    store.reserve("campaign", "first", output_tokens=10)
    store.reserve("campaign", "unused", output_tokens=10)
    settled = store.reconcile("campaign", "first", actual_output_tokens=15, outcome="completed")
    assert settled["actual_output_tokens"] == 15
    assert settled["overage"] == 5
    assert store.get("campaign")["used_output_tokens"] == 25
    assert store.get("campaign")["blocked"] is True
    store.reconcile("campaign", "unused", actual_output_tokens=None, outcome="rejected")
    assert store.get("campaign")["blocked"] is True
    with pytest.raises(BudgetExceeded, match="blocked"):
        store.reserve("campaign", "later", output_tokens=0)
    # Reading/retrying an existing reservation does not consume a second quota.
    assert store.reserve("campaign", "first", output_tokens=10)["state"] == "completed"


@pytest.mark.parametrize(
    "actual,outcome",
    [
        (True, "completed"),
        (-1, "completed"),
        (1, "unknown"),
        (2, "rejected"),
        (None, "invented"),
    ],
)
def test_invalid_reconciliation_does_not_change_usage(store, actual, outcome):
    store.create("campaign", 1, 10)
    store.reserve("campaign", "first", output_tokens=10)
    before = store.get("campaign")
    with pytest.raises(ValueError):
        store.reconcile("campaign", "first", actual_output_tokens=actual, outcome=outcome)
    assert store.get("campaign") == before


class FakeProvider:
    def __init__(self, *, output_tokens=3, error=None, finish_reason="stop"):
        self.profile = ProviderProfile(
            name="fixture", base_url="http://127.0.0.1:8080/v1", location="local"
        )
        self.calls = []
        self.error = error
        self.generation = Generation(
            text="generated private content",
            model="fixture-model",
            usage=ProviderUsage(output_tokens=output_tokens),
            finish_reason=finish_reason,
        )

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.generation


def request(max_output_tokens=10):
    return {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "private prompt must never enter ledger"}],
        "max_output_tokens": max_output_tokens,
        "local_only": True,
    }


def test_wrapper_preserves_profile_forwards_once_and_refunds_unused_tokens(store):
    store.create("campaign", 2, 20)
    provider = FakeProvider(output_tokens=3)
    wrapped = BudgetedProviderClient(provider, store, "campaign", run_id="run-a")
    assert wrapped.profile is provider.profile
    result = wrapped.generate(**request())
    assert result is provider.generation
    assert provider.calls == [request()]
    assert store.get("campaign")["used_calls"] == 1
    assert store.get("campaign")["used_output_tokens"] == 3
    assert wrapped.last_reservation_id is not None
    events = store.events("campaign")
    assert events[-1]["reservation_id"] == wrapped.last_reservation_id
    assert events[-2]["data"]["run_id"] == "run-a"
    assert "private prompt" not in repr(events)
    assert "generated private" not in repr(events)


def test_wrapper_exhaustion_rejects_before_dispatch(store):
    store.create("campaign", 0, 0)
    provider = FakeProvider()
    wrapped = BudgetedProviderClient(provider, store, "campaign")
    with pytest.raises(BudgetExceeded):
        wrapped.generate(**request())
    assert provider.calls == []
    assert wrapped.last_reservation_id is None


@pytest.mark.parametrize(
    "error",
    [
        OutcomeUnknown("request may have completed"),
        ProviderRequestError("outage"),
        RuntimeError("invalid response after dispatch"),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
def test_wrapper_errors_and_cancellation_retain_full_reservation(store, error):
    store.create("campaign", 1, 10)
    provider = FakeProvider(error=error)
    wrapped = BudgetedProviderClient(provider, store, "campaign")
    with pytest.raises(type(error)):
        wrapped.generate(**request())
    assert len(provider.calls) == 1
    assert store.get("campaign")["used_calls"] == 1
    assert store.get("campaign")["used_output_tokens"] == 10
    assert store.events("campaign")[-1]["data"]["outcome"] == "unknown"
    with pytest.raises(BudgetExceeded):
        wrapped.generate(**request())
    assert len(provider.calls) == 1


def test_wrapper_unknown_completion_usage_conservatively_holds_tokens(store):
    store.create("campaign", 1, 10)
    provider = FakeProvider(output_tokens=None)
    wrapped = BudgetedProviderClient(provider, store, "campaign")
    wrapped.generate(**request())
    assert store.get("campaign")["used_calls"] == 1
    assert store.get("campaign")["used_output_tokens"] == 10
    assert store.events("campaign")[-1]["data"]["outcome"] == "completed"
    assert store.events("campaign")[-1]["data"]["actual_output_tokens"] is None


def test_wrapper_partial_generation_never_retries(store):
    store.create("campaign", 2, 20)
    provider = FakeProvider(output_tokens=10, finish_reason="length")
    wrapped = BudgetedProviderClient(provider, store, "campaign")
    result = wrapped.generate(**request())
    assert result.finish_reason == "length"
    assert len(provider.calls) == 1
    assert store.get("campaign")["used_output_tokens"] == 10


@pytest.mark.parametrize("reported", [15, 10**12])
def test_wrapper_records_actual_overage_then_blocks_and_raises(store, reported):
    store.create("campaign", 2, 20)
    provider = FakeProvider(output_tokens=reported)
    wrapped = BudgetedProviderClient(provider, store, "campaign")
    with pytest.raises(BudgetExceeded, match="provider reported"):
        wrapped.generate(**request())
    assert len(provider.calls) == 1
    assert store.get("campaign")["used_output_tokens"] == reported
    assert store.get("campaign")["blocked"] is True
    assert store.events("campaign")[-1]["data"]["actual_output_tokens"] == reported
    with pytest.raises(BudgetExceeded):
        wrapped.generate(**request())
    assert len(provider.calls) == 1


def test_wrapper_unknown_reconciliation_failure_keeps_durable_reservation(store, monkeypatch):
    store.create("campaign", 1, 10)
    original_error = OutcomeUnknown("dispatch outcome unknown")
    wrapped = BudgetedProviderClient(FakeProvider(error=original_error), store, "campaign")

    def unavailable(*args, **kwargs):
        raise RuntimeError("ledger temporarily unavailable")

    monkeypatch.setattr(store, "reconcile", unavailable)
    with pytest.raises(OutcomeUnknown) as raised:
        wrapped.generate(**request())
    assert raised.value is original_error
    assert store.get("campaign")["used_calls"] == 1
    assert store.get("campaign")["used_output_tokens"] == 10


@pytest.mark.parametrize("tokens", [None, True, 0, -1, "10"])
def test_wrapper_invalid_requested_cap_never_dispatches_or_reserves(store, tokens):
    store.create("campaign", 1, 10)
    provider = FakeProvider()
    wrapped = BudgetedProviderClient(provider, store, "campaign")
    with pytest.raises(ValueError):
        wrapped.generate(**request(tokens))
    assert provider.calls == []
    assert store.get("campaign")["used_calls"] == 0
