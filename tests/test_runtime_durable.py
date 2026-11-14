"""Restart/concurrency tests for model accounting; no HTTP or paid calls."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from paper2bench.journal import RunJournal
from paper2bench.llm import chat
from paper2bench.providers import ProviderError
from paper2bench.runtime_config import BudgetExceeded, BudgetLedger, RuntimeConfig, RuntimeConfigError, runtime_scope


def profile(requests=2):
    return RuntimeConfig.from_dict({
        "models": {"local": {"provider": "ollama", "model": "test-model", "locality_assurance": "operator_trusted"}},
        "roles": {role: "local" for role in ("author", "reference", "judge", "agent")},
        "policy": {"data_egress": "local_only"},
        "budgets": {pool: {"requests": requests, "input_tokens": 1000, "output_tokens": 100}
                    for pool in ("authoring", "evaluation", "experiments")},
    })


def durable(tmp_path, cfg):
    journal = RunJournal(tmp_path / "runs")
    limits = {f"{pool}.{dimension}": value for pool, dimensions in cfg.budgets.items()
              for dimension, value in dimensions.items()}
    run_id = journal.create({"kind": "model_budget_test"}, limits=limits)
    return journal, run_id


def test_process_restart_retains_submitted_but_unsettled_reservation(tmp_path):
    cfg = profile()
    journal, run_id = durable(tmp_path, cfg)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as first:
        known = first.ledger.reserve("author", 70, 20)
        first.ledger.settle(known, 50, 10)
        first.ledger.reserve("author", 30, 20)  # Simulate process death after dispatch.
    reopened = RunJournal(tmp_path / "runs")
    with runtime_scope(cfg, budget_journal=reopened, budget_run_id=run_id) as restarted:
        with pytest.raises(BudgetExceeded):
            restarted.ledger.reserve("author", 10, 10)
        snapshot = restarted.ledger.snapshot()
    assert snapshot["scope"] == "durable_run"
    assert snapshot["used"]["authoring"] == {"requests": 1, "input_tokens": 50, "output_tokens": 10}
    assert snapshot["reserved"]["authoring"] == {"requests": 1, "input_tokens": 30, "output_tokens": 20}
    assert snapshot["usage"]["authoring"]["requests"] == 2
    assert snapshot["usage"]["authoring"]["usage_unknown_requests"] == 1
    assert snapshot["reconciliation_required"]


def test_reservation_is_committed_before_provider_dispatch(tmp_path, monkeypatch):
    from paper2bench.providers import http
    cfg = profile()
    journal, run_id = durable(tmp_path, cfg)
    observed = []
    def fake_http(*args):
        observed.append(RunJournal(tmp_path / "runs").get(run_id)["budget"])
        return {"done": True, "done_reason": "stop", "message": {"content": "answer"},
                "prompt_eval_count": 7, "eval_count": 2}, "request-id"
    monkeypatch.setattr(http, "_post_json", fake_http)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10)
        snapshot = ctx.ledger.snapshot()
    assert observed[0]["reserved"]["authoring.requests"] == 1
    assert observed[0]["reserved"]["authoring.input_tokens"] >= 1
    assert snapshot["used"]["authoring"] == {"requests": 1, "input_tokens": 7, "output_tokens": 2}
    assert not any(snapshot["reserved"]["authoring"].values())
    assert not snapshot["reconciliation_required"]


@pytest.mark.parametrize("partial", [False, True])
def test_unknown_and_partial_usage_never_release_bounds(tmp_path, monkeypatch, partial):
    from paper2bench.providers import http
    cfg = profile(requests=1)
    journal, run_id = durable(tmp_path, cfg)
    calls = []
    def fake_http(*args):
        calls.append(1)
        response = {"done": True, "message": {"content": "answer"}}
        if partial:
            response["eval_count"] = 2
        return response, None
    monkeypatch.setattr(http, "_post_json", fake_http)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        response = chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10)
        assert not response.usage_complete
        held = ctx.ledger.snapshot()["reserved"]["authoring"]
        assert held["requests"] == 1 and held["input_tokens"] >= 1 and held["output_tokens"] == 10
    with runtime_scope(cfg, budget_journal=RunJournal(tmp_path / "runs"), budget_run_id=run_id):
        with pytest.raises(BudgetExceeded):
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10)
    assert len(calls) == 1


def test_ambiguous_provider_error_retains_budget_and_no_secrets(tmp_path, monkeypatch):
    from paper2bench.providers import http
    cfg = profile(requests=1)
    journal, run_id = durable(tmp_path, cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-recorded")
    def fake_http(*args):
        raise ProviderError("timeout", "ambiguous request", outcome_unknown=True)
    monkeypatch.setattr(http, "_post_json", fake_http)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        with pytest.raises(ProviderError):
            chat("ignored", [{"role": "user", "content": "private prompt"}], max_tokens=10)
        assert ctx.ledger.snapshot()["reserved"]["authoring"]["requests"] == 1
    records = json.dumps(journal.events(run_id)) + json.dumps(journal.get(run_id))
    assert "must-not-be-recorded" not in records
    assert "private prompt" not in records


def test_mixed_role_pools_and_multiple_controllers_share_atomic_limits(tmp_path):
    cfg = profile(requests=3)
    journal, run_id = durable(tmp_path, cfg)
    def reserve_one(index):
        reopened = RunJournal(tmp_path / "runs")
        with runtime_scope(cfg, budget_journal=reopened, budget_run_id=run_id) as ctx:
            try:
                return ctx.ledger.reserve("judge" if index % 2 else "reference", 10, 10).key
            except BudgetExceeded:
                return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = [key for key in pool.map(reserve_one, range(20)) if key]
    assert len(keys) == 3 and len(set(keys)) == 3
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        ctx.ledger.reserve("author", 10, 10)
        ctx.ledger.reserve("agent", 10, 10)
        assert ctx.ledger.snapshot()["reserved"]["evaluation"]["requests"] == 3


def test_durable_configuration_mismatch_and_half_binding_rejected(tmp_path):
    cfg = profile()
    journal, run_id = durable(tmp_path, cfg)
    with pytest.raises(RuntimeConfigError):
        BudgetLedger(cfg.budgets, journal=journal)
    changed = cfg.to_dict()
    changed["budgets"]["authoring"]["requests"] = 9
    with pytest.raises(RuntimeConfigError, match="match"):
        with runtime_scope(changed, budget_journal=journal, budget_run_id=run_id):
            pass


def test_partial_overage_blocks_restart_until_owner_reconciles(tmp_path):
    cfg = profile(requests=3)
    journal, run_id = durable(tmp_path, cfg)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        reservation = ctx.ledger.reserve("author", 10, 10)
        with pytest.raises(BudgetExceeded, match="reconciliation"):
            ctx.ledger.settle(reservation, None, 20)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        with pytest.raises(BudgetExceeded, match="reconciliation"):
            ctx.ledger.reserve("author", 1, 1)
    journal.settle(run_id, reservation.key, {"authoring.requests": 1, "authoring.input_tokens": 10, "authoring.output_tokens": 20})
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        ctx.ledger.reserve("author", 1, 1)


def test_predispatch_validation_leaves_no_reservation(tmp_path):
    cfg = profile()
    journal, run_id = durable(tmp_path, cfg)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        with pytest.raises(ProviderError):
            chat("ignored", [{"role": "tool", "content": "invalid"}], max_tokens=10)
        assert ctx.ledger.snapshot()["usage"] == {}
    assert journal.get(run_id)["budget"]["reserved"] == {}


def test_durable_known_overrun_is_recorded_before_failure(tmp_path):
    cfg = profile()
    journal, run_id = durable(tmp_path, cfg)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        reservation = ctx.ledger.reserve("author", 10, 10)
        with pytest.raises(BudgetExceeded):
            ctx.ledger.settle(reservation, 2000, 20)
    assert journal.get(run_id)["budget"]["used"]["authoring.input_tokens"] == 2000


def test_cancelled_model_run_cannot_dispatch(tmp_path):
    cfg = profile()
    journal, run_id = durable(tmp_path, cfg)
    journal.request_cancel(run_id)
    with runtime_scope(cfg, budget_journal=journal, budget_run_id=run_id) as ctx:
        with pytest.raises(BudgetExceeded):
            ctx.ledger.reserve("author", 10, 10)
