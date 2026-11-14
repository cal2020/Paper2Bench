import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from paper2bench.runtime_config import (
    BudgetExceeded, BudgetLedger, ModelRef, PolicyDenied, RuntimeConfig,
    RuntimeConfigError, get_context, get_role, load_runtime_config,
    role_scope, runtime_scope,
)


def config(**changes):
    value = {
        "models": {"local": {"provider": "ollama", "model": "test-model", "hosting": "local_server", "locality_assurance": "operator_trusted"}},
        "roles": {role: "local" for role in ("author", "reference", "agent", "judge")},
        "policy": {"data_egress": "local_only"},
        "budgets": {pool: {"requests": 10, "input_tokens": 100000, "output_tokens": 10000} for pool in ("authoring", "evaluation", "experiments")},
    }
    value.update(changes)
    return value


def test_context_and_role_are_nested_and_restored():
    assert get_context() is None
    with runtime_scope(config()) as outer:
        assert get_context() is outer
        with role_scope("judge"):
            assert get_role() == "judge"
            with pytest.raises(RuntimeError):
                with runtime_scope(config()):
                    raise RuntimeError("test")
            assert get_context() is outer
        assert get_role() == "author"
    assert get_context() is None


def test_profile_roundtrip_and_yaml(tmp_path):
    cfg = RuntimeConfig.from_dict(config())
    assert RuntimeConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    path = tmp_path / "runtime.yaml"
    path.write_text(json.dumps(config()))
    assert load_runtime_config(path).roles["agent"] == "local"


@pytest.mark.parametrize("bad", [None, [], {"schema_version": 2}, {"api_key": "must-not-be-stored"}])
def test_invalid_runtime_shapes_fail(bad):
    with pytest.raises(RuntimeConfigError):
        RuntimeConfig.from_dict(bad)


def test_unknown_roles_and_capability_shapes_rejected():
    with pytest.raises(RuntimeConfigError):
        RuntimeConfig.from_dict(config(roles={"curator_typo": "local"}))
    with pytest.raises(RuntimeConfigError):
        ModelRef.from_dict({"provider": "ollama", "model": "test", "required_capabilities": "text_generation"})


@pytest.mark.parametrize("endpoint", ["file:///etc/passwd", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret", "http://example.com/v1", "https://example.com:bad"])
def test_unsafe_endpoint_rejected(endpoint):
    with pytest.raises(RuntimeConfigError):
        ModelRef.from_dict({"provider": "openai_compatible", "model": "test", "endpoint": endpoint})


def test_credential_references_not_values(monkeypatch):
    cfg = config()
    cfg["models"]["local"]["credential_ref"] = {"env": "P2B_TEST_KEY"}
    monkeypatch.setenv("P2B_TEST_KEY", "private-value")
    with runtime_scope(cfg) as ctx:
        ref = ctx.resolve("author")
        assert ctx.credential(ref) == "private-value"
        assert "private-value" not in repr(ctx)
        assert "private-value" not in json.dumps(ctx.config.to_dict())


def test_execution_metadata_redacts_accidental_credentials():
    cfg = RuntimeConfig.from_dict(config(execution={"env": {"OPENAI_API_KEY": "private-value", "PATH": "/usr/bin"}}))
    assert "private-value" not in json.dumps(cfg.to_dict())
    assert cfg.to_dict()["execution"]["env"]["PATH"] == "/usr/bin"


def test_local_endpoint_does_not_inherit_cloud_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-secret")
    with runtime_scope(config()) as ctx:
        assert ctx.credential(ctx.resolve("author")) is None


def test_model_from_env_is_explicit(monkeypatch):
    with pytest.raises(RuntimeConfigError, match="unset"):
        ModelRef.from_dict({"provider": "ollama", "model_from_env": "P2B_TEST_MODEL_MISSING"})
    monkeypatch.setenv("P2B_TEST_MODEL", "local-model")
    assert ModelRef.from_dict({"provider": "ollama", "model_from_env": "P2B_TEST_MODEL"}).model == "local-model"


@pytest.mark.parametrize("mutation", [
    {"provider": "openai", "endpoint": "https://api.openai.com/v1"},
    {"provider": "ollama", "model": "test-cloud"},
    {"locality_assurance": "unknown"},
    {"hosting": "self_managed_cloud"},
    {"endpoint": "https://remote.example.com"},
])
def test_local_only_blocks_known_cloud_and_unknown_residency(mutation):
    cfg = config()
    cfg["models"]["local"].update(mutation)
    with runtime_scope(cfg) as ctx, pytest.raises(PolicyDenied):
        ctx.resolve("author")


def test_verified_offline_not_proven_by_yaml():
    cfg = config(policy={"data_egress": "local_only", "verified_offline_required": True})
    cfg["models"]["local"]["locality_assurance"] = "egress_enforced"
    with runtime_scope(cfg) as ctx, pytest.raises(PolicyDenied, match="verifier"):
        ctx.resolve("author")
    with runtime_scope(cfg, offline_verifier=lambda ref: True) as ctx:
        assert ctx.resolve("author").provider == "ollama"


def test_mixed_provider_generic_key_rejected():
    cfg = config(policy={"data_egress": "allow"})
    cfg["models"]["remote"] = {"provider": "anthropic", "model": "claude-test"}
    cfg["roles"]["judge"] = "remote"
    with pytest.raises(RuntimeConfigError, match="ambiguous"):
        with runtime_scope(cfg, api_key="generic"):
            pass


def test_same_provider_different_endpoints_cannot_share_generic_key():
    cfg = config(policy={"data_egress": "allow"})
    cfg["models"]["other"] = {"provider": "ollama", "model": "other-model", "endpoint": "http://127.0.0.1:11435"}
    cfg["roles"]["judge"] = "other"
    with pytest.raises(RuntimeConfigError, match="ambiguous"):
        with runtime_scope(cfg, api_key="must-not-cross-endpoints"):
            pass
    with runtime_scope(cfg) as ctx:
        with pytest.raises(RuntimeConfigError, match="ambiguous"):
            ctx.credential(ctx.resolve("author"), "must-not-cross-endpoints")


def test_distinct_endpoint_credentials_stay_explicit(monkeypatch):
    cfg = config(policy={"data_egress": "allow"})
    cfg["models"]["local"]["credential_ref"] = "P2B_AUTHOR_KEY"
    cfg["models"]["other"] = {"provider": "ollama", "model": "other-model", "endpoint": "http://127.0.0.1:11435", "credential_ref": "P2B_JUDGE_KEY"}
    cfg["roles"]["judge"] = "other"
    monkeypatch.setenv("P2B_AUTHOR_KEY", "author-private")
    monkeypatch.setenv("P2B_JUDGE_KEY", "judge-private")
    with runtime_scope(cfg) as ctx:
        assert ctx.credential(ctx.resolve("author")) == "author-private"
        assert ctx.credential(ctx.resolve("judge")) == "judge-private"


def test_atomic_budget_reservations_and_unknown_usage():
    ledger = BudgetLedger({"authoring": {"requests": 3, "input_tokens": 100, "output_tokens": 100}})
    def reserve():
        try:
            return ledger.reserve("author", 10, 10)
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = [r for r in pool.map(lambda _: reserve(), range(20)) if r]
    assert len(reservations) == 3
    ledger.settle(reservations[0], None, None)
    ledger.settle(reservations[1], 2, 3)
    used = ledger.snapshot()["usage"]["authoring"]
    assert used == {"requests": 3, "input_tokens": 22, "output_tokens": 23, "usage_unknown_requests": 1}
    with pytest.raises(RuntimeConfigError):
        ledger.settle(reservations[0], 0, 0)


def test_unbounded_role_pool_is_not_allowed():
    ledger = BudgetLedger({})
    with pytest.raises(BudgetExceeded):
        ledger.reserve("judge", 10, 10)


def test_provider_overage_is_recorded_and_fails():
    ledger = BudgetLedger({"authoring": {"requests": 1, "input_tokens": 10, "output_tokens": 10}})
    reservation = ledger.reserve("author", 2, 3)
    with pytest.raises(BudgetExceeded):
        ledger.settle(reservation, 100, 100)
    assert ledger.snapshot()["usage"]["authoring"]["input_tokens"] == 100


@pytest.mark.parametrize("bad_limit", [-1, True, 2.5])
def test_budget_limits_strictly_typed(bad_limit):
    with pytest.raises(RuntimeConfigError):
        RuntimeConfig.from_dict(config(budgets={"authoring": {"requests": bad_limit}}))
