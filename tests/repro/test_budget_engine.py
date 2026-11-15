from __future__ import annotations

import json

from paper2bench.repro.engine import Engine
from paper2bench.repro.models import ExperimentSpec
from paper2bench.repro.providers import Generation, OutcomeUnknown, ProviderUsage


def model_spec(engine):
    text = "Classify negative and positive numbers; reported accuracy is 1.0."
    source = engine.artifacts.put(text.encode())
    dataset = engine.register_dataset(
        {
            "train": [{"id": "a", "x": [-1.0], "y": "a"}, {"id": "b", "x": [1.0], "y": "b"}],
            "test": [{"id": "heldout", "x": [-0.5], "y": "a"}],
        }
    )
    profile = engine.register_profile(
        {"name": "test", "base_url": "http://127.0.0.1:1234/v1", "location": "local"}
    )
    return ExperimentSpec.model_validate(
        {
            "title": "Budget integration",
            "source_sha256": source,
            "dataset_sha256": dataset,
            "evidence": [{"start": 0, "end": len(text), "quote": text}],
            "claim": {"target": 1.0},
            "workload": {
                "kind": "model_classification",
                "provider_profile": profile,
                "model": "test",
                "prompt": "classify",
            },
            "policy": {"max_provider_calls": 1, "max_output_tokens": 16, "budget_id": "shared"},
        }
    )


def test_shared_budget_caps_multiple_runs_before_provider_dispatch(tmp_path, monkeypatch):
    engine = Engine(tmp_path)
    engine.budgets.create("shared", max_calls=1, max_output_tokens=16)
    calls = []

    def generate(*_, **__):
        calls.append(1)
        return Generation(
            text=json.dumps({"label": "a"}), model="test", usage=ProviderUsage(output_tokens=2)
        )

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", generate)
    frozen = engine.freeze(model_spec(engine))
    first = engine.run(frozen, "first")
    second = engine.run(frozen, "second")
    assert first["state"] == "completed"
    assert second["state"] == "resource_exhausted"
    assert len(calls) == 1
    assert engine.result(second["run_id"])["result"]["assessment"]["verdict"] == "not_tested"
    quota = engine.budgets.get("shared")
    assert quota["used_calls"] == 1 and quota["used_output_tokens"] == 2


def test_unknown_model_outcome_retains_quota_across_restart(tmp_path, monkeypatch):
    engine = Engine(tmp_path)
    engine.budgets.create("shared", 1, 16)
    calls = []

    def generate(*_, **__):
        calls.append(1)
        raise OutcomeUnknown("accepted but completion unavailable")

    monkeypatch.setattr("paper2bench.repro.engine.ProviderClient.generate", generate)
    frozen = engine.freeze(model_spec(engine))
    assert engine.run(frozen, "unknown")["state"] == "outcome_unknown"
    reopened = Engine(tmp_path)
    assert reopened.run(frozen, "after-restart")["state"] == "resource_exhausted"
    assert len(calls) == 1
    assert reopened.budgets.get("shared")["used_output_tokens"] == 16
