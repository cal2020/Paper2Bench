"""Knapsack paper-to-code drafting with no oracle disclosure or model retries."""

import hashlib
import json

import pytest

from paper2bench.repro.authoring import (
    AuthoringError,
    EvidenceValidationError,
    draft_experiment,
    validate_evidence,
)
from paper2bench.repro.models import OptimizationClaim
from paper2bench.repro.providers import Generation, OutcomeUnknown, ProviderProfile, ProviderUsage
from paper2bench.repro.storage import ArtifactNotFound, ArtifactStore

IMAGE = "sha256:" + "b" * 64
SOURCE = (
    "For 0-1 knapsack, sort items by decreasing value per weight and pack each "
    "item if it fits. A mean optimality ratio of 0.95 is reported."
)
CODE = """import json, sys
data = json.load(sys.stdin)
ids, selections = [], []
for row in data["instances"]:
    remaining = row["capacity"]
    selected = []
    order = sorted(range(len(row["weights"])), key=lambda i: (
        -row["values"][i] / row["weights"][i], i))
    for i in order:
        if row["weights"][i] <= remaining:
            selected.append(i)
            remaining -= row["weights"][i]
    ids.append(row["id"])
    selections.append(selected)
print(json.dumps({"ids": ids, "selections": selections}))
"""


class MockClient:
    def __init__(self, response, *, error=None, finish_reason="stop"):
        self.response = response
        self.error = error
        self.finish_reason = finish_reason
        self.calls = []
        self.profile = ProviderProfile(
            name="mock-author", base_url="http://127.0.0.1:9999/v1", location="local"
        )

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return Generation(
            text=self.response,
            model="actual-author-revision",
            request_id="author-request",
            fingerprint="author-fingerprint",
            finish_reason=self.finish_reason,
            usage=ProviderUsage(input_tokens=100, output_tokens=200, total_tokens=300),
        )


def response(**updates):
    value = {
        "title": "Source-grounded ratio-greedy knapsack",
        "evidence": [{"start": 0, "end": len(SOURCE), "quote": SOURCE}],
        "assumptions": ["Item order breaks equal-ratio ties"],
        "unresolved": [],
        "claim": {"metric": "optimality_ratio", "target": 0.95, "tolerance": 0.1},
        "implementation": CODE,
    }
    return json.dumps({**value, **updates})


@pytest.fixture
def inputs(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.put(SOURCE.encode())
    dataset = artifacts.put_json(
        {
            "task": "knapsack",
            "instances": [
                {
                    "id": "private-optimization-instance-654",
                    "weights": [7919, 8123],
                    "values": [4343, 5657],
                    "capacity": 9000,
                },
                {
                    "id": "private-optimization-instance-655",
                    "weights": [1, 2, 3],
                    "values": [4, 5, 6],
                    "capacity": 10,
                },
            ],
        }
    )
    return artifacts, source, dataset


def draft(inputs, client, **overrides):
    artifacts, source, dataset = inputs
    kwargs = {
        "source_sha256": source,
        "dataset_sha256": dataset,
        "artifacts": artifacts,
        "client": client,
        "model": "requested-author",
        "local_only": True,
        "image": IMAGE,
        "max_output_tokens": 4096,
    }
    return draft_experiment(**{**kwargs, **overrides})


def test_knapsack_draft_uses_task_protocol_and_withholds_instances_and_oracles(inputs):
    client = MockClient(response())
    spec = draft(inputs, client)
    assert len(client.calls) == 1
    assert isinstance(spec.claim, OptimizationClaim)
    assert spec.claim.target == 0.95
    sent = client.calls[0]
    payload = json.loads(sent["messages"][1]["content"])
    assert payload == {
        "source_text": SOURCE,
        "task": "knapsack",
        "problem_shape": {
            "instance_count": 2,
            "min_items": 2,
            "max_items": 3,
            "min_capacity": 10,
            "max_capacity": 9000,
        },
    }
    for private_value in ("private-optimization-instance", "7919", "8123", "4343", "5657"):
        assert private_value not in sent["messages"][1]["content"]
    assert "optimal_objective" not in payload
    prompt = sent["messages"][0]["content"]
    assert 'metric: "optimality_ratio"' in prompt
    assert '"instances"' in prompt and '"selections"' in prompt
    assert "UNTRUSTED DATA" in prompt
    assert "independently computes" in prompt
    assert "No reference solutions or oracle outputs" in prompt
    assert sent["max_output_tokens"] == 4096
    assert sent["local_only"] is True
    assert spec.workload.kind == "python" and spec.workload.image == IMAGE
    assert spec.policy.max_provider_calls == 0
    assert inputs[0].get(spec.workload.implementation_sha256).decode() == CODE


def test_knapsack_receipt_preserves_one_call_and_source_code_provider_identities(inputs):
    artifacts, source, dataset = inputs
    client = MockClient(response())
    spec = draft(inputs, client)
    receipt = artifacts.get_json(spec.authoring_receipt_sha256)
    assert receipt["task"] == "knapsack"
    assert receipt["source_sha256"] == source
    assert receipt["dataset_sha256"] == dataset
    assert receipt["implementation_sha256"] == spec.workload.implementation_sha256
    assert receipt["requested_model"] == "requested-author"
    assert receipt["actual_model"] == "actual-author-revision"
    assert receipt["request_id"] == "author-request"
    assert receipt["fingerprint"] == "author-fingerprint"
    assert receipt["provider_calls"] == 1
    assert receipt["usage"]["total_tokens"] == 300
    assert receipt["implementation_fidelity"] == receipt["protocol_fidelity"] == "unverified"
    assert artifacts.get_json(receipt["prompt_sha256"]) == client.calls[0]["messages"]
    assert artifacts.get(receipt["response_sha256"]).decode() == client.response


@pytest.mark.parametrize("claim", [{"target": 0.95}, {"metric": "accuracy", "target": 0.95}])
def test_knapsack_draft_with_classification_metric_is_rejected_before_code_storage(inputs, claim):
    client = MockClient(response(claim=claim, unresolved=["Unsupported source task"]))
    with pytest.raises(AuthoringError, match="optimality_ratio"):
        draft(inputs, client)
    assert len(client.calls) == 1
    code_hash = hashlib.sha256(CODE.encode()).hexdigest()
    with pytest.raises(ArtifactNotFound):
        inputs[0].get(code_hash)


def test_classification_dataset_rejects_optimization_claim(inputs):
    dataset = inputs[0].put_json(
        {
            "train": [{"id": "train", "x": [0], "y": "a"}],
            "test": [{"id": "test", "x": [1], "y": "a"}],
        }
    )
    client = MockClient(response())
    with pytest.raises(AuthoringError, match="accuracy"):
        draft(inputs, client, dataset_sha256=dataset)
    assert len(client.calls) == 1


def test_unsupported_paper_semantics_remain_unresolved_and_block_freeze(inputs):
    client = MockClient(
        response(
            claim={"metric": "optimality_ratio", "target": 0},
            unresolved=["The paper studies unbounded knapsack and reports no supported ratio"],
        )
    )
    spec = draft(inputs, client)
    assert spec.claim.target == 0
    assert spec.unresolved
    validate_evidence(spec, SOURCE, require_resolved=False)
    with pytest.raises(EvidenceValidationError, match="Unresolved"):
        validate_evidence(spec, SOURCE)
    assert "unbounded knapsack" in client.calls[0]["messages"][0]["content"]


def test_knapsack_wrong_evidence_quote_rejects_before_implementation_storage(inputs):
    bad_quote = "X" + SOURCE[1:]
    client = MockClient(response(evidence=[{"start": 0, "end": len(SOURCE), "quote": bad_quote}]))
    with pytest.raises(EvidenceValidationError, match="source text"):
        draft(inputs, client)
    assert len(client.calls) == 1
    with pytest.raises(ArtifactNotFound):
        inputs[0].get(hashlib.sha256(CODE.encode()).hexdigest())


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_knapsack_response_is_not_repaired_or_retried(inputs, finish_reason):
    client = MockClient(response(), finish_reason=finish_reason)
    with pytest.raises(AuthoringError, match="truncated or filtered"):
        draft(inputs, client)
    assert len(client.calls) == 1


def test_knapsack_authoring_unknown_outcome_propagates_without_retry(inputs):
    error = OutcomeUnknown("outcome unknown", request_id="maybe-billed")
    client = MockClient("", error=error)
    with pytest.raises(OutcomeUnknown) as caught:
        draft(inputs, client)
    assert caught.value is error
    assert len(client.calls) == 1


def test_knapsack_authoring_never_executes_generated_code(inputs):
    code = "raise RuntimeError('authoring must never run draft code')\n"
    spec = draft(inputs, MockClient(response(implementation=code)))
    assert inputs[0].get(spec.workload.implementation_sha256).decode() == code


@pytest.mark.parametrize(
    "bad_dataset",
    [
        {"task": "unbounded_knapsack", "instances": []},
        {"task": "knapsack", "instances": []},
        {
            "task": "knapsack",
            "instances": [{"id": "bad", "weights": [0], "values": [1], "capacity": 1}],
        },
    ],
)
def test_invalid_or_unsupported_dataset_rejects_before_authoring_call(inputs, bad_dataset):
    dataset = inputs[0].put_json(bad_dataset)
    client = MockClient(response())
    with pytest.raises(AuthoringError, match="Invalid dataset"):
        draft(inputs, client, dataset_sha256=dataset)
    assert client.calls == []
