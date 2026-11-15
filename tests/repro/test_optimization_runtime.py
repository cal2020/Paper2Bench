from __future__ import annotations

import hashlib
import itertools
import json
import os
import random

import pytest

from paper2bench.repro.models import (
    EvidenceAnchor,
    ExperimentSpec,
    KnapsackDataset,
    KnapsackInstance,
    Limits,
    OptimizationClaim,
    OptimizationPredictions,
    WorkloadSpec,
)
from paper2bench.repro.runtime import (
    BuiltinKnapsackRunner,
    DockerRunner,
    ExecutionCancelled,
    ExecutionFailure,
    ResourceExhausted,
    RuntimeSpec,
    sanitized_input,
)

IMAGE = "sha256:" + "a" * 64
CODE = b"import json,sys; p=json.load(sys.stdin); print(json.dumps({'ids':[r['id'] for r in p['instances']], 'selections':[[1,2] for r in p['instances']]}))"


def dataset():
    return KnapsackDataset(
        instances=(KnapsackInstance(id="case", weights=(2, 3, 3), values=(3, 4, 4), capacity=6),)
    )


def spec(kind="builtin_knapsack", code=CODE, **overrides):
    return ExperimentSpec(
        title="knapsack runtime fixture",
        source_sha256="0" * 64,
        dataset_sha256="1" * 64,
        evidence=(EvidenceAnchor(start=0, end=4, quote="test"),),
        claim=OptimizationClaim(target=1.0),
        workload=WorkloadSpec(
            kind=kind,
            **(
                {"implementation_sha256": hashlib.sha256(code).hexdigest(), "image": IMAGE}
                if kind == "python"
                else {}
            ),
        ),
        **overrides,
    )


def test_exact_runner_avoids_greedy_trap_and_is_deterministic():
    runner = BuiltinKnapsackRunner()
    first = runner.run(dataset(), spec(), cancelled=lambda: False)
    second = runner.run(dataset(), spec(), cancelled=lambda: False)
    assert first.predictions == second.predictions
    assert first.predictions.selections == ((1, 2),)
    assert first.measurements["memory_limit_enforced"] is False


def test_exact_runner_agrees_with_exhaustive_independent_oracle():
    rng = random.Random(33)
    cases = tuple(
        KnapsackInstance(
            id=f"case-{index}",
            weights=tuple(rng.randint(1, 10) for _ in range(7)),
            values=tuple(rng.randint(0, 20) for _ in range(7)),
            capacity=rng.randint(0, 30),
        )
        for index in range(20)
    )
    data = KnapsackDataset(instances=cases)
    result = BuiltinKnapsackRunner().run(data, spec(), cancelled=lambda: False)
    for instance, selected in zip(data.instances, result.predictions.selections, strict=True):
        possible = (
            indices
            for count in range(len(instance.weights) + 1)
            for indices in itertools.combinations(range(len(instance.weights)), count)
            if sum(instance.weights[index] for index in indices) <= instance.capacity
        )
        optimum = max(sum(instance.values[index] for index in indices) for indices in possible)
        assert sum(instance.weights[index] for index in selected) <= instance.capacity
        assert sum(instance.values[index] for index in selected) == optimum


def test_zero_capacity_zero_objective_and_ties():
    data = KnapsackDataset(
        instances=(
            KnapsackInstance(id="zero-capacity", weights=(1, 2), values=(3, 4), capacity=0),
            KnapsackInstance(id="zero-value", weights=(1, 2), values=(0, 0), capacity=2),
            KnapsackInstance(id="tie", weights=(1, 1), values=(1, 1), capacity=1),
        )
    )
    result = BuiltinKnapsackRunner().run(data, spec(), cancelled=lambda: False)
    assert result.predictions.selections == ((), (), (0,))


def test_exact_runner_cancellation_and_output_limit():
    with pytest.raises(ExecutionCancelled):
        BuiltinKnapsackRunner().run(dataset(), spec(), cancelled=lambda: True)
    with pytest.raises(ResourceExhausted, match="output"):
        BuiltinKnapsackRunner().run(
            dataset(), spec(limits=Limits(output_bytes=1)), cancelled=lambda: False
        )


def test_exact_runner_memory_preflight():
    data = KnapsackDataset(
        instances=(
            KnapsackInstance(
                id="large-integers", weights=(1,), values=(1 << 10000,), capacity=10000
            ),
        )
    )
    with pytest.raises(ResourceExhausted, match="working-memory"):
        BuiltinKnapsackRunner().run(
            data, spec(limits=Limits(memory_mb=16)), cancelled=lambda: False
        )


def test_knapsack_public_input_and_minimal_worker_spec(monkeypatch):
    frozen = spec("python")
    public = sanitized_input(dataset(), frozen)
    assert public == {
        "instances": [{"id": "case", "weights": [2, 3, 3], "values": [3, 4, 4], "capacity": 6}],
        "seed": 0,
    }
    minimal = RuntimeSpec(workload=frozen.workload, limits=frozen.limits, seed=0, task="knapsack")
    runner = DockerRunner()
    calls = []
    captured = {}

    def control(args, **kwargs):
        calls.append(args)
        if args[0] == "create":
            return ("b" * 64).encode()
        if args[0] == "inspect":
            return b'{"OOMKilled":false,"ExitCode":0,"Running":false}'
        return b""

    def attached(name, payload, **kwargs):
        captured.update(json.loads(payload))
        return b'{"ids":["case"],"selections":[[1,2]]}', b"", 0

    monkeypatch.setattr(runner, "_control", control)
    monkeypatch.setattr(runner, "_attached", attached)
    result = runner.run_input(CODE, public, minimal, cancelled=lambda: False)
    assert isinstance(result.predictions, OptimizationPredictions)
    assert result.predictions.selections == ((1, 2),)
    assert captured["input"] == public
    assert result.measurements["task"] == "knapsack"
    assert calls[-1][0] == "rm"


@pytest.mark.parametrize(
    "alteration",
    [
        lambda p: p["instances"][0].update(capacity=True),
        lambda p: p["instances"][0].update(weights=[0, 3, 3]),
        lambda p: p["instances"][0].update(values=[-1, 4, 4]),
        lambda p: p["instances"][0].update(optimal_objective=8),
        lambda p: p["instances"].append(p["instances"][0].copy()),
        lambda p: p.update(seed=1),
    ],
)
def test_invalid_public_optimization_input_rejected_without_docker(monkeypatch, alteration):
    runner = DockerRunner()
    calls = []
    monkeypatch.setattr(runner, "_control", lambda *args, **kwargs: calls.append(args))
    public = sanitized_input(dataset(), spec("python"))
    alteration(public)
    with pytest.raises(ExecutionFailure):
        runner.run_input(CODE, public, spec("python"), cancelled=lambda: False)
    assert calls == []


def test_worker_rejects_mismatched_task(monkeypatch):
    frozen = spec("python")
    minimal = RuntimeSpec(
        workload=frozen.workload, limits=frozen.limits, seed=0, task="classification"
    )
    runner = DockerRunner()
    calls = []
    monkeypatch.setattr(runner, "_control", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ExecutionFailure, match="input"):
        runner.run_input(CODE, sanitized_input(dataset(), frozen), minimal, cancelled=lambda: False)
    assert calls == []


@pytest.mark.docker
def test_real_optimization_candidate_uses_same_docker_lifecycle():
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to a staged immutable Python image")
    frozen = spec("python")
    frozen = frozen.model_copy(
        update={"workload": frozen.workload.model_copy(update={"image": image})}
    )
    result = DockerRunner().run(CODE, dataset(), frozen, cancelled=lambda: False)
    assert result.predictions.selections == ((1, 2),)
    assert result.measurements["task"] == "knapsack"
    assert result.measurements["cleanup_confirmed"] is True
