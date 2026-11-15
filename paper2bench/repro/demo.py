"""An explicitly synthetic fixture, not an empirical research-paper reproduction."""

from .models import ClaimSpec, EvidenceAnchor, ExperimentSpec, OptimizationClaim, WorkloadSpec
from .runtime import DockerRunner

PAPER = """Synthetic Paper2Bench reference fixture

Method: fit one centroid per class using training examples only. Assign each test
example to the nearest centroid in Euclidean distance; break ties by label order.
On the supplied synthetic, separated one-dimensional dataset, accuracy is 1.0.
This document is a software test fixture, not a published research result.
"""

CODE = """import json, math, sys
data = json.load(sys.stdin)
labels = sorted({row["y"] for row in data["train"]})
centroids = {}
for label in labels:
    rows = [row["x"] for row in data["train"] if row["y"] == label]
    centroids[label] = [math.fsum(col) / len(rows) for col in zip(*rows)]
result = []
for row in data["test"]:
    result.append(min(labels, key=lambda label: (
        math.dist(row["x"], centroids[label]), label)))
print(json.dumps({"ids": [row["id"] for row in data["test"]], "labels": result}))
"""


def create_demo(engine, *, python: bool = False, image: str = "python:3.12-slim") -> str:
    source = engine.artifacts.put(PAPER.encode())
    dataset = {
        "train": [
            {"id": "train-a1", "x": [-2.0], "y": "a"},
            {"id": "train-a2", "x": [-1.0], "y": "a"},
            {"id": "train-b1", "x": [1.0], "y": "b"},
            {"id": "train-b2", "x": [2.0], "y": "b"},
        ],
        "test": [
            {
                "id": f"test-{i:03d}",
                "x": [(-1.0 if i % 2 == 0 else 1.0) * (1.0 + i / 1000)],
                "y": "a" if i % 2 == 0 else "b",
            }
            for i in range(80)
        ],
    }
    data_hash = engine.register_dataset(dataset)
    if python:
        resolved = DockerRunner().resolve_image(image)
        workload = WorkloadSpec(
            kind="python", implementation_sha256=engine.artifacts.put(CODE.encode()), image=resolved
        )
    else:
        workload = WorkloadSpec(kind="builtin_centroid")
    spec = ExperimentSpec(
        title="Synthetic nearest-centroid reference fixture",
        source_sha256=source,
        dataset_sha256=data_hash,
        evidence=(EvidenceAnchor(start=0, end=len(PAPER), quote=PAPER),),
        assumptions=(
            "Synthetic software fixture; no published paper is being reproduced.",
            "Binomial interval is illustrative, not a population generalization claim.",
        ),
        claim=ClaimSpec(target=1.0, tolerance=0.1),
        workload=workload,
    )
    return engine.freeze(spec)


OPTIMIZATION_PAPER = """Synthetic knapsack software fixture
Solve each zero-one knapsack instance by maximizing total item value without
exceeding capacity. Each item may be selected at most once. The reference method
uses capacity dynamic programming. The optimality ratio is 1.0 on these instances.
This is a software verification fixture, not a published research result.
"""

OPTIMIZATION_CODE = """import json,sys
p=json.load(sys.stdin)
selections=[]
for problem in p['instances']:
    cap=problem['capacity']
    states=[(0,()) for _ in range(cap+1)]
    for i,(w,v) in enumerate(zip(problem['weights'],problem['values'])):
        for c in range(cap,w-1,-1):
            score,items=states[c-w]
            if score+v>states[c][0]:
                states[c]=(score+v,items+(i,))
    selections.append(list(states[cap][1]))
print(json.dumps({'ids':[q['id'] for q in p['instances']],'selections':selections}))
"""


def create_optimization_demo(
    engine, *, python: bool = False, image: str = "python:3.12-slim"
) -> str:
    import random

    rng = random.Random(17)
    data = {
        "task": "knapsack",
        "instances": [
            {
                "id": f"knapsack-{i:03d}",
                "weights": [rng.randrange(1, 10) for _ in range(8)],
                "values": [rng.randrange(1, 20) for _ in range(8)],
                "capacity": 20,
            }
            for i in range(80)
        ],
    }
    workload = WorkloadSpec(kind="builtin_knapsack")
    if python:
        workload = WorkloadSpec(
            kind="python",
            implementation_sha256=engine.artifacts.put(OPTIMIZATION_CODE.encode()),
            image=DockerRunner().resolve_image(image),
        )
    spec = ExperimentSpec(
        title="Synthetic bounded knapsack verification",
        source_sha256=engine.artifacts.put(OPTIMIZATION_PAPER.encode()),
        dataset_sha256=engine.register_dataset(data),
        evidence=(EvidenceAnchor(start=0, end=len(OPTIMIZATION_PAPER), quote=OPTIMIZATION_PAPER),),
        assumptions=("Synthetic software fixture; no published result is being reproduced.",),
        claim=OptimizationClaim(target=1.0, tolerance=0.2),
        workload=workload,
    )
    return engine.freeze(spec)
