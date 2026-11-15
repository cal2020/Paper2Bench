"""Predeclared paired accuracy comparisons over two frozen experiment runs.

The percentile bootstrap resamples paired test examples, conditional on the two
fixed fitted systems. It does not estimate variability across training runs or
establish paper reproduction. Runtime resources may differ and are disclosed.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, StrictFloat, StrictInt

from .evaluation import evaluate
from .models import (
    SHA256,
    Dataset,
    FrozenModel,
    canonical_bytes,
    digest,
    parse_dataset,
    parse_predictions,
)

if TYPE_CHECKING:
    from .engine import Engine


MAX_BOOTSTRAP_DRAWS = 20_000_000


class InvalidComparison(ValueError):
    """A comparison plan or its run artifacts violate the paired contract."""


class ComparisonSpec(FrozenModel):
    candidate_spec_hash: SHA256
    baseline_spec_hash: SHA256
    minimum_effect: Annotated[StrictFloat, Field(ge=0, le=1)] = 0.0
    confidence: Annotated[StrictFloat, Field(gt=0, lt=1)] = 0.95
    bootstrap_samples: Annotated[StrictInt, Field(ge=200, le=20000)] = 2000
    seed: Annotated[StrictInt, Field(ge=0)] = 0
    min_test_examples: Annotated[StrictInt, Field(gt=0)] = 20


def _check_bootstrap_budget(sample_count: int, plan: ComparisonSpec) -> None:
    if sample_count * plan.bootstrap_samples > MAX_BOOTSTRAP_DRAWS:
        raise InvalidComparison(
            f"bootstrap work exceeds the limit of {MAX_BOOTSTRAP_DRAWS} paired draws"
        )


def _compatible_specs(engine: Engine, plan: ComparisonSpec):
    candidate = engine.load_spec(plan.candidate_spec_hash)
    baseline = engine.load_spec(plan.baseline_spec_hash)
    if candidate.dataset_sha256 != baseline.dataset_sha256:
        raise InvalidComparison("paired comparisons require the exact same dataset artifact")
    if (
        candidate.schema_version != baseline.schema_version
        or candidate.claim.metric != baseline.claim.metric
        or candidate.engine_identity_sha256 != baseline.engine_identity_sha256
    ):
        raise InvalidComparison("paired comparisons require the same frozen evaluator")
    dataset = parse_dataset(engine.artifacts.get_json(candidate.dataset_sha256))
    _check_bootstrap_budget(
        len(dataset.test if isinstance(dataset, Dataset) else dataset.instances), plan
    )
    return candidate, baseline


def freeze_comparison(engine: Engine, value: dict | ComparisonSpec) -> str:
    """Validate both frozen experiment identities and store the plan before execution."""
    raw = value.model_dump(mode="json") if isinstance(value, ComparisonSpec) else value
    plan = ComparisonSpec.model_validate(raw)
    _compatible_specs(engine, plan)
    return engine.artifacts.put(canonical_bytes(plan))


def _load_plan(engine: Engine, plan_hash: str) -> ComparisonSpec:
    raw = engine.artifacts.get(plan_hash)
    plan = ComparisonSpec.model_validate_json(raw)
    if canonical_bytes(plan) != raw:
        raise InvalidComparison("comparison plan is not in canonical frozen form")
    return plan


def _run_key(key: str, plan_hash: str, role: str) -> str:
    return "comparison:" + digest(
        {
            "namespace": "paper2bench.paired-comparison.v1",
            "key": key,
            "plan_hash": plan_hash,
            "role": role,
        }
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Linearly interpolated empirical quantile, including exact endpoints."""
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _paired_assessment(dataset, candidate_result, baseline_result, candidate, baseline, plan):
    # Recompute trusted correctness from predictions, rather than reusing a
    # candidate-supplied metric or relying on previous assessment contents.
    candidate_score = evaluate(
        dataset, parse_predictions(candidate_result["predictions"], dataset), candidate.claim
    )
    baseline_score = evaluate(
        dataset, parse_predictions(baseline_result["predictions"], dataset), baseline.claim
    )
    classification = isinstance(dataset, Dataset)
    value_key = "correct" if classification else "score"
    pairs = [
        {
            "id": item["id"],
            ("candidate_correct" if classification else "candidate_score"): item[value_key],
            ("baseline_correct" if classification else "baseline_score"): reference[value_key],
            "difference": float(item[value_key]) - float(reference[value_key]),
        }
        for item, reference in zip(
            candidate_score["per_item"], baseline_score["per_item"], strict=True
        )
    ]
    differences = [item["difference"] for item in pairs]
    count = len(differences)
    _check_bootstrap_budget(count, plan)
    effect = sum(differences) / count
    if all(value == differences[0] for value in differences):
        # The ordinary paired bootstrap is exactly degenerate in this case.
        distribution = [effect] * plan.bootstrap_samples
    else:
        rng = random.Random(plan.seed)
        distribution = sorted(
            sum(differences[rng.randrange(count)] for _ in range(count)) / count
            for _ in range(plan.bootstrap_samples)
        )
    tail = (1 - plan.confidence) / 2
    lower = _percentile(distribution, tail)
    upper = _percentile(distribution, 1 - tail)
    minimum_met = count >= plan.min_test_examples
    if not minimum_met:
        verdict = "inconclusive"
    elif lower > plan.minimum_effect:
        verdict = "supported"
    elif upper <= plan.minimum_effect:
        verdict = "not_supported"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "criterion": "paired_accuracy_superiority"
        if classification
        else "paired_optimality_superiority",
        "metric": "accuracy_difference" if classification else "optimality_ratio_difference",
        "effect_definition": "candidate_accuracy_minus_baseline_accuracy"
        if classification
        else "candidate_mean_optimality_minus_baseline_mean_optimality",
        "effect": effect,
        ("candidate_accuracy" if classification else "candidate_mean_score"): candidate_score[
            "accuracy" if classification else "mean_score"
        ],
        ("baseline_accuracy" if classification else "baseline_mean_score"): baseline_score[
            "accuracy" if classification else "mean_score"
        ],
        "minimum_effect": plan.minimum_effect,
        "sample_count": count,
        "min_test_examples": plan.min_test_examples,
        "minimum_sample_count_met": minimum_met,
        "confidence": plan.confidence,
        "confidence_interval": {"lower": lower, "upper": upper, "method": "percentile_bootstrap"},
        "bootstrap_samples": plan.bootstrap_samples,
        "bootstrap_seed": plan.seed,
        "per_item": pairs,
    }


def run_comparison(engine: Engine, plan_hash: str, key: str) -> dict[str, Any]:
    """Register both roles before either executes and retain a deterministic result.

    Repeated invocations reuse the same role runs, including terminal failures.
    Existing active leases may yield a not-tested snapshot; a later invocation
    can report completed runs without repeating their execution.
    """
    if not isinstance(key, str) or not key.strip():
        raise InvalidComparison("a nonempty comparison idempotency key is required")
    plan = _load_plan(engine, plan_hash)
    candidate_spec, baseline_spec = _compatible_specs(engine, plan)
    specifications = {"candidate": candidate_spec, "baseline": baseline_spec}
    references = {
        "candidate": plan.candidate_spec_hash,
        "baseline": plan.baseline_spec_hash,
    }
    # Keep registration separate from execution: a preflight error on either
    # side must never start the other side's potentially billable workload.
    registered = {
        role: engine.submit(references[role], _run_key(key, plan_hash, role))
        for role in ("candidate", "baseline")
    }
    for role in ("candidate", "baseline"):
        engine.execute(registered[role]["run_id"])
    records = {
        role: engine.result(registered[role]["run_id"]) for role in ("candidate", "baseline")
    }
    completed = all(record["run"]["state"] == "completed" for record in records.values())
    if completed:
        for role, record in records.items():
            if (
                record["run"]["spec_hash"] != references[role]
                or record["result"].get("spec_hash") != references[role]
                or record["result"].get("run_id") != registered[role]["run_id"]
                or record["result"].get("execution_state") != "completed"
            ):
                raise InvalidComparison("completed run result does not match its frozen identity")
        dataset = parse_dataset(engine.artifacts.get_json(candidate_spec.dataset_sha256))
        assessment = _paired_assessment(
            dataset,
            records["candidate"]["result"],
            records["baseline"]["result"],
            candidate_spec,
            baseline_spec,
            plan,
        )
    else:
        assessment = {
            "verdict": "not_tested",
            "criterion": "paired_accuracy_superiority"
            if candidate_spec.claim.metric == "accuracy"
            else "paired_optimality_superiority",
            "metric": "accuracy_difference"
            if candidate_spec.claim.metric == "accuracy"
            else "optimality_ratio_difference",
            "reason": "both_runs_must_complete",
            "effect": None,
            "confidence_interval": None,
            "sample_count": 0,
            "min_test_examples": plan.min_test_examples,
            "minimum_sample_count_met": False,
            "minimum_effect": plan.minimum_effect,
            "confidence": plan.confidence,
            "bootstrap_samples": plan.bootstrap_samples,
            "bootstrap_seed": plan.seed,
        }
    roles = {}
    for role, record in records.items():
        spec = specifications[role]
        roles[role] = {
            "spec_hash": references[role],
            "run_id": record["run"]["run_id"],
            "state": record["run"]["state"],
            "result_hash": record["run"]["result_hash"],
            "seed": spec.seed,
            "track": spec.track,
            "workload_kind": spec.workload.kind,
            "resources": {
                "limits": spec.limits.model_dump(mode="json"),
                "policy": spec.policy.model_dump(mode="json"),
                "measurements": (record["result"] or {}).get("measurements"),
            },
            "assumptions": list(spec.assumptions),
        }
    result = {
        "schema_version": "1",
        "kind": "paired_accuracy_comparison",
        "plan_hash": plan_hash,
        "dataset_sha256": candidate_spec.dataset_sha256,
        "candidate": roles["candidate"],
        "baseline": roles["baseline"],
        "assessment": assessment,
        "conditions": {
            "same_dataset": True,
            "same_evaluator": True,
            "equal_resource_limits": candidate_spec.limits == baseline_spec.limits,
            "equal_policies": candidate_spec.policy == baseline_spec.policy,
            "equal_seeds": candidate_spec.seed == baseline_spec.seed,
        },
        "implementation_fidelity": "unverified",
        "protocol_fidelity": "unverified",
        "limitations": [
            "Statistical comparison is conditional on the two fixed fitted systems.",
            "Paired percentile bootstrap assumes independent test cases; clusters are unsupported.",
            "This comparison does not estimate variation across independent training runs.",
            "Percentile intervals are approximate and can be degenerate on small samples.",
            "Differing runtime resources or tuning budgets do not establish resource-matched gains.",
            "A supported numerical comparison does not establish paper reproduction or fidelity.",
        ],
    }
    return {**result, "result_hash": engine.artifacts.put_json(result)}
