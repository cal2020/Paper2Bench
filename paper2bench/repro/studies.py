"""Prespecified paired classification studies across seeds and shared test cases.

The crossed bootstrap resamples seed pairs and test cases as two factors. It
does not verify candidate seed consumption, independence of stochastic fits,
implementation fidelity, a causal effect, or paper reproduction.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, StrictFloat, StrictInt, model_validator

from .evaluation import evaluate
from .models import SHA256, Dataset, FrozenModel, Predictions, canonical_bytes, digest

if TYPE_CHECKING:
    from .engine import Engine


MAX_BOOTSTRAP_DRAWS = 20_000_000
_ROLES = ("candidate", "baseline")


class InvalidStudy(ValueError):
    """A study or its trial artifacts violate the frozen paired contract."""


class StudySpec(FrozenModel):
    candidate_spec_hash: SHA256
    baseline_spec_hash: SHA256
    seeds: Annotated[
        tuple[Annotated[StrictInt, Field(ge=0, le=2**63 - 1)], ...],
        Field(min_length=2, max_length=50),
    ]
    minimum_effect: Annotated[StrictFloat, Field(ge=0, le=1)] = 0.0
    confidence: Annotated[StrictFloat, Field(gt=0, lt=1)] = 0.95
    bootstrap_samples: Annotated[StrictInt, Field(ge=200, le=20000)] = 2000
    min_test_examples: Annotated[StrictInt, Field(gt=0)] = 20

    @model_validator(mode="after")
    def unique_seeds(self) -> Self:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("study seeds must be unique")
        return self


class _Trial(FrozenModel):
    role: Literal["candidate", "baseline"]
    seed: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    spec_hash: SHA256


class _FrozenPlan(FrozenModel):
    schema_version: Literal["1"] = "1"
    kind: Literal["paired_accuracy_study_plan"] = "paired_accuracy_study_plan"
    study: StudySpec
    dataset_sha256: SHA256
    evaluator_identity_sha256: SHA256
    bootstrap_seed: Annotated[StrictInt, Field(ge=0)]
    trials: Annotated[tuple[_Trial, ...], Field(min_length=4, max_length=100)]


def _bootstrap_seed(study: StudySpec) -> int:
    return int(digest({"namespace": "paper2bench.crossed-bootstrap.v1", "study": study})[:16], 16)


def _check_budget(sample_count: int, study: StudySpec) -> None:
    if sample_count * len(study.seeds) * study.bootstrap_samples > MAX_BOOTSTRAP_DRAWS:
        raise InvalidStudy(
            f"bootstrap work exceeds the limit of {MAX_BOOTSTRAP_DRAWS} paired draws"
        )


def _compatible_specs(engine: Engine, study: StudySpec):
    candidate = engine.load_spec(study.candidate_spec_hash)
    baseline = engine.load_spec(study.baseline_spec_hash)
    if candidate.dataset_sha256 != baseline.dataset_sha256:
        raise InvalidStudy("paired studies require the exact same dataset artifact")
    if (
        candidate.schema_version != baseline.schema_version
        or candidate.claim.metric != "accuracy"
        or baseline.claim.metric != "accuracy"
        or candidate.engine_identity_sha256 != baseline.engine_identity_sha256
    ):
        raise InvalidStudy("paired studies require the same frozen classification evaluator")
    for spec in (candidate, baseline):
        if spec.workload.kind not in {"builtin_centroid", "python"}:
            raise InvalidStudy(
                "multi-seed studies support builtin_centroid or python; model seed control is unsupported"
            )
        engine._validate(spec)
    dataset = Dataset.model_validate(engine.artifacts.get_json(candidate.dataset_sha256))
    _check_budget(len(dataset.test), study)
    return {"candidate": candidate, "baseline": baseline}, dataset


def freeze_study(engine: Engine, value: dict | StudySpec) -> str:
    """Resolve and freeze every trial before registering or executing any job."""
    raw = value.model_dump(mode="json") if isinstance(value, StudySpec) else value
    study = StudySpec.model_validate(raw)
    originals, _dataset = _compatible_specs(engine, study)
    trials = []
    for seed in study.seeds:
        for role in _ROLES:
            child = originals[role].model_copy(update={"seed": seed})
            trials.append(_Trial(role=role, seed=seed, spec_hash=engine.freeze(child)))
    plan = _FrozenPlan(
        study=study,
        dataset_sha256=originals["candidate"].dataset_sha256,
        evaluator_identity_sha256=originals["candidate"].engine_identity_sha256,
        bootstrap_seed=_bootstrap_seed(study),
        trials=tuple(trials),
    )
    return engine.artifacts.put(canonical_bytes(plan))


def _load_plan(engine: Engine, plan_hash: str):
    raw = engine.artifacts.get(plan_hash)
    plan = _FrozenPlan.model_validate_json(raw)
    if canonical_bytes(plan) != raw:
        raise InvalidStudy("study plan is not in canonical frozen form")
    originals, dataset = _compatible_specs(engine, plan.study)
    if (
        plan.dataset_sha256 != originals["candidate"].dataset_sha256
        or plan.evaluator_identity_sha256 != originals["candidate"].engine_identity_sha256
        or plan.bootstrap_seed != _bootstrap_seed(plan.study)
    ):
        raise InvalidStudy("study plan identity does not match its frozen experiment contracts")
    expected_schedule = [(role, seed) for seed in plan.study.seeds for role in _ROLES]
    if [(trial.role, trial.seed) for trial in plan.trials] != expected_schedule:
        raise InvalidStudy("study trial schedule differs from the prespecified paired seeds")
    for trial in plan.trials:
        child = engine.load_spec(trial.spec_hash)
        expected = originals[trial.role].model_copy(update={"seed": trial.seed})
        if canonical_bytes(child) != canonical_bytes(expected):
            raise InvalidStudy("trial specification changed fields beyond the declared seed")
        engine._validate(child)
    return plan, originals, dataset


def _trial_key(plan_hash: str, key: str, trial: _Trial) -> str:
    return "study:" + digest(
        {
            "namespace": "paper2bench.seed-study.v1",
            "plan_hash": plan_hash,
            "key": key,
            "role": trial.role,
            "seed": trial.seed,
        }
    )


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _assessment(dataset, plan, originals, records):
    by_trial = {(trial.role, trial.seed): record for trial, record in records}
    matrix, between_seeds = [], []
    candidate_accuracies, baseline_accuracies = [], []
    for seed in plan.study.seeds:
        scores = {}
        for role in _ROLES:
            record = by_trial[(role, seed)]
            scores[role] = evaluate(
                dataset,
                Predictions.model_validate(record["result"]["predictions"]),
                originals[role].claim,
            )
        differences = [
            int(candidate["correct"]) - int(baseline["correct"])
            for candidate, baseline in zip(
                scores["candidate"]["per_item"], scores["baseline"]["per_item"], strict=True
            )
        ]
        matrix.append(differences)
        candidate_accuracies.append(scores["candidate"]["accuracy"])
        baseline_accuracies.append(scores["baseline"]["accuracy"])
        between_seeds.append(
            {
                "seed": seed,
                "candidate_accuracy": scores["candidate"]["accuracy"],
                "baseline_accuracy": scores["baseline"]["accuracy"],
                "effect": sum(differences) / len(differences),
            }
        )

    seeds, cases = len(matrix), len(dataset.test)
    _check_budget(cases, plan.study)
    effect = sum(sum(row) for row in matrix) / (seeds * cases)
    if all(value == matrix[0][0] for row in matrix for value in row):
        distribution = [effect] * plan.study.bootstrap_samples
    else:
        rng = random.Random(plan.bootstrap_seed)
        distribution = []
        for _ in range(plan.study.bootstrap_samples):
            # One case sample is shared across all selected seed pairs. Drawing
            # new cases for each seed would silently replace the crossed design.
            seed_indices = [rng.randrange(seeds) for _ in range(seeds)]
            case_indices = [rng.randrange(cases) for _ in range(cases)]
            total = sum(matrix[seed][case] for seed in seed_indices for case in case_indices)
            distribution.append(total / (seeds * cases))
        distribution.sort()
    tail = (1 - plan.study.confidence) / 2
    lower = _percentile(distribution, tail)
    upper = _percentile(distribution, 1 - tail)
    minimum_met = cases >= plan.study.min_test_examples and seeds >= 2
    if not minimum_met:
        verdict = "inconclusive"
    elif lower > plan.study.minimum_effect:
        verdict = "supported"
    elif upper <= plan.study.minimum_effect:
        verdict = "not_supported"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "criterion": "paired_multiseed_accuracy_superiority",
        "metric": "accuracy_difference",
        "effect_definition": "mean_candidate_accuracy_minus_mean_baseline_accuracy",
        "effect": effect,
        "candidate_accuracy": math.fsum(candidate_accuracies) / seeds,
        "baseline_accuracy": math.fsum(baseline_accuracies) / seeds,
        "minimum_effect": plan.study.minimum_effect,
        "sample_count": cases,
        "seed_count": seeds,
        "paired_observation_count": seeds * cases,
        "min_test_examples": plan.study.min_test_examples,
        "minimum_sample_count_met": minimum_met,
        "confidence": plan.study.confidence,
        "confidence_interval": {
            "lower": lower,
            "upper": upper,
            "method": "two_way_crossed_percentile_bootstrap",
        },
        "bootstrap_samples": plan.study.bootstrap_samples,
        "bootstrap_seed": plan.bootstrap_seed,
        "between_seed_effects": between_seeds,
        "between_seed_effect_standard_deviation": statistics.stdev(
            row["effect"] for row in between_seeds
        ),
        "seed_order": list(plan.study.seeds),
        "case_order": [example.id for example in dataset.test],
        "paired_correctness_differences": matrix,
    }


def _resources(trials):
    states = Counter(trial["state"] for trial in trials)
    walls, costs = [], []
    prior_attempts = sum(max(0, len(trial["attempts"]) - 1) for trial in trials)
    for trial in trials:
        values = trial["measurements"] or {}
        wall = values.get("wall_seconds", trial["failure_wall_seconds"])
        if type(wall) in (int, float) and math.isfinite(wall) and wall >= 0:
            walls.append(wall)
        cost = values.get("cost_usd")
        if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
            costs.append(cost)
    return {
        "trial_count": len(trials),
        "states": dict(sorted(states.items())),
        "completed_trials": states.get("completed", 0),
        "attempt_count": sum(len(trial["attempts"]) for trial in trials),
        "earlier_attempts_without_usage": prior_attempts,
        "wall_seconds_known_total": math.fsum(walls),
        "wall_seconds_unknown_trials": len(trials) - len(walls),
        "wall_seconds_unknown_attempts": len(trials) - len(walls) + prior_attempts,
        "cost_usd_known_total": math.fsum(costs),
        "cost_usd_unknown_trials": len(trials) - len(costs),
        "cost_usd_unknown_attempts": len(trials) - len(costs) + prior_attempts,
        "cost_interpretation": "unknown monetary costs are not treated as zero",
    }


def run_study(engine: Engine, plan_hash: str, key: str) -> dict:
    """Register all trials before execution; retain failures and never select winners."""
    if not isinstance(key, str) or not key.strip() or len(key) > 4096:
        raise InvalidStudy(
            "a nonempty study idempotency key of at most 4096 characters is required"
        )
    plan, originals, dataset = _load_plan(engine, plan_hash)
    registered = [
        (trial, engine.submit(trial.spec_hash, _trial_key(plan_hash, key, trial)))
        for trial in plan.trials
    ]
    for _trial, run in registered:
        engine.execute(run["run_id"])
    records = [(trial, engine.result(run["run_id"])) for trial, run in registered]
    all_completed = all(record["run"]["state"] == "completed" for _, record in records)
    trial_ledger = []
    for trial, record in records:
        run, result = record["run"], record["result"]
        if run["spec_hash"] != trial.spec_hash:
            raise InvalidStudy("trial run identity differs from the frozen study plan")
        if run["state"] == "completed" and (
            not result
            or result.get("spec_hash") != trial.spec_hash
            or result.get("run_id") != run["run_id"]
            or result.get("execution_state") != "completed"
        ):
            raise InvalidStudy("completed trial result does not match its run identity")
        trial_ledger.append(
            {
                "role": trial.role,
                "seed": trial.seed,
                "spec_hash": trial.spec_hash,
                "run_id": run["run_id"],
                "state": run["state"],
                "result_hash": run["result_hash"],
                "attempts": run["attempts"],
                "measurements": (result or {}).get("measurements"),
                "failure_wall_seconds": (result or {}).get("wall_seconds"),
                "limits": originals[trial.role].limits.model_dump(mode="json"),
                "policy": originals[trial.role].policy.model_dump(mode="json"),
                "workload_kind": originals[trial.role].workload.kind,
                "assumptions": list(originals[trial.role].assumptions),
            }
        )
    if all_completed:
        assessment = _assessment(dataset, plan, originals, records)
    else:
        assessment = {
            "verdict": "not_tested",
            "criterion": "paired_multiseed_accuracy_superiority",
            "metric": "accuracy_difference",
            "reason": "all_prespecified_trials_must_complete",
            "effect": None,
            "confidence_interval": None,
            "sample_count": len(dataset.test),
            "seed_count": len(plan.study.seeds),
            "minimum_effect": plan.study.minimum_effect,
            "min_test_examples": plan.study.min_test_examples,
            "minimum_sample_count_met": False,
            "bootstrap_samples": plan.study.bootstrap_samples,
            "bootstrap_seed": plan.bootstrap_seed,
            "confidence": plan.study.confidence,
        }
    result = {
        "schema_version": "1",
        "kind": "paired_accuracy_study",
        "plan_hash": plan_hash,
        "dataset_sha256": plan.dataset_sha256,
        "evaluator_identity_sha256": plan.evaluator_identity_sha256,
        "original_specs": {
            "candidate": plan.study.candidate_spec_hash,
            "baseline": plan.study.baseline_spec_hash,
        },
        "seeds": list(plan.study.seeds),
        "trials": trial_ledger,
        "assessment": assessment,
        "resources": {
            role: _resources([trial for trial in trial_ledger if trial["role"] == role])
            for role in _ROLES
        },
        "conditions": {
            "same_dataset": True,
            "same_evaluator": True,
            "same_seed_schedule": True,
            "equal_limits": originals["candidate"].limits == originals["baseline"].limits,
            "equal_policies": originals["candidate"].policy == originals["baseline"].policy,
        },
        "implementation_fidelity": "unverified",
        "protocol_fidelity": "unverified",
        "limitations": [
            "Seeds are passed to Python input; actual seed consumption and independent fits are unverified.",
            "The builtin centroid algorithm is deterministic; repeated seeds do not add stochastic evidence.",
            "The crossed percentile bootstrap assumes exchangeable seed pairs and independent test cases.",
            "Seed sensitivity is conditional on the supplied training data and implementation.",
            "Clustered cases, dataset resampling, tuning selection, and sequential testing are unsupported.",
            "Small seed sets and percentile intervals can be unstable or degenerate.",
            "All failed, exhausted, cancelled, or incomplete trials block the primary numerical verdict.",
            "Resource differences and unknown monetary costs prevent an automatic efficiency claim.",
            "Numerical superiority does not establish causality, implementation fidelity, or paper reproduction.",
        ],
    }
    return {**result, "result_hash": engine.artifacts.put_json(result)}
