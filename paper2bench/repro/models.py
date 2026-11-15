"""Immutable, validated inputs and deterministic identities for experiments.

Arrays are stored as tuples so freezing a model also freezes its collections.
Source quote equality is checked by the freeze operation against the source;
the models check that the declared character span has the correct length.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain a non-whitespace character")
    return value


def _sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("must be a raw lowercase SHA-256 digest")
    return value


def _immutable_image(value: str) -> str:
    if re.fullmatch(r"(?:[A-Za-z0-9._:/-]+@)?sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("image must be an immutable sha256 ID or digest reference")
    return value


Nonblank = Annotated[StrictStr, AfterValidator(_nonblank)]
SHA256 = Annotated[StrictStr, AfterValidator(_sha256)]
ImmutableImage = Annotated[StrictStr, AfterValidator(_immutable_image)]
Probability = Annotated[StrictFloat, Field(ge=0, le=1)]


class FrozenModel(BaseModel):
    """Base for validated JSON inputs; coercion of scalar types is forbidden."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True
    )


class EvidenceAnchor(FrozenModel):
    start: Annotated[StrictInt, Field(ge=0)]
    end: Annotated[StrictInt, Field(gt=0)]
    quote: Annotated[StrictStr, Field(min_length=1)]

    @model_validator(mode="after")
    def matching_span_length(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("evidence span must be nonempty and match the quote character length")
        return self


class Example(FrozenModel):
    id: Nonblank
    x: Annotated[tuple[StrictFloat, ...], Field(min_length=1)]
    y: Nonblank


class Dataset(FrozenModel):
    train: Annotated[tuple[Example, ...], Field(min_length=1)]
    test: Annotated[tuple[Example, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def consistent_dataset(self) -> Self:
        examples = self.train + self.test
        ids = [example.id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("train and test IDs must be unique and disjoint")
        dimension = len(self.train[0].x)
        if any(len(example.x) != dimension for example in examples):
            raise ValueError("all examples must have the same nonzero feature dimension")
        train_labels = {example.y for example in self.train}
        if any(example.y not in train_labels for example in self.test):
            raise ValueError("test labels must be represented in training data")
        return self


MAX_KNAPSACK_DP_WORK = 20_000_000


class KnapsackInstance(FrozenModel):
    """A bounded 0-1 knapsack problem; every item may be chosen at most once."""

    id: Nonblank
    weights: Annotated[
        tuple[Annotated[StrictInt, Field(gt=0)], ...], Field(min_length=1, max_length=100)
    ]
    values: Annotated[
        tuple[Annotated[StrictInt, Field(ge=0)], ...], Field(min_length=1, max_length=100)
    ]
    capacity: Annotated[StrictInt, Field(ge=0, le=10000)]

    @model_validator(mode="after")
    def matching_items(self) -> Self:
        if len(self.weights) != len(self.values):
            raise ValueError("knapsack weights and values must have equal nonempty lengths")
        return self


class KnapsackDataset(FrozenModel):
    task: Literal["knapsack"] = "knapsack"
    instances: Annotated[tuple[KnapsackInstance, ...], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def bounded_unique_instances(self) -> Self:
        ids = [instance.id for instance in self.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("knapsack instance IDs must be unique")
        # This conservative bound includes unreachable capacity states and
        # overweight items, so it cannot underestimate the evaluator's loops.
        work = sum(len(instance.weights) * (instance.capacity + 1) for instance in self.instances)
        if work > MAX_KNAPSACK_DP_WORK:
            raise ValueError("knapsack exact-evaluation work exceeds 20000000 states")
        return self


class ClaimSpec(FrozenModel):
    metric: Literal["accuracy"] = "accuracy"
    target: Probability
    tolerance: Annotated[StrictFloat, Field(gt=0, le=1)] = 0.1
    confidence: Annotated[StrictFloat, Field(gt=0, lt=1)] = 0.95
    min_test_examples: Annotated[StrictInt, Field(gt=0)] = 20


class OptimizationClaim(FrozenModel):
    metric: Literal["optimality_ratio"] = "optimality_ratio"
    target: Probability
    tolerance: Annotated[StrictFloat, Field(gt=0, le=1)] = 0.1
    confidence: Annotated[StrictFloat, Field(gt=0, lt=1)] = 0.95
    min_test_examples: Annotated[StrictInt, Field(gt=0)] = 20


class Limits(FrozenModel):
    timeout_seconds: Annotated[StrictFloat, Field(gt=0, le=3600)] = 30.0
    memory_mb: Annotated[StrictInt, Field(ge=16, le=65536)] = 256
    output_bytes: Annotated[StrictInt, Field(gt=0, le=16 * 1024 * 1024)] = 1048576


class Policy(FrozenModel):
    local_only: StrictBool = True
    placement: Literal["local", "remote"] = "local"
    budget_id: (
        Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")] | None
    ) = None
    max_provider_calls: Annotated[StrictInt, Field(ge=0, le=10000)] = 0
    max_output_tokens: Annotated[StrictInt, Field(gt=0, le=32768)] = 256


class WorkloadSpec(FrozenModel):
    kind: Literal["builtin_centroid", "builtin_knapsack", "python", "model_classification"]
    implementation_sha256: SHA256 | None = None
    image: ImmutableImage | None = None
    provider_profile: Nonblank | None = None
    model: Nonblank | None = None
    prompt: Nonblank | None = None

    @model_validator(mode="after")
    def valid_workload_fields(self) -> Self:
        implementation = (self.implementation_sha256, self.image)
        provider = (self.provider_profile, self.model, self.prompt)
        if self.kind in {"builtin_centroid", "builtin_knapsack"}:
            if any(value is not None for value in implementation + provider):
                raise ValueError(
                    "builtin workloads do not accept implementation or provider fields"
                )
        elif self.kind == "python":
            if any(value is None for value in implementation):
                raise ValueError("python requires implementation_sha256 and an immutable image")
            if any(value is not None for value in provider):
                raise ValueError("python does not accept provider fields")
        else:
            if any(value is None for value in provider):
                raise ValueError(
                    "model_classification requires provider_profile, model, and prompt"
                )
            if any(value is not None for value in implementation):
                raise ValueError(
                    "model_classification does not accept implementation or image fields"
                )
        return self


class ExperimentSpec(FrozenModel):
    schema_version: Literal["1"] = "1"
    title: Nonblank
    track: Literal["reimplementation", "author_code", "adaptation"] = "reimplementation"
    source_sha256: SHA256
    dataset_sha256: SHA256
    authoring_receipt_sha256: SHA256 | None = None
    source_manifest_sha256: SHA256 | None = None
    engine_identity_sha256: SHA256 | None = None
    evidence: Annotated[tuple[EvidenceAnchor, ...], Field(min_length=1)]
    assumptions: tuple[Nonblank, ...] = ()
    unresolved: tuple[Nonblank, ...] = ()
    claim: ClaimSpec | OptimizationClaim
    workload: WorkloadSpec
    seed: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)] = 0
    limits: Limits = Limits()
    policy: Policy = Policy()


class Predictions(FrozenModel):
    ids: Annotated[tuple[Nonblank, ...], Field(min_length=1)]
    labels: Annotated[tuple[Nonblank, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def aligned_predictions(self) -> Self:
        if len(self.ids) != len(self.labels):
            raise ValueError("prediction IDs and labels must have equal lengths")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("prediction IDs must be unique")
        return self


class OptimizationPredictions(FrozenModel):
    ids: Annotated[tuple[Nonblank, ...], Field(min_length=1)]
    selections: Annotated[tuple[tuple[StrictInt, ...], ...], Field(min_length=1)]

    @model_validator(mode="after")
    def aligned_selections(self) -> Self:
        if len(self.ids) != len(self.selections):
            raise ValueError("prediction IDs and selections must have equal lengths")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("prediction IDs must be unique")
        return self


def parse_dataset(value: Any) -> Dataset | KnapsackDataset:
    """Select the task schema and revalidate model instances at input boundaries."""
    raw = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
    if isinstance(raw, dict) and (
        raw.get("task") == "knapsack" or ("instances" in raw and "task" not in raw)
    ):
        return KnapsackDataset.model_validate(raw)
    return Dataset.model_validate(raw)


def parse_predictions(
    value: Any, dataset: Dataset | KnapsackDataset | Literal["classification", "knapsack"]
) -> Predictions | OptimizationPredictions:
    """Parse task-specific output without needing reference answers.

    Remote execution may use a task string because its input deliberately omits
    protected labels. Exact IDs and mathematical feasibility are checked by the
    trusted evaluator, rather than inferred by this structural parser.
    """
    raw = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
    if isinstance(dataset, KnapsackDataset) or dataset == "knapsack":
        return OptimizationPredictions.model_validate(raw)
    if isinstance(dataset, Dataset) or dataset == "classification":
        return Predictions.model_validate(raw)
    raise ValueError("unsupported prediction task")


def _json_value(value: Any) -> Any:
    """Restrict identities to JSON values; never stringify unsupported objects."""
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python"))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot contain non-finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 JSON, with sorted keys and no non-finite numbers."""
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: Any) -> str:
    """SHA-256 identity of canonical JSON (use the artifact store for raw bytes)."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
