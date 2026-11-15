"""Fenced remote Python execution with server-side scoring and no hidden labels.

Authentication belongs to the HTTP boundary. Worker identities must come from
that boundary. Workers receive only executable code and public task inputs;
worker execution and measurements are declarations, not remote attestation.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from .evaluation import evaluate
from .models import (
    SHA256,
    FrozenModel,
    ImmutableImage,
    KnapsackDataset,
    Limits,
    canonical_bytes,
    digest,
    parse_dataset,
    parse_predictions,
)
from .runtime import sanitized_input


class RemoteConflict(ValueError):
    """An assignment is stale, belongs to another worker, or conflicts with a receipt."""


class WorkerMeasurements(FrozenModel):
    """Bounded metadata only; arbitrary scores, logs and credentials are forbidden."""

    runtime: Literal["docker"] | None = None
    task: Literal["classification", "knapsack"] | None = None
    image: ImmutableImage | None = None
    wall_seconds: Annotated[StrictFloat, Field(ge=0)] | None = None
    stdout_bytes: Annotated[StrictInt, Field(ge=0, le=64 * 1024 * 1024)] | None = None
    stderr_bytes: Annotated[StrictInt, Field(ge=0, le=64 * 1024 * 1024)] | None = None
    memory_limit_mb: Annotated[StrictInt, Field(ge=16, le=65536)] | None = None
    memory_limit_enforced: StrictBool | None = None
    network_used: StrictBool | None = None
    isolation_scope: Literal["single_user_container"] | None = None
    cleanup_confirmed: StrictBool | None = None
    candidate_uid: Literal[65534] | None = None
    supervisor_deadline_enforced: StrictBool | None = None
    container_name: (
        Annotated[StrictStr, Field(pattern=r"^paper2bench-[a-f0-9]+$", max_length=64)] | None
    ) = None
    worker_engine_identity_sha256: SHA256 | None = None
    worker_python_version: Annotated[StrictStr, Field(max_length=64)] | None = None
    worker_platform: Annotated[StrictStr, Field(max_length=256)] | None = None


class WorkerCompletion(FrozenModel):
    state: Literal["completed", "failed", "unsupported", "resource_exhausted", "cancelled"]
    input_sha256: SHA256
    implementation_sha256: SHA256
    image: ImmutableImage
    predictions: dict[str, Any] | None = None
    measurements: WorkerMeasurements = WorkerMeasurements()
    error: Annotated[StrictStr, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")] | None = None

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.state == "completed":
            if self.predictions is None or self.error is not None:
                raise ValueError("Completed work requires predictions and no error code.")
        elif self.predictions is not None:
            raise ValueError("Unsuccessful work must not supply predictions.")
        return self


class WorkerAssignment(FrozenModel):
    schema_version: Literal["1"] = "1"
    worker_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    run_id: StrictStr
    token: Annotated[StrictInt, Field(gt=0)]
    attempt: Annotated[StrictInt, Field(gt=0)]
    lease_seconds: Annotated[StrictFloat, Field(gt=0, le=3600)]
    spec_hash: SHA256
    implementation_sha256: SHA256
    input_sha256: SHA256
    image: ImmutableImage
    limits: Limits
    seed: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    task: Literal["classification", "knapsack"]
    code: Annotated[StrictStr, Field(max_length=256 * 1024)]
    input: dict[str, Any]


def _worker(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
    ):
        raise ValueError("Worker identifier must contain 1-128 safe identifier characters.")
    return "remote:" + value


def _duration(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 3600
    ):
        raise ValueError("Remote lease duration must be finite and at most 3600 seconds.")
    return float(value)


class RemoteCoordinator:
    """Coordinate remote leases without exposing source text or reference answers.

    A separate SQLite write lock serializes claim selection and completion
    deduplication across API processes. RunStore remains the sole lease/result
    authority, including races with administrator cancellation.
    """

    def __init__(self, engine):
        self.engine = engine
        self._mutex_path = Path(engine.root) / "remote.sqlite3"
        connection = sqlite3.connect(self._mutex_path, timeout=30)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS coordinator_lock (id INTEGER PRIMARY KEY)"
            )
            connection.commit()
        finally:
            connection.close()

    @contextmanager
    def _serialized(self) -> Iterator[None]:
        connection = sqlite3.connect(self._mutex_path, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.commit()
        finally:
            connection.close()

    def _assignment(self, run: dict, worker_id: str, lease_seconds: float) -> dict:
        spec = self.engine.load_spec(run["spec_hash"])
        dataset = self.engine._validate(spec)
        if (
            spec.workload.kind != "python"
            or spec.policy.placement != "remote"
            or spec.policy.local_only
        ):
            raise RemoteConflict("Run is not authorized for remote Python execution.")
        public = sanitized_input(dataset, spec)
        code = self.engine.artifacts.get(spec.workload.implementation_sha256).decode("utf-8")
        assignment = WorkerAssignment(
            worker_id=worker_id,
            run_id=run["run_id"],
            token=run["token"],
            attempt=run["attempt"],
            lease_seconds=lease_seconds,
            spec_hash=run["spec_hash"],
            implementation_sha256=spec.workload.implementation_sha256,
            input_sha256=self.engine.artifacts.put_json(public),
            image=spec.workload.image,
            limits=spec.limits,
            seed=spec.seed,
            task="knapsack" if isinstance(dataset, KnapsackDataset) else "classification",
            code=code,
            input=public,
        )
        return assignment.model_dump(mode="json")

    def claim(
        self, worker_id: str, images: tuple[str, ...], max_memory_mb: int, lease_seconds: float = 30
    ) -> dict | None:
        owner, duration = _worker(worker_id), _duration(lease_seconds)
        if (
            not isinstance(images, tuple)
            or len(images) > 256
            or any(not isinstance(image, str) for image in images)
        ):
            raise ValueError("Worker must advertise at most 256 immutable images.")
        for image in images:
            # Reuse the assignment model's exact immutable-image field validator.
            from pydantic import TypeAdapter

            TypeAdapter(ImmutableImage).validate_python(image)
        if type(max_memory_mb) is not int or not 16 <= max_memory_mb <= 65536:
            raise ValueError("Worker memory capacity must be an integer from 16 to 65536 MiB.")
        with self._serialized():
            runs = self.engine.runs.list_runs()
            for run in runs:
                if (
                    run["worker"] == owner
                    and run["state"] == "running"
                    and self.engine.runs.is_active(run["run_id"], run["token"])
                ):
                    if self.engine.runs.heartbeat(run["run_id"], run["token"], duration):
                        return self._assignment(run, worker_id, duration)
            for run in runs:
                if run["state"] not in {"queued", "running"}:
                    continue
                spec = self.engine.load_spec(run["spec_hash"])
                if (
                    spec.workload.kind != "python"
                    or spec.policy.placement != "remote"
                    or spec.policy.local_only
                    or spec.workload.image not in images
                    or spec.limits.memory_mb > max_memory_mb
                ):
                    continue
                lease = self.engine.runs.claim(run["run_id"], owner, duration)
                if lease is not None:
                    return self._assignment(lease, worker_id, duration)
        return None

    def heartbeat(self, worker_id: str, run_id: str, token: int, lease_seconds: float = 30) -> bool:
        owner, duration = _worker(worker_id), _duration(lease_seconds)
        run = self.engine.runs.get_run(run_id)
        if run["worker"] != owner or type(token) is not int or run["token"] != token:
            return False
        return self.engine.runs.heartbeat(run_id, token, duration)

    def _duplicate(self, run: dict, owner: str, token: int, completion_hash: str) -> bool:
        known_attempt = next((item for item in run["attempts"] if item["token"] == token), None)
        if known_attempt is None or known_attempt["worker"] != owner:
            return False
        candidates = []
        if run["result_hash"]:
            candidates.append(run["result_hash"])
        if run["state"] == "cancelled":
            candidates.extend(
                item["artifact_sha256"]
                for item in self.engine.runs.diagnostics(run["run_id"])
                if item["token"] == token
            )
        for ref in candidates:
            result = self.engine.artifacts.get_json(ref)
            if (
                not result.get("diagnostic_only", False)
                and result.get("remote_completion_sha256") == completion_hash
                and result.get("remote_worker") == owner
                and result.get("remote_token") == token
            ):
                return True
        return False

    def _record_stale_failure(
        self,
        run: dict,
        owner: str,
        token: int,
        parsed: WorkerCompletion,
        normalized: dict,
        completion_hash: str,
    ) -> None:
        """Preserve cleanup evidence without accepting a stale scientific result.

        A known worker may report failure after a lost lease or administrator
        cancellation. The report is an attempt diagnostic only; completion still
        returns a conflict so the client archives its fenced receipt.
        """
        attempt = next((item for item in run["attempts"] if item["token"] == token), None)
        if attempt is None or attempt["worker"] != owner or parsed.state == "completed":
            return
        spec = self.engine.load_spec(run["spec_hash"])
        if (
            spec.workload.kind != "python"
            or spec.policy.placement != "remote"
            or spec.policy.local_only
        ):
            return
        # Resolve only the immutable public input. Source documents, provider
        # configuration and the current server version are irrelevant to cleanup.
        dataset = parse_dataset(self.engine.artifacts.get_json(spec.dataset_sha256))
        expected_input = self.engine.artifacts.put_json(sanitized_input(dataset, spec))
        if (
            parsed.input_sha256 != expected_input
            or parsed.implementation_sha256 != spec.workload.implementation_sha256
            or parsed.image != spec.workload.image
            or parsed.measurements.image not in {None, spec.workload.image}
        ):
            return
        self.engine.artifacts.put_json(normalized)
        artifact = self.engine.artifacts.put_json(
            {
                "schema_version": "1",
                "run_id": run["run_id"],
                "spec_hash": run["spec_hash"],
                "execution_state": parsed.state,
                "diagnostic_only": True,
                "assessment": {
                    "verdict": "not_tested",
                    "reason": parsed.error or "remote_stale_failure",
                },
                "measurements": {
                    **parsed.measurements.model_dump(mode="json", exclude_none=True),
                    "worker_id": owner.removeprefix("remote:"),
                    "measurement_source": "remote_worker_reported",
                    "runtime_attestation": "not_verified",
                },
                "provider_calls": [],
                "remote_completion_sha256": completion_hash,
                "remote_worker": owner,
                "remote_token": token,
                "input_sha256": expected_input,
                "implementation_sha256": spec.workload.implementation_sha256,
                "limitations": [
                    "Fenced attempt diagnostic only; this is not an accepted execution result.",
                    "Remote cleanup and execution are worker-reported, not attested.",
                ],
            }
        )
        self.engine.runs.attach_diagnostic(run["run_id"], token, artifact)

    def complete(self, worker_id: str, run_id: str, token: int, completion: dict) -> dict:
        owner = _worker(worker_id)
        if type(token) is not int or token <= 0:
            raise RemoteConflict("Invalid assignment token.")
        parsed = WorkerCompletion.model_validate(completion)
        normalized = parsed.model_dump(mode="json", exclude_none=True)
        if len(canonical_bytes(normalized)) > 17 * 1024 * 1024:
            raise ValueError("Remote completion exceeds 17 MiB.")
        completion_hash = digest(normalized)
        with self._serialized():
            run = self.engine.runs.get_run(run_id)
            if self._duplicate(run, owner, token, completion_hash):
                return run
            if (
                run["worker"] != owner
                or run["token"] != token
                or not self.engine.runs.is_active(run_id, token)
            ):
                self._record_stale_failure(run, owner, token, parsed, normalized, completion_hash)
                raise RemoteConflict("Assignment is no longer active for this worker.")
            spec = self.engine.load_spec(run["spec_hash"])
            dataset = self.engine._validate(spec)
            if (
                spec.workload.kind != "python"
                or spec.policy.placement != "remote"
                or spec.policy.local_only
            ):
                raise RemoteConflict("Run does not permit remote execution.")
            expected_input = self.engine.artifacts.put_json(sanitized_input(dataset, spec))
            if (
                parsed.input_sha256 != expected_input
                or parsed.implementation_sha256 != spec.workload.implementation_sha256
                or parsed.image != spec.workload.image
                or parsed.measurements.image not in {None, spec.workload.image}
            ):
                raise RemoteConflict("Completion does not match its frozen assignment.")
            self.engine.artifacts.put_json(normalized)
            state, error, predictions = parsed.state, parsed.error, None
            assessment = {"verdict": "not_tested", "reason": error or f"remote_{state}"}
            if state == "completed":
                try:
                    if len(canonical_bytes(parsed.predictions)) > spec.limits.output_bytes:
                        state, error = "resource_exhausted", "remote_output_limit_reached"
                    else:
                        predictions = parse_predictions(parsed.predictions, dataset)
                        assessment = evaluate(dataset, predictions, spec.claim)
                except ValueError:
                    state, error = "invalid", "remote_prediction_validation_failed"
                if state != "completed":
                    assessment = {"verdict": "not_tested", "reason": error}
                    predictions = None
            result = {
                "schema_version": "1",
                "run_id": run_id,
                "spec_hash": run["spec_hash"],
                "execution_state": state,
                "assessment": assessment,
                "measurements": {
                    **parsed.measurements.model_dump(mode="json", exclude_none=True),
                    "worker_id": worker_id,
                    "measurement_source": "remote_worker_reported",
                    "runtime_attestation": "not_verified",
                },
                "provider_calls": [],
                "assumptions": list(spec.assumptions),
                "authoring_receipt_sha256": spec.authoring_receipt_sha256,
                "remote_completion_sha256": completion_hash,
                "remote_worker": owner,
                "remote_token": token,
                "input_sha256": expected_input,
                "implementation_sha256": spec.workload.implementation_sha256,
                "limitations": [
                    "Remote execution, isolation and measurements are worker-reported, not attested.",
                    "The server evaluates predictions against retained reference data.",
                    "Numerical agreement does not establish implementation or protocol fidelity.",
                ],
            }
            if predictions is not None:
                result["predictions"] = predictions.model_dump(mode="json")
            result_hash = self.engine.artifacts.put_json(result)
            if state == "cancelled":
                # A worker cancellation is an audit record, never an unfenced result.
                if not self.engine.runs.cancel_if_active(
                    run_id, token, diagnostic_hash=result_hash
                ):
                    raise RemoteConflict("Run became terminal before cancellation was accepted.")
                return self.engine.runs.get_run(run_id)
            if not self.engine.runs.finish(run_id, token, state, result_hash, error):
                raise RemoteConflict("Lease was lost before completion could publish.")
            if state != "completed":
                self.engine.runs.attach_diagnostic(run_id, token, result_hash)
            return self.engine.runs.get_run(run_id)
