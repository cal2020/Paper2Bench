"""A durable HTTP worker; generated code executes only through DockerRunner.

Completion delivery is at-least-once and idempotent at the coordinator. A lost
completion response never causes this client to execute the candidate again.
Crashing before the durable completion write can cause a later deterministic
attempt to execute again; no exactly-once execution claim is made.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import re
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, StrictFloat, StrictInt, model_validator

from .models import SHA256, FrozenModel, Limits, WorkloadSpec, canonical_bytes, parse_predictions
from .providers import ProviderClient, ProviderProfile
from .remote import WorkerMeasurements
from .runtime import (
    DockerRunner,
    ExecutionCancelled,
    ExecutionFailure,
    ResourceExhausted,
    RuntimeSpec,
    RuntimeUnavailable,
    _validated_input,
)

MAX_BODY_BYTES = 20 * 1024 * 1024
MAX_CODE_BYTES = 1024 * 1024
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONTAINER_NAME = re.compile(r"paper2bench-[0-9a-f]{32}\Z")
_TERMINAL = {"completed", "failed", "invalid", "unsupported", "resource_exhausted", "cancelled"}
_MEASUREMENTS = {
    "runtime",
    "task",
    "image",
    "wall_seconds",
    "stdout_bytes",
    "stderr_bytes",
    "memory_limit_mb",
    "memory_limit_enforced",
    "network_used",
    "isolation_scope",
    "cleanup_confirmed",
    "candidate_uid",
    "supervisor_deadline_enforced",
    "container_name",
    "worker_engine_identity_sha256",
    "worker_python_version",
    "worker_platform",
}
_ERRORS = {
    "runtime_unavailable",
    "resource_limit_reached",
    "lease_lost_or_cancelled",
    "candidate_execution_failed",
    "worker_internal_error",
}


class WorkerClientError(RuntimeError):
    """A safe diagnostic without credentials, response bodies, or candidate errors."""


class _RemoteHTTPError(WorkerClientError):
    def __init__(self, code: str, status: int | None = None):
        super().__init__(code)
        self.code = code
        self.status = status


class _Assignment(FrozenModel):
    schema_version: Literal["1"]
    worker_id: str
    run_id: str
    token: Annotated[StrictInt, Field(gt=0)]
    attempt: Annotated[StrictInt, Field(gt=0)]
    lease_seconds: Annotated[StrictFloat, Field(gt=0, le=3600)]
    spec_hash: SHA256
    implementation_sha256: SHA256
    input_sha256: SHA256
    image: str
    limits: Limits
    seed: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    task: Literal["classification", "knapsack"]
    code: str
    input: dict

    @model_validator(mode="after")
    def safe_identity(self):
        if not _RUN_ID.fullmatch(self.run_id) or not _WORKER_ID.fullmatch(self.worker_id):
            raise ValueError("invalid remote identity")
        return self


class _Completion(FrozenModel):
    state: Literal["completed", "failed", "unsupported", "resource_exhausted", "cancelled"]
    input_sha256: SHA256
    implementation_sha256: SHA256
    image: str
    predictions: dict | None = None
    measurements: dict
    error: str | None = None

    @model_validator(mode="after")
    def bounded_diagnostics(self):
        WorkloadSpec(
            kind="python", image=self.image, implementation_sha256=self.implementation_sha256
        )
        if self.error is not None and self.error not in _ERRORS:
            raise ValueError("unsupported worker diagnostic")
        if self.state == "completed" and (self.predictions is None or self.error is not None):
            raise ValueError("completed submission requires predictions without an error")
        if self.state != "completed" and self.predictions is not None:
            raise ValueError("failed submission cannot publish predictions")
        if set(self.measurements) - _MEASUREMENTS:
            raise ValueError("unexpected worker measurements")
        canonical_bytes(self.measurements)
        return self


class _Pending(FrozenModel):
    schema_version: Literal["1"] = "1"
    coordinator_sha256: SHA256
    worker_id: str
    run_id: str
    token: Annotated[StrictInt, Field(gt=0)]
    spec_hash: SHA256
    completion: _Completion
    completion_sha256: SHA256

    @model_validator(mode="after")
    def valid_identity(self):
        if not _RUN_ID.fullmatch(self.run_id) or not _WORKER_ID.fullmatch(self.worker_id):
            raise ValueError("invalid pending identity")
        if hashlib.sha256(canonical_bytes(self.completion)).hexdigest() != self.completion_sha256:
            raise ValueError("pending completion integrity mismatch")
        return self


def _strict_json(data: bytes) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate response key")
            value[key] = item
        return value

    def reject_constant(_):
        raise ValueError("nonfinite response number")

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject_constant)


class WorkerClient:
    """Consume one remote assignment at a time using a private durable spool."""

    def __init__(
        self,
        base_url: str,
        worker_id: str,
        token_env: str,
        spool: Path,
        images: tuple[str, ...],
        max_memory_mb: int = 256,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        try:
            hostname = urlsplit(base_url).hostname or ""
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
            self._profile = ProviderProfile(
                name="remote-worker",
                base_url=base_url,
                location="local" if loopback else "cloud",
                api_key_env=token_env,
                timeout=10,
            )
            if not loopback and urlsplit(self._profile.base_url).scheme != "https":
                raise ValueError("remote coordinators require HTTPS")
            if not _WORKER_ID.fullmatch(worker_id):
                raise ValueError("invalid worker ID")
            if type(max_memory_mb) is not int or not 16 <= max_memory_mb <= 65536:
                raise ValueError("invalid memory capability")
            if not isinstance(images, tuple) or not images or len(images) > 100:
                raise ValueError("declare at least one staged immutable image")
            for image in images:
                WorkloadSpec(kind="python", image=image, implementation_sha256="0" * 64)
            if transport is not None and not isinstance(transport, httpx.AsyncBaseTransport):
                raise ValueError("injected transport must support async requests")
        except Exception:
            raise WorkerClientError("invalid_worker_configuration") from None
        self.base_url = self._profile.base_url
        self.worker_id = worker_id
        self.token_env = token_env
        self.spool = Path(spool)
        self.images = tuple(sorted(set(images)))
        self.max_memory_mb = max_memory_mb
        self._transport = transport
        self._coordinator_hash = hashlib.sha256(self.base_url.encode()).hexdigest()

    @contextmanager
    def _locked_spool(self):
        if self.spool.is_symlink():
            raise WorkerClientError("unsafe_spool_directory")
        self.spool.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.spool, 0o700)
        descriptor = os.open(
            self.spool / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkerClientError("worker_spool_busy") from None
            yield
        finally:
            os.close(descriptor)

    def _sync_directory(self):
        descriptor = os.open(self.spool, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_pending(self, pending: _Pending):
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.spool)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_bytes(pending))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.spool / "pending.json")
            self._sync_directory()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _read_pending(self) -> _Pending | None:
        path = self.spool / "pending.json"
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise WorkerClientError("unsafe_pending_submission")
            raw = handle.read(MAX_BODY_BYTES + 1)
        if len(raw) > MAX_BODY_BYTES:
            raise WorkerClientError("pending_submission_too_large")
        try:
            pending = _Pending.model_validate(_strict_json(raw))
        except Exception:
            raise WorkerClientError("invalid_pending_submission") from None
        if (
            pending.worker_id != self.worker_id
            or pending.coordinator_sha256 != self._coordinator_hash
        ):
            raise WorkerClientError("pending_submission_owner_mismatch")
        return pending

    async def _arequest(self, endpoint: str, payload: dict, timeout: float) -> dict:
        # Re-read credentials for each request to support rotation without storing values.
        ProviderClient(self._profile).validate(local_only=self._profile.location == "local")
        credential = os.environ.get(self.token_env, "")
        raw_request = canonical_bytes(payload)
        if len(raw_request) > MAX_BODY_BYTES:
            raise _RemoteHTTPError("remote_request_too_large")
        try:
            async with (
                asyncio.timeout(timeout),
                httpx.AsyncClient(
                    transport=self._transport,
                    timeout=timeout,
                    follow_redirects=False,
                    trust_env=self._profile.location != "local",
                ) as client,
                client.stream(
                    "POST",
                    self.base_url + endpoint,
                    content=raw_request,
                    headers={
                        "Authorization": "Bearer " + credential,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                    },
                ) as response,
            ):
                if response.status_code != 200:
                    raise _RemoteHTTPError("remote_request_rejected", response.status_code)
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise _RemoteHTTPError("remote_encoding_unsupported")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        raise _RemoteHTTPError("remote_response_too_large")
                    body.extend(chunk)
        except (TimeoutError, httpx.TransportError):
            raise _RemoteHTTPError("remote_outcome_unknown") from None
        try:
            parsed = _strict_json(bytes(body))
            if not isinstance(parsed, dict):
                raise ValueError("response is not an object")
            return parsed
        except Exception:
            raise _RemoteHTTPError("remote_response_invalid") from None

    def _request(self, endpoint: str, payload: dict, *, timeout: float = 5) -> dict:
        if endpoint not in {"/worker/claim", "/worker/heartbeat", "/worker/complete"}:
            raise WorkerClientError("unsupported_worker_endpoint")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise WorkerClientError("worker_requires_synchronous_entrypoint")
        return asyncio.run(self._arequest(endpoint, payload, timeout))

    def _validate_assignment(self, value: dict) -> tuple[_Assignment, RuntimeSpec]:
        try:
            assignment = _Assignment.model_validate(value)
            if assignment.worker_id != self.worker_id:
                raise ValueError("wrong worker")
            if (
                assignment.image not in self.images
                or assignment.limits.memory_mb > self.max_memory_mb
            ):
                raise ValueError("unsupported advertised capability")
            code = assignment.code.encode("utf-8")
            if (
                len(code) > MAX_CODE_BYTES
                or hashlib.sha256(code).hexdigest() != assignment.implementation_sha256
            ):
                raise ValueError("code integrity mismatch")
            if (
                hashlib.sha256(canonical_bytes(assignment.input)).hexdigest()
                != assignment.input_sha256
            ):
                raise ValueError("input integrity mismatch")
            spec = RuntimeSpec(
                task=assignment.task,
                seed=assignment.seed,
                limits=assignment.limits,
                workload=WorkloadSpec(
                    kind="python",
                    image=assignment.image,
                    implementation_sha256=assignment.implementation_sha256,
                ),
            )
            normalized, _ = _validated_input(assignment.input, spec)
            if canonical_bytes(normalized) != canonical_bytes(assignment.input):
                raise ValueError("noncanonical execution input")
            return assignment, spec
        except Exception:
            raise WorkerClientError("invalid_remote_assignment") from None

    @staticmethod
    def _measurements(values: dict, *, strict: bool = False) -> dict:
        if not isinstance(values, dict):
            raise ExecutionFailure("runtime measurements must be an object")
        result = {}
        for key in _MEASUREMENTS:
            if key not in values:
                continue
            value = values[key]
            if key in {"wall_seconds"}:
                valid = type(value) in {int, float} and math.isfinite(value) and value >= 0
            elif key in {"stdout_bytes", "stderr_bytes", "memory_limit_mb"}:
                valid = type(value) is int and value >= 0
            elif key == "candidate_uid":
                valid = type(value) is int and value == 65534
            elif key == "supervisor_deadline_enforced":
                valid = type(value) is bool
            elif key in {"memory_limit_enforced", "network_used", "cleanup_confirmed"}:
                valid = type(value) is bool
            elif key == "container_name":
                valid = isinstance(value, str) and bool(_CONTAINER_NAME.fullmatch(value))
            elif key == "runtime":
                valid = value == "docker"
            elif key == "task":
                valid = value in {"classification", "knapsack"}
            elif key == "isolation_scope":
                valid = value == "single_user_container"
            elif key == "worker_engine_identity_sha256":
                valid = isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))
            elif key == "image":
                try:
                    WorkloadSpec(kind="python", image=value, implementation_sha256="0" * 64)
                    valid = True
                except ValueError:
                    valid = False
            else:
                valid = isinstance(value, str) and bool(
                    re.fullmatch(r"[A-Za-z0-9_.+-]{1,100}", value)
                )
            if valid:
                result[key] = value
            elif strict:
                raise ExecutionFailure("runtime supplied an invalid measurement")
        return result

    def _execute(self, assignment: _Assignment, spec: RuntimeSpec) -> dict:
        stop, lost = threading.Event(), threading.Event()
        interval = max(0.01, min(assignment.lease_seconds / 3, 5))

        def maintain_lease():
            while not stop.wait(interval):
                try:
                    response = self._request(
                        "/worker/heartbeat",
                        {
                            "worker_id": self.worker_id,
                            "run_id": assignment.run_id,
                            "token": assignment.token,
                        },
                        timeout=min(5, interval),
                    )
                    if (
                        set(response) != {"active"}
                        or type(response["active"]) is not bool
                        or not response["active"]
                    ):
                        lost.set()
                        return
                except Exception:
                    lost.set()
                    return

        heartbeat = threading.Thread(target=maintain_lease, name="paper2bench-remote-heartbeat")
        heartbeat.start()
        measurements, predictions, error, state = {}, None, None, "completed"
        try:
            try:
                result = DockerRunner().run_input(
                    assignment.code.encode(), assignment.input, spec, cancelled=lost.is_set
                )
                if lost.is_set():
                    raise ExecutionCancelled("lease lost")
                predictions = parse_predictions(result.predictions, assignment.task).model_dump(
                    mode="json"
                )
                # A structurally valid string may still contain an invalid UTF-8
                # surrogate. Detect serialization errors before building/spooling
                # a successful submission, while failure recovery is still active.
                if len(canonical_bytes(predictions)) > assignment.limits.output_bytes:
                    raise ResourceExhausted("runtime predictions exceed the output limit")
                measurements = self._measurements(result.measurements, strict=True)
                measurements = WorkerMeasurements.model_validate(measurements).model_dump(
                    mode="json", exclude_none=True
                )
                expected_config = {
                    "runtime": "docker",
                    "task": assignment.task,
                    "image": assignment.image,
                    "memory_limit_mb": assignment.limits.memory_mb,
                    "memory_limit_enforced": True,
                    "network_used": False,
                    "isolation_scope": "single_user_container",
                    "cleanup_confirmed": True,
                }
                if any(
                    key in measurements and measurements[key] != expected
                    for key, expected in expected_config.items()
                ):
                    raise ExecutionFailure("runtime configuration differs from assignment")
                canonical_bytes(measurements)
            except Exception as exc:
                # Never retain output assigned before a later validation error.
                predictions = None
                if isinstance(exc, ExecutionCancelled):
                    state, error = "cancelled", "lease_lost_or_cancelled"
                elif isinstance(exc, RuntimeUnavailable):
                    state, error = "unsupported", "runtime_unavailable"
                elif isinstance(exc, ResourceExhausted):
                    state, error = "resource_exhausted", "resource_limit_reached"
                elif isinstance(exc, ExecutionFailure):
                    state, error = "failed", "candidate_execution_failed"
                else:
                    state, error = "failed", "worker_internal_error"
                try:
                    measurements = self._measurements(
                        {
                            key: getattr(exc, key)
                            for key in ("cleanup_confirmed", "container_name")
                            if hasattr(exc, key)
                        }
                    )
                except Exception:
                    measurements = {}
            completion = _Completion(
                state=state,
                input_sha256=assignment.input_sha256,
                implementation_sha256=assignment.implementation_sha256,
                image=assignment.image,
                predictions=predictions,
                measurements=measurements,
                error=error,
            )
            pending = _Pending(
                coordinator_sha256=self._coordinator_hash,
                worker_id=self.worker_id,
                run_id=assignment.run_id,
                token=assignment.token,
                spec_hash=assignment.spec_hash,
                completion=completion,
                completion_sha256=hashlib.sha256(canonical_bytes(completion)).hexdigest(),
            )
            # The durable write must complete before any submission is dispatched.
            self._write_pending(pending)
            return self._submit_pending(pending)
        finally:
            stop.set()
            heartbeat.join()

    def _submit_pending(self, pending: _Pending) -> dict:
        try:
            response = self._request(
                "/worker/complete",
                {
                    "worker_id": self.worker_id,
                    "run_id": pending.run_id,
                    "token": pending.token,
                    "completion": pending.completion.model_dump(mode="json"),
                },
            )
        except _RemoteHTTPError as exc:
            if exc.status in {409, 410}:
                archive = self.spool / f"fenced-{pending.run_id}-{pending.token}.json"
                os.replace(self.spool / "pending.json", archive)
                self._sync_directory()
                return {"state": "fenced", "run_id": pending.run_id, "error": "submission_fenced"}
            return {"state": "pending_submission", "run_id": pending.run_id, "error": exc.code}
        except Exception:
            return {
                "state": "pending_submission",
                "run_id": pending.run_id,
                "error": "submission_unacknowledged",
            }
        expected_token = pending.token + (1 if response.get("state") == "cancelled" else 0)
        if (
            response.get("run_id") != pending.run_id
            or response.get("id", pending.run_id) != pending.run_id
            or response.get("worker") != "remote:" + self.worker_id
            or response.get("spec_hash") != pending.spec_hash
            or type(response.get("token")) is not int
            or response["token"] != expected_token
            or response.get("state") not in _TERMINAL
        ):
            return {
                "state": "pending_submission",
                "run_id": pending.run_id,
                "error": "submission_acknowledgement_invalid",
            }
        (self.spool / "pending.json").unlink()
        self._sync_directory()
        return {"state": "completed", "run_id": pending.run_id, "run_state": response["state"]}

    def run_once(self) -> dict:
        """Replay a pending completion first, otherwise execute at most one assignment."""
        try:
            ProviderClient(self._profile).validate(local_only=self._profile.location == "local")
            with self._locked_spool():
                pending = self._read_pending()
                if pending is not None:
                    return self._submit_pending(pending)
                response = self._request(
                    "/worker/claim",
                    {
                        "worker_id": self.worker_id,
                        "images": list(self.images),
                        "max_memory_mb": self.max_memory_mb,
                    },
                )
                if set(response) != {"assignment"}:
                    raise WorkerClientError("invalid_claim_response")
                if response["assignment"] is None:
                    return {"state": "idle"}
                assignment, spec = self._validate_assignment(response["assignment"])
                return self._execute(assignment, spec)
        except WorkerClientError as exc:
            return {"state": "error", "error": str(exc)}
        except Exception:
            return {"state": "error", "error": "worker_operation_failed"}
