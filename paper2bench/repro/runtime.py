"""Trusted reference execution and bounded, disposable Docker workloads.

Docker is an isolation layer for local, single-user experiments, not a claim of
hostile multi-tenant security. Arbitrary candidate code never executes in this
Python process. The only inputs passed to it are training records, test features,
and the declared seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, StrictFloat, StrictInt, ValidationError, model_validator

from .models import (
    Dataset,
    Example,
    ExperimentSpec,
    FrozenModel,
    KnapsackDataset,
    KnapsackInstance,
    Limits,
    Nonblank,
    OptimizationPredictions,
    Predictions,
    WorkloadSpec,
    canonical_bytes,
    parse_predictions,
)


class RuntimeUnavailable(RuntimeError):
    """Required runtime or staged immutable image is unavailable."""


class ExecutionFailure(RuntimeError):
    """Candidate execution or its output violated the workload contract."""


class ResourceExhausted(ExecutionFailure):
    """Declared wall-time, memory, input, or output bounds were exceeded."""


class ExecutionCancelled(ExecutionFailure):
    """Execution was cancelled before publication."""


class RuntimeSpec(FrozenModel):
    """Minimal validated worker contract; no claim, evidence, labels, or secrets."""

    workload: WorkloadSpec
    limits: Limits
    seed: StrictInt = Field(ge=0, le=2**63 - 1)
    task: Literal["classification", "knapsack"]


@dataclass(frozen=True)
class ExecutionResult:
    predictions: Predictions | OptimizationPredictions
    measurements: dict


_IMAGE = re.compile(r"(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_CODE_BYTES = 1024 * 1024

# This trusted PID1 supervisor executes only INSIDE the disposable container.
# Arm the deadline before reading input, including when the host worker vanishes
# halfway through a transfer. Candidate code executes as a different UID with
# no capabilities; it cannot signal the root supervisor. PID1 exit tears down
# the namespace, including descendants that escaped the original process group.
_BOOTSTRAP = """import json,os,signal,subprocess,sys,tempfile,time
seconds=float(sys.argv[1])
child=None
def expired(*_):
    if child is not None:
        try:
            os.killpg(child.pid,signal.SIGKILL)
        except OSError:
            pass
    os._exit(124)
signal.signal(signal.SIGALRM,expired)
signal.setitimer(signal.ITIMER_REAL,seconds)
started=time.monotonic()
envelope=json.load(sys.stdin)
if envelope.get('timeout_seconds',seconds)!=seconds:
    sys.exit(2)
fd,path=tempfile.mkstemp(prefix='paper2bench-candidate-',suffix='.py',dir='/tmp')
try:
    with os.fdopen(fd,'wb') as stream:
        stream.write(envelope['code'].encode('utf-8'))
    os.chmod(path,0o444)
    child=subprocess.Popen(
        [sys.executable,'-I','-u',path],stdin=subprocess.PIPE,
        user=65534,group=65534,extra_groups=[],start_new_session=True,
        env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/tmp',
             'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED','0')})
    try:
        child.communicate(
            json.dumps(envelope['input'],allow_nan=False).encode('utf-8'),
            timeout=max(.001,seconds-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
        expired()
    status=child.returncode
finally:
    os.unlink(path)
# Reserve 124 for a trusted supervisor deadline; a candidate cannot assert it.
sys.exit(125 if status==124 else status if status>=0 else 128-status)
"""


def _check(cancelled: Callable[[], bool], deadline: float) -> None:
    if cancelled():
        raise ExecutionCancelled("execution cancelled")
    if time.monotonic() >= deadline:
        raise ResourceExhausted("declared wall-time limit exceeded")


class _FeatureExample(FrozenModel):
    id: Nonblank
    x: tuple[StrictFloat, ...] = Field(min_length=1)


class _ClassificationInput(FrozenModel):
    train: tuple[Example, ...] = Field(min_length=1)
    test: tuple[_FeatureExample, ...] = Field(min_length=1)
    seed: StrictInt = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def consistent_features(self):
        examples = self.train + self.test
        ids = [example.id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("input IDs must be unique and disjoint")
        dimension = len(self.train[0].x)
        if any(len(example.x) != dimension for example in examples):
            raise ValueError("input feature dimensions must match")
        return self


class _KnapsackInput(FrozenModel):
    instances: tuple[KnapsackInstance, ...] = Field(min_length=1)
    seed: StrictInt = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def bounded_instances(self):
        KnapsackDataset(instances=self.instances)
        return self


def _validated_input(input_payload: dict, spec: ExperimentSpec | RuntimeSpec) -> tuple[dict, str]:
    """Validate a public execution payload without constructing hidden labels."""
    if not isinstance(input_payload, dict):
        raise ExecutionFailure("candidate input must be a JSON object")
    if not isinstance(spec, (ExperimentSpec, RuntimeSpec)):
        raise ExecutionFailure("execution requires a validated experiment or runtime specification")
    task = "knapsack" if "instances" in input_payload else "classification"
    expected = (
        spec.task
        if isinstance(spec, RuntimeSpec)
        else "classification"
        if spec.claim.metric == "accuracy"
        else "knapsack"
    )
    try:
        if task != expected:
            raise ValueError("public input task conflicts with execution specification")
        if task == "knapsack":
            validated = _KnapsackInput.model_validate(input_payload)
        else:
            validated = _ClassificationInput.model_validate(input_payload)
        if validated.seed != spec.seed:
            raise ValueError("input seed does not match frozen experiment seed")
        public_input = validated.model_dump(mode="json")
        if len(json.dumps(public_input, allow_nan=False).encode()) > _MAX_INPUT_BYTES:
            raise ResourceExhausted("candidate input exceeds 16 MiB limit")
        return public_input, task
    except (ValueError, ValidationError, TypeError, RecursionError) as exc:
        raise ExecutionFailure("candidate input violates the typed public-input contract") from exc


def sanitized_input(dataset: Dataset | KnapsackDataset, spec: ExperimentSpec) -> dict:
    """Export the exact public input protocol for local or remote execution.

    Classification exposes training labels and test features only. Knapsack
    exposes problem instances; objective references belong to trusted scoring.
    """
    if isinstance(dataset, KnapsackDataset):
        payload = {
            "instances": [instance.model_dump(mode="json") for instance in dataset.instances],
            "seed": spec.seed,
        }
    elif isinstance(dataset, Dataset):
        payload = {
            "train": [item.model_dump(mode="json") for item in dataset.train],
            "test": [{"id": item.id, "x": list(item.x)} for item in dataset.test],
            "seed": spec.seed,
        }
    else:
        raise ExecutionFailure("unsupported dataset type")
    return _validated_input(payload, spec)[0]


def _decode_predictions(
    raw: bytes, task: str = "classification"
) -> Predictions | OptimizationPredictions:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("non-finite JSON number")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant
        )
        predictions = parse_predictions(value, task)
        # Escaped lone surrogates pass JSON/Pydantic string validation but
        # cannot become canonical UTF-8 artifacts or a durable worker receipt.
        canonical_bytes(predictions)
        return predictions
    except (UnicodeError, ValueError, ValidationError, RecursionError) as exc:
        raise ExecutionFailure(
            "candidate output must be one valid Predictions JSON object"
        ) from exc


class BuiltinRunner:
    """Deterministic nearest-centroid reference; fit uses training data only."""

    def run(
        self,
        dataset: Dataset,
        spec: ExperimentSpec,
        *,
        cancelled: Callable[[], bool],
    ) -> ExecutionResult:
        if spec.workload.kind != "builtin_centroid":
            raise ExecutionFailure("builtin runner requires builtin_centroid workload")
        started = time.monotonic()
        deadline = started + spec.limits.timeout_seconds
        _check(cancelled, deadline)
        labels = sorted({item.y for item in dataset.train})
        groups = {label: [] for label in labels}
        for item in dataset.train:
            _check(cancelled, deadline)
            groups[item.y].append(item.x)
        centroids = {
            # Divide before summing so a finite mean does not overflow merely
            # because several large, finite coordinates share a class.
            label: tuple(
                math.fsum(value / len(rows) for value in column)
                for column in zip(*rows, strict=True)
            )
            for label, rows in groups.items()
        }
        predicted = []
        for item in dataset.test:
            _check(cancelled, deadline)
            # hypot avoids unnecessary overflow when finite coordinates are large.
            distances = {
                label: math.hypot(*(a - b for a, b in zip(item.x, centroid, strict=True)))
                for label, centroid in centroids.items()
            }
            if not all(math.isfinite(distance) for distance in distances.values()):
                raise ExecutionFailure("feature magnitude exceeds reference numeric range")
            predicted.append(min(labels, key=lambda label: (distances[label], label)))
        predictions = Predictions(
            ids=tuple(item.id for item in dataset.test), labels=tuple(predicted)
        )
        if len(predictions.model_dump_json().encode()) > spec.limits.output_bytes:
            raise ResourceExhausted("declared output limit exceeded")
        return ExecutionResult(
            predictions,
            {
                "runtime": "trusted_builtin",
                "algorithm": "nearest_centroid",
                "wall_seconds": time.monotonic() - started,
                "training_examples": len(dataset.train),
                "test_examples": len(dataset.test),
                "memory_limit_enforced": False,
                "network_used": False,
            },
        )


class BuiltinKnapsackRunner:
    """Trusted exact 0/1 knapsack with bounded, train-free dynamic programming."""

    def run(
        self,
        dataset: KnapsackDataset,
        spec: ExperimentSpec,
        *,
        cancelled: Callable[[], bool],
    ) -> ExecutionResult:
        if spec.workload.kind != "builtin_knapsack" or not isinstance(dataset, KnapsackDataset):
            raise ExecutionFailure("knapsack runner requires builtin_knapsack and knapsack data")
        if spec.claim.metric == "accuracy":
            raise ExecutionFailure("knapsack runner requires an optimization claim")
        started = time.monotonic()
        deadline = started + spec.limits.timeout_seconds
        _check(cancelled, deadline)
        selections = []
        transitions = 0
        peak_estimate = 0
        for instance in dataset.instances:
            _check(cancelled, deadline)
            capacity = instance.capacity
            # Conservative allocation estimate for two Python lists containing
            # integer objective values and selected-item bit masks. This is a
            # preflight guard, not a claim of OS-enforced process memory limits.
            value_bytes = 28 + (sum(instance.values).bit_length() + 7) // 8
            mask_bytes = 28 + (len(instance.weights) + 7) // 8
            estimated = (capacity + 1) * (16 + value_bytes + mask_bytes)
            peak_estimate = max(peak_estimate, estimated)
            if estimated > spec.limits.memory_mb * 1024 * 1024 // 2:
                raise ResourceExhausted("knapsack working-memory estimate exceeds declared budget")
            objectives = [0] * (capacity + 1)
            masks = [0] * (capacity + 1)
            for index, (weight, value) in enumerate(
                zip(instance.weights, instance.values, strict=True)
            ):
                _check(cancelled, deadline)
                if weight > capacity or value == 0:
                    continue
                for allowed_weight in range(capacity, weight - 1, -1):
                    transitions += 1
                    if transitions > 20_000_000:
                        raise ResourceExhausted("knapsack transition limit exceeded")
                    if transitions % 128 == 0:
                        _check(cancelled, deadline)
                    candidate = objectives[allowed_weight - weight] + value
                    if candidate > objectives[allowed_weight]:
                        objectives[allowed_weight] = candidate
                        masks[allowed_weight] = masks[allowed_weight - weight] | (1 << index)
            mask = masks[capacity]
            selections.append(
                tuple(index for index in range(len(instance.weights)) if mask >> index & 1)
            )
        _check(cancelled, deadline)
        predictions = OptimizationPredictions(
            ids=tuple(instance.id for instance in dataset.instances),
            selections=tuple(selections),
        )
        if len(predictions.model_dump_json().encode()) > spec.limits.output_bytes:
            raise ResourceExhausted("declared output limit exceeded")
        return ExecutionResult(
            predictions,
            {
                "runtime": "trusted_builtin",
                "task": "knapsack",
                "algorithm": "knapsack_01_exact_dp",
                "wall_seconds": time.monotonic() - started,
                "instances": len(dataset.instances),
                "dp_transitions": transitions,
                "working_memory_estimate_bytes": peak_estimate,
                "memory_limit_enforced": False,
                "network_used": False,
            },
        )


class DockerRunner:
    """Run Python candidates with an immutable, pre-staged image and no egress.

    The host Docker endpoint is deliberately fixed. No image pull or unsafe host
    execution fallback occurs. Docker's configured registry credentials remain
    usable for explicit image staging outside this runner.
    """

    def __init__(self, docker: str = "docker", *, poll_interval: float = 0.05):
        self.command = [docker, "--host=unix:///var/run/docker.sock"]
        self.poll_interval = poll_interval

    @staticmethod
    def _environment() -> dict[str, str]:
        result = os.environ.copy()
        for key in (
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        ):
            result.pop(key, None)
        return result

    def _control(self, args: list[str], *, check: bool = True, timeout: float = 10) -> bytes:
        try:
            result = subprocess.run(
                self.command + args,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env=self._environment(),
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeUnavailable("managed local Docker daemon is unavailable") from exc
        if check and result.returncode:
            # Daemon errors can contain registry URLs or config details. Do not
            # propagate them into ordinary run records.
            raise RuntimeUnavailable(
                "Docker control operation failed; check daemon and staged image"
            )
        return result.stdout

    def resolve_image(self, reference: str) -> str:
        """Resolve a locally staged reference to its immutable image ID; never pull."""
        if (
            not reference
            or reference.startswith("-")
            or len(reference) > 512
            or any(character.isspace() for character in reference)
        ):
            raise RuntimeUnavailable("invalid image reference")
        image = self._control(["image", "inspect", "--format", "{{.Id}}", reference])
        try:
            resolved = image.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeUnavailable("Docker returned an invalid image identity") from exc
        if not _IMAGE_ID.fullmatch(resolved):
            raise RuntimeUnavailable("Docker returned an invalid image identity")
        return resolved

    def _create_args(self, name: str, image: str, spec: ExperimentSpec | RuntimeSpec) -> list[str]:
        memory = f"{spec.limits.memory_mb}m"
        args = [
            "create",
            "--name",
            name,
            "--interactive",
            "--attach",
            "stdin",
            "--attach",
            "stdout",
            "--attach",
            "stderr",
            "--pull",
            "never",
            "--label",
            "paper2bench.managed=true",
            "--network",
            "none",
            "--read-only",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--cap-add",
            "KILL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            "64",
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--cpus",
            "1",
            "--ulimit",
            "nofile=64:64",
            "--ipc",
            "none",
            "--log-driver",
            "none",
            "--stop-timeout",
            "1",
            "--workdir",
            "/tmp",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777",
            "--env",
            "HOME=/tmp",
            "--env",
            f"PYTHONHASHSEED={spec.seed % (2**32)}",
        ]
        # Docker's client proxy configuration otherwise injects proxy settings
        # even when none were explicitly supplied to docker create.
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "FTP_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "ftp_proxy",
        ):
            args.extend(["--env", f"{key}="])
        args.extend(
            [
                "--entrypoint",
                "python",
                image,
                "-I",
                "-u",
                "-c",
                _BOOTSTRAP,
                str(spec.limits.timeout_seconds),
            ]
        )
        return args

    def _cleanup(self, name: str, *, uncertain_create: bool) -> bool:
        deadline = time.monotonic() + 10
        while True:
            try:
                self._control(["rm", "--force", name], timeout=2)
                return True
            except RuntimeUnavailable:
                # A timed-out create request may still be finishing in the
                # daemon. An immediate "not found" is not cleanup evidence.
                if not uncertain_create or time.monotonic() >= deadline:
                    return False
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def _attached(
        self,
        name: str,
        payload: bytes,
        *,
        deadline: float,
        output_limit: int,
        cancelled: Callable[[], bool],
    ) -> tuple[bytes, bytes, int]:
        try:
            process = subprocess.Popen(
                self.command + ["start", "--attach", "--interactive", name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._environment(),
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeUnavailable("cannot attach to managed Docker container") from exc
        output = bytearray()
        errors = bytearray()
        offset = 0
        selector = selectors.DefaultSelector()
        try:
            assert process.stdin and process.stdout and process.stderr
            for pipe, event, kind in (
                (process.stdin, selectors.EVENT_WRITE, "stdin"),
                (process.stdout, selectors.EVENT_READ, "stdout"),
                (process.stderr, selectors.EVENT_READ, "stderr"),
            ):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, event, kind)
            while selector.get_map():
                _check(cancelled, deadline)
                for key, _events in selector.select(self.poll_interval):
                    if key.data == "stdin":
                        try:
                            written = os.write(
                                key.fileobj.fileno(), payload[offset : offset + 65536]
                            )
                            offset += written
                        except BrokenPipeError:
                            offset = len(payload)
                        except BlockingIOError:
                            continue
                        if offset >= len(payload):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    try:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    else:
                        target = output if key.data == "stdout" else errors
                        target.extend(chunk)
                        if len(output) + len(errors) > output_limit:
                            raise ResourceExhausted("declared combined output limit exceeded")
            _check(cancelled, deadline)
            try:
                returncode = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise ResourceExhausted("declared wall-time limit exceeded") from exc
            return bytes(output), bytes(errors), returncode
        finally:
            selector.close()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def run(
        self,
        code: bytes,
        dataset: Dataset | KnapsackDataset,
        spec: ExperimentSpec,
        *,
        cancelled: Callable[[], bool],
    ) -> ExecutionResult:
        return self.run_input(code, sanitized_input(dataset, spec), spec, cancelled=cancelled)

    def run_input(
        self,
        code: bytes,
        input_payload: dict,
        spec: ExperimentSpec | RuntimeSpec,
        *,
        cancelled: Callable[[], bool],
    ) -> ExecutionResult:
        public_input, task = _validated_input(input_payload, spec)
        if spec.workload.kind != "python":
            raise ExecutionFailure("Docker runner requires python workload")
        image = spec.workload.image or ""
        if not _IMAGE.fullmatch(image):
            raise RuntimeUnavailable("Python workloads require a pinned image ID or digest")
        if len(code) > _MAX_CODE_BYTES:
            raise ResourceExhausted("candidate source exceeds 1 MiB limit")
        if hashlib.sha256(code).hexdigest() != spec.workload.implementation_sha256:
            raise ExecutionFailure("candidate source does not match frozen implementation identity")
        try:
            source = code.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExecutionFailure("candidate source must be UTF-8 Python") from exc
        payload = json.dumps(
            {
                "code": source,
                "input": public_input,
                "timeout_seconds": spec.limits.timeout_seconds,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        if len(payload) > _MAX_INPUT_BYTES:
            raise ResourceExhausted("candidate input exceeds 16 MiB limit")
        started = time.monotonic()
        deadline = started + spec.limits.timeout_seconds
        _check(cancelled, deadline)
        name = f"paper2bench-{uuid.uuid4().hex}"
        measurements = {"cleanup_confirmed": False}
        uncertain_create = False
        primary_error = None
        try:
            # Cleanup by known generated name also covers create timeout after
            # acceptance, when no container ID reached the client.
            try:
                created = self._control(
                    self._create_args(name, image, spec),
                    timeout=min(10, spec.limits.timeout_seconds),
                )
            except RuntimeUnavailable as exc:
                uncertain_create = isinstance(exc.__cause__, subprocess.TimeoutExpired)
                # Setup consumes the declared wall budget too. A control-call
                # timeout at that deadline is exhaustion, not proof that Docker
                # is absent. The generated name remains available for cleanup.
                _check(cancelled, deadline)
                raise
            container_id = created.decode("ascii").strip()
            if not _CONTAINER_ID.fullmatch(container_id):
                raise RuntimeUnavailable("Docker returned an invalid container identity")
            _check(cancelled, deadline)
            output, stderr, client_exit = self._attached(
                name,
                payload,
                deadline=deadline,
                output_limit=spec.limits.output_bytes,
                cancelled=cancelled,
            )
            try:
                state = json.loads(self._control(["inspect", "--format", "{{json .State}}", name]))
            except (ValueError, TypeError) as exc:
                raise RuntimeUnavailable("cannot verify candidate container exit state") from exc
            if state.get("OOMKilled"):
                raise ResourceExhausted("candidate exceeded container memory limit")
            if state.get("ExitCode") == 124:
                raise ResourceExhausted("container-side wall-time limit exceeded")
            if client_exit or state.get("ExitCode") != 0 or state.get("Running"):
                raise ExecutionFailure("candidate exited unsuccessfully")
            _check(cancelled, deadline)
            predictions = _decode_predictions(output, task)
            measurements.update(
                {
                    "runtime": "docker",
                    "task": task,
                    "image": image,
                    "wall_seconds": time.monotonic() - started,
                    "stdout_bytes": len(output),
                    "stderr_bytes": len(stderr),
                    "memory_limit_mb": spec.limits.memory_mb,
                    "memory_limit_enforced": True,
                    "network_used": False,
                    "isolation_scope": "single_user_container",
                    "candidate_uid": 65534,
                    "supervisor_deadline_enforced": True,
                }
            )
            return ExecutionResult(predictions, measurements)
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            confirmed = self._cleanup(name, uncertain_create=uncertain_create)
            measurements["cleanup_confirmed"] = confirmed
            if primary_error is not None:
                # Callers can persist reconciliation information without losing
                # the primary timeout/cancellation/failure classification.
                primary_error.cleanup_confirmed = confirmed
                primary_error.container_name = name
