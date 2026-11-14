"""Bounded, network-disabled Docker execution without host filesystem mounts.

The controller imports/exports regular files over bounded streams. Candidate code
runs in a disposable tmpfs; no candidate process receives a host directory, host
environment, Docker socket, or provider credential. A Docker Desktop Linux VM is
appropriate for the initial single-user profile, not a hostile multitenant claim.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional


class ExecutionUnavailable(RuntimeError):
    """The configured isolation runtime or staged image is unavailable."""


class ExecutionPolicyError(ValueError):
    """Inputs cannot be safely executed within the requested limits."""


class LifecyclePersistenceError(RuntimeError):
    """A trusted controller could not persist an execution lifecycle event."""


@dataclass(frozen=True)
class ExecutionRequest:
    code: str
    workspace: Path
    timeout_seconds: float = 300
    memory_bytes: int = 536870912
    cpus: float = 1.0
    pids_limit: int = 64
    workspace_bytes: int = 67108864
    output_limit_bytes: int = 65536
    max_files: int = 1024
    seed: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "workspace", Path(self.workspace))
        if not isinstance(self.code, str) or len(self.code.encode()) > 1048576:
            raise ExecutionPolicyError("Code must be text no larger than 1 MiB")
        for name in ("memory_bytes", "pids_limit", "workspace_bytes", "output_limit_bytes", "max_files"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ExecutionPolicyError(f"{name} must be a positive integer")
        for name in ("cpus", "timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ExecutionPolicyError(f"{name} must be positive and finite")
        if self.memory_bytes < 16777216 or self.pids_limit < 8:
            raise ExecutionPolicyError("At least 16 MiB memory and 8 process slots are required")
        if self.workspace_bytes > self.memory_bytes:
            raise ExecutionPolicyError("tmpfs workspace cannot exceed the container memory limit")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed <= 4294967295):
            raise ExecutionPolicyError("seed must be an integer in [0, 2**32-1] or None")


@dataclass
class ExecutionResult:
    status: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str = "docker"
    isolation: str = "single_user_linux_container"
    evidence: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _CommandResult:
    returncode: Optional[int]
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    output_exceeded: bool = False
    cancelled: bool = False


def _client_environment() -> dict:
    # These configure the trusted local Docker client, not candidate execution.
    return {key: os.environ[key] for key in ("PATH", "HOME", "SYSTEMROOT") if key in os.environ}


def _command(args, *, data=None, timeout=15, limit=65536, cancel=None) -> _CommandResult:
    """Drain both pipes continuously, retaining at most ``limit`` bytes total."""
    try:
        process = subprocess.Popen(
            args, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_client_environment(),
            start_new_session=True,
        )
    except OSError as exc:
        return _CommandResult(None, stderr=str(exc).encode()[:limit])
    output = [bytearray(), bytearray()]
    lock = threading.Lock()
    exceeded = threading.Event()
    retained = 0

    def drain(pipe, index):
        nonlocal retained
        try:
            while True:
                chunk = pipe.read(8192)
                if not chunk:
                    break
                with lock:
                    room = max(0, limit - retained)
                    output[index].extend(chunk[:room])
                    retained += min(room, len(chunk))
                    if len(chunk) > room:
                        exceeded.set()
        finally:
            pipe.close()

    readers = [threading.Thread(target=drain, args=(process.stdout, 0), daemon=True),
               threading.Thread(target=drain, args=(process.stderr, 1), daemon=True)]
    for reader in readers:
        reader.start()
    writer = None
    if data is not None:
        def write():
            try:
                process.stdin.write(data)
                process.stdin.flush()
            except (OSError, BrokenPipeError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
    deadline = time.monotonic() + timeout
    result = _CommandResult(None)
    while process.poll() is None:
        result.cancelled = bool(cancel and cancel.is_set())
        result.output_exceeded = exceeded.is_set()
        result.timed_out = time.monotonic() >= deadline
        if result.cancelled or result.output_exceeded or result.timed_out:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            break
        time.sleep(0.02)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    for reader in readers:
        reader.join(timeout=3)
    if writer:
        writer.join(timeout=1)
    result.returncode = process.returncode
    result.stdout, result.stderr = bytes(output[0]), bytes(output[1])
    result.output_exceeded = result.output_exceeded or exceeded.is_set()
    return result


_BLOCKED_NAMES = {".env", ".git", ".aws", ".ssh", ".docker", "credentials", "id_rsa", "id_ed25519"}


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if name in (".", "./", ""):
        return ""
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or "\x00" in name:
        raise ExecutionPolicyError("Invalid workspace artifact path")
    parts = tuple(part for part in path.parts if part != ".")
    if any(part in _BLOCKED_NAMES or part.startswith(".env.") for part in parts):
        raise ExecutionPolicyError("Credential/control files are prohibited in candidate workspaces")
    return str(PurePosixPath(*parts))


def _snapshot(workspace: Path, request: ExecutionRequest) -> bytes:
    if workspace.is_symlink():
        raise ExecutionPolicyError("Workspace must not be a symlink")
    workspace.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    total, count = 0, 0
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(workspace.rglob("*")):
            name = _safe_name(path.relative_to(workspace).as_posix())
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
                raise ExecutionPolicyError("Workspace allows only regular files and directories")
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise ExecutionPolicyError("Hard-linked workspace files are prohibited")
            count += 1
            if count > request.max_files:
                raise ExecutionPolicyError("Workspace file-count limit exceeded")
            header = tarfile.TarInfo(name)
            header.uid = header.gid = 65534
            header.mode = 0o700 if path.is_dir() else 0o600
            if path.is_dir():
                header.type = tarfile.DIRTYPE
                archive.addfile(header)
            else:
                total += info.st_size
                if total > request.workspace_bytes:
                    raise ExecutionPolicyError("Workspace byte limit exceeded")
                header.size = info.st_size
                with path.open("rb") as handle:
                    archive.addfile(header, handle)
    return stream.getvalue()


def _restore(archive_bytes: bytes, workspace: Path, request: ExecutionRequest) -> dict:
    """Validate first, stage regular files, then replace the dedicated workspace."""
    parent = workspace.parent
    stage = Path(tempfile.mkdtemp(prefix=".p2b-output-", dir=parent))
    artifacts, names = {}, set()
    total, count = 0, 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            for member in archive:
                name = _safe_name(member.name)
                if not name:
                    if not member.isdir():
                        raise ExecutionPolicyError("Invalid archive root")
                    continue
                count += 1
                if count > request.max_files or name in names:
                    raise ExecutionPolicyError("Duplicate artifact or file-count limit exceeded")
                names.add(name)
                if not (member.isfile() or member.isdir()):
                    raise ExecutionPolicyError("Links and special output files are prohibited")
                destination = stage.joinpath(*PurePosixPath(name).parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                total += member.size
                if member.size < 0 or total > request.workspace_bytes:
                    raise ExecutionPolicyError("Output workspace byte limit exceeded")
                destination.parent.mkdir(parents=True, exist_ok=True)
                handle = archive.extractfile(member)
                if handle is None:
                    raise ExecutionPolicyError("Output artifact content is missing")
                content = handle.read(member.size + 1)
                if len(content) != member.size:
                    raise ExecutionPolicyError("Output artifact content length mismatch")
                destination.write_bytes(content)
                artifacts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        backup = parent / (".p2b-previous-" + uuid.uuid4().hex)
        workspace.rename(backup)
        try:
            stage.rename(workspace)
        except BaseException:
            backup.rename(workspace)
            raise
        shutil.rmtree(backup)
        return artifacts
    except (tarfile.TarError, OSError) as exc:
        raise ExecutionPolicyError("Invalid or uncommittable output archive") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)


_IMPORT = r'''
import sys, tarfile, pathlib
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    for member in archive:
        path = pathlib.Path("/workspace") / member.name
        if member.isdir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as dest:
                source = archive.extractfile(member)
                while True:
                    chunk = source.read(65536)
                    if not chunk: break
                    dest.write(chunk)
'''


def _export_script(request):
    return r'''
import os, sys, stat, tarfile
root = "/workspace"
count = total = 0
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = os.path.join(current, name)
            info = os.lstat(path)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise RuntimeError("Non-regular artifact rejected")
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise RuntimeError("Hardlinked artifact rejected")
            count += 1
            total += info.st_size if stat.S_ISREG(info.st_mode) else 0
            if count > MAX_FILES or total > MAX_BYTES:
                raise RuntimeError("Artifact limits exceeded")
            archive.add(path, arcname=os.path.relpath(path, root), recursive=False)
'''.replace("MAX_FILES", str(request.max_files)).replace("MAX_BYTES", str(request.workspace_bytes))


class DockerExecutor:
    """Single-job-at-a-time disposable Linux executor using a staged Python image."""

    def __init__(self, image="python:3.11-slim", binary="docker", lifecycle_callback=None,
                 gpu_request=None):
        self.image, self.binary = image, binary
        if lifecycle_callback is not None and not callable(lifecycle_callback):
            raise ExecutionPolicyError("lifecycle_callback must be a trusted controller callable")
        self.lifecycle_callback = lifecycle_callback
        self._lock = threading.Lock()
        self._active = None
        self._cancel = threading.Event()
        from .gpu import GPURequest
        self.gpu_request = GPURequest.from_dict(gpu_request) if gpu_request is not None else None
        self._gpu_devices = []

    def preflight(self, *, require_image=True) -> dict:
        if not shutil.which(self.binary):
            raise ExecutionUnavailable("Docker CLI is unavailable; no host execution fallback")
        endpoint_result = _command([self.binary, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], timeout=10)
        endpoint = endpoint_result.stdout.decode(errors="replace").strip()
        if endpoint_result.returncode != 0 or not endpoint.startswith("unix://"):
            raise ExecutionUnavailable("The initial execution profile requires an operator-controlled local Unix Docker socket")
        server = _command([self.binary, "info", "--format", "{{.OSType}}"], timeout=10)
        if server.returncode != 0 or server.stdout.strip() != b"linux":
            raise ExecutionUnavailable("A reachable Linux Docker engine is required")
        if not require_image:
            return {"backend": "docker", "daemon_endpoint": endpoint,
                    "isolation": "single_user_linux_container", "network": "none"}
        image = _command([self.binary, "image", "inspect", "--format", "{{.Id}}", self.image], timeout=10)
        identity = image.stdout.decode(errors="replace").strip()
        if image.returncode != 0 or not re.fullmatch(r"sha256:[a-f0-9]{64}", identity):
            raise ExecutionUnavailable("Configured image is not staged locally; pull it explicitly before execution")
        result = {"backend": "docker", "image": self.image, "image_id": identity, "daemon_endpoint": endpoint,
                  "isolation": "single_user_linux_container", "network": "none"}
        if self.gpu_request is not None:
            from .gpu import preflight_gpu
            result["accelerator"] = preflight_gpu(self.gpu_request, self.binary, _command)
            self._gpu_devices = [device["uuid"] for device in result["accelerator"]["devices"]]
        return result

    def _create_command(self, name: str, request: ExecutionRequest, image_id: str) -> list:
        clean = ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/workspace", "TMPDIR=/tmp"]
        command = [
            self.binary, "create", "--pull", "never", "--name", name,
            "--network", "none", "--read-only", "--user", "65534:65534",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
            "--pids-limit", str(request.pids_limit), "--memory", str(request.memory_bytes),
            "--memory-swap", str(request.memory_bytes), "--cpus", str(request.cpus),
            "--ulimit", "nofile=256:256", "--ulimit", "fsize=67108864:67108864",
            "--log-driver", "none", "--workdir", "/workspace",
            "--tmpfs", f"/workspace:rw,noexec,nosuid,nodev,size={request.workspace_bytes},mode=700,uid=65534,gid=65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16777216,mode=1777",
            image_id, *clean, "python", "-I", "-u", "-c", "import time; time.sleep(31536000)",
        ]
        if self.gpu_request is not None:
            if not self._gpu_devices:
                raise ExecutionPolicyError("GPU admission must succeed before container creation")
            # Docker's device-request CSV grammar requires quotes around a list.
            argument = '"device=' + ",".join(self._gpu_devices) + '"'
            command[2:2] = ["--gpus", argument, "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
        return command

    def _exec(self, name, code, *, data=None, timeout=15, limit=65536, cancel=None):
        return _command([
            self.binary, "exec", "-i", "--user", "65534:65534", "--workdir", "/workspace", name,
            "/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/workspace", "TMPDIR=/tmp",
            "python", "-I", "-u", "-c", code,
        ], data=data, timeout=timeout, limit=limit, cancel=cancel)

    def _remove(self, name):
        cleanup = _command([self.binary, "rm", "--force", name], timeout=10)
        missing = b"No such container" in cleanup.stderr or b"No such object" in cleanup.stderr
        return cleanup.returncode == 0 or missing

    def _notify(self, event, payload):
        if self.lifecycle_callback is not None:
            try:
                self.lifecycle_callback(event, dict(payload))
            except Exception as exc:
                raise LifecyclePersistenceError("Controller lifecycle persistence failed for " + event) from exc

    def reconcile(self, container_name: str, terminate: bool = False) -> dict:
        """Inspect an exact recorded job; terminate only on explicit request.

        A missing staged image does not prevent orphan cleanup. No arbitrary
        container name, filter, label query, or remote Docker daemon is accepted.
        """
        if not isinstance(container_name, str) or not re.fullmatch(r"p2b-[a-f0-9]{32}", container_name):
            raise ExecutionPolicyError("Reconciliation requires an exact recorded p2b UUID container name")
        if type(terminate) is not bool:
            raise ExecutionPolicyError("terminate must be an explicit boolean")
        metadata = self.preflight(require_image=False)
        metadata["container_name"] = container_name
        inspected = _command([self.binary, "inspect", "--type", "container", "--format", "{{json .State}}", container_name], timeout=10)
        missing = b"No such container" in inspected.stderr or b"No such object" in inspected.stderr
        if inspected.returncode != 0:
            return dict(metadata, found=False if missing else None, running=False if missing else None,
                        cleanup_confirmed=missing, termination_status="confirmed" if missing else "unknown",
                        error=None if missing else "Container inspection failed; outcome remains unknown")
        try:
            state = json.loads(inspected.stdout)
            if not isinstance(state, dict) or type(state.get("Running")) is not bool:
                raise ValueError("invalid state")
        except (ValueError, UnicodeError) as exc:
            raise ExecutionUnavailable("Docker returned an invalid container state") from exc
        running = state["Running"]
        confirmed = self._remove(container_name) if terminate else False
        return dict(metadata, found=True, running=False if confirmed else running,
                    cleanup_confirmed=confirmed, termination_status="confirmed" if confirmed else ("running" if running else "stopped"),
                    state={key: state.get(key) for key in ("Status", "Running", "Paused", "OOMKilled", "ExitCode")})

    def cancel(self) -> bool:
        """Request cancellation; return true only when Docker confirms removal."""
        self._cancel.set()
        with self._lock:
            name = self._active
        return bool(name and self._remove(name))

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if self.gpu_request is None:
            return self._execute_isolated(request)
        from .gpu import GPUDeviceLease
        self.preflight()
        with GPUDeviceLease(self._gpu_devices):
            return self._execute_isolated(request)

    def _execute_isolated(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.monotonic()
        metadata = self.preflight()
        snapshot = _snapshot(request.workspace, request)
        name = "p2b-" + uuid.uuid4().hex
        with self._lock:
            if self._active:
                raise ExecutionPolicyError("Executor instance already has an active job")
            self._active = name
            self._cancel.clear()
        metadata.update({"container_name": name, "input_sha256": hashlib.sha256(snapshot).hexdigest(),
                         "limits": {k: v for k, v in asdict(request).items() if k not in ("code", "workspace")}})
        metadata["seed"] = request.seed
        metadata["seed_scope"] = "python_random_and_PAPER2BENCH_SEED_only" if request.seed is not None else "unset"
        status, error, output, artifacts = "execution_error", "", _CommandResult(None), {}
        create_attempted = False
        try:
            self._notify("job_intent", {"container_name": name, "image_id": metadata["image_id"],
                                        "input_sha256": metadata["input_sha256"]})
            create_attempted = True
            created_result = _command(self._create_command(name, request, metadata["image_id"]), timeout=15)
            if created_result.returncode != 0:
                error = "Container creation failed: " + created_result.stderr.decode(errors="replace")
            else:
                metadata["container_id"] = created_result.stdout.decode(errors="replace").strip()
                self._notify("job_created", {"container_name": name, "container_id": metadata["container_id"],
                                             "image_id": metadata["image_id"]})
                launched = _command([self.binary, "start", name], timeout=15)
                if launched.returncode != 0:
                    error = "Container startup failed: " + launched.stderr.decode(errors="replace")
                else:
                    imported = self._exec(name, _IMPORT, data=snapshot, timeout=30)
                    if imported.returncode != 0:
                        error = "Input staging failed: " + imported.stderr.decode(errors="replace")
                    else:
                        seed_prelude = ("import os, random; random.seed(" + str(request.seed) + "); "
                                        "os.environ['PAPER2BENCH_SEED']=" + repr(str(request.seed)) + "; ") if request.seed is not None else ""
                        output = self._exec(
                            name, seed_prelude + 'import sys; exec(compile(sys.stdin.read(), "<candidate>", "exec"))',
                            data=request.code.encode(), timeout=request.timeout_seconds,
                            limit=request.output_limit_bytes, cancel=self._cancel,
                        )
                        if output.cancelled:
                            status = "cancelled"
                        elif output.timed_out:
                            status = "timed_out"
                        elif output.output_exceeded:
                            status, error = "failed", "Candidate output limit exceeded"
                        elif output.returncode is None:
                            error = "Candidate command could not start"
                        else:
                            status = "succeeded" if output.returncode == 0 else "failed"
                            # SIGSTOP cannot be caught. Stop same-UID descendants before
                            # a separate trusted exporter snapshots the bounded tmpfs.
                            stopped = self._exec(name, "import os, signal\ntry:\n os.kill(-1, signal.SIGSTOP)\nexcept ProcessLookupError:\n pass", timeout=10)
                            if stopped.returncode != 0:
                                status, error = "execution_error", "Could not quiesce candidate descendants"
                            else:
                                archive_limit = request.workspace_bytes + request.max_files * 2048 + 1048576
                                exported = self._exec(name, _export_script(request), timeout=30, limit=archive_limit)
                                if exported.returncode != 0 or exported.output_exceeded or exported.timed_out:
                                    status, error = "execution_error", "Artifact export failed or exceeded limits"
                                else:
                                    try:
                                        artifacts = _restore(exported.stdout, request.workspace, request)
                                    except ExecutionPolicyError as exc:
                                        status, error = "execution_error", str(exc)
        except LifecyclePersistenceError as exc:
            status, error = "execution_error", str(exc)
        finally:
            # Even an ambiguous create response may have created the named job.
            metadata["cleanup_confirmed"] = self._remove(name) if create_attempted else True
            with self._lock:
                self._active = None
            try:
                self._notify("job_cleanup", {"container_name": name, "container_id": metadata.get("container_id"),
                                             "cleanup_confirmed": metadata["cleanup_confirmed"]})
            except LifecyclePersistenceError as exc:
                status, error = "execution_error", str(exc)
                metadata["lifecycle_recording_error"] = True
        if not metadata["cleanup_confirmed"]:
            metadata["termination_status"] = "unknown"
            status = "cancellation_pending" if self._cancel.is_set() else "execution_error"
            error = (error + "; " if error else "") + "Container termination could not be confirmed; reconcile container_name"
        else:
            metadata["termination_status"] = "confirmed"
            if self._cancel.is_set():
                status = "cancelled"
        if error:
            metadata["error"] = error
        metadata["output_truncated"] = output.output_exceeded
        return ExecutionResult(
            status=status, exit_code=None if (output.timed_out or output.cancelled or output.output_exceeded) else output.returncode,
            stdout=output.stdout.decode(errors="replace"),
            stderr=output.stderr.decode(errors="replace") + ("\n" + error if error else ""),
            duration_seconds=round(time.monotonic() - started, 6),
            evidence=metadata, artifacts=artifacts,
        )
