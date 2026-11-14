"""Pinned-host SSH transport for the bounded Paper2Bench Docker worker.

The remote command is fixed and quoted; task/code bytes travel only through a
data-only stdio protocol. SSH keys, host configuration and provider credentials
are never sent to candidate containers. The operator owns both the account and
the worker policy. This adapter does not provision a host or install software.
"""
from __future__ import annotations

import base64
import math
import os
import selectors
from collections import deque
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .docker import (ExecutionPolicyError, ExecutionRequest, ExecutionResult,
                     ExecutionUnavailable, _client_environment, _restore, _snapshot)
from .gpu import GPURequest
from .remote_protocol import (MAX_CONTROL_BYTES, MAX_FRAME_BYTES, MAX_WORKSPACE_BYTES,
                              PROTOCOL, archive_decode, decode, encode)


class RemoteProtocolError(ExecutionUnavailable):
    """A remote outcome is unknown unless explicit cleanup evidence says otherwise."""


class SSHExecutor:
    def __init__(self, *, host, user, known_hosts, identity_file, port=22,
                 image="python:3.11-slim", remote_python="python3", remote_policy=None,
                 binary="ssh", connect_timeout_seconds=10, lifecycle_callback=None,
                 gpu_request=None):
        if (not isinstance(host, str) or len(host) > 253
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host)
                or ".." in host):
            raise ExecutionPolicyError("SSH host must be an explicit DNS name or IPv4 address")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", user):
            raise ExecutionPolicyError("SSH user must be explicit and normalized")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ExecutionPolicyError("Invalid SSH port")
        if type(connect_timeout_seconds) is not int or not 1 <= connect_timeout_seconds <= 60:
            raise ExecutionPolicyError("SSH connect timeout must be an integer in [1, 60]")
        if not isinstance(remote_python, str) or not re.fullmatch(r"(?:/[A-Za-z0-9_./-]+|python3(?:\.[0-9]+)?)", remote_python):
            raise ExecutionPolicyError("Remote interpreter must be a fixed normalized executable")
        if remote_policy is not None and (not isinstance(remote_policy, str)
                or not re.fullmatch(r"/[A-Za-z0-9_./-]{1,500}", remote_policy)
                or ".." in Path(remote_policy).parts):
            raise ExecutionPolicyError("Remote policy must be a normalized absolute operator path")
        self.host, self.user, self.port, self.image = host, user, port, image
        self.remote_python, self.remote_policy, self.binary = remote_python, remote_policy, binary
        self.connect_timeout_seconds = connect_timeout_seconds
        self.known_hosts = self._local_file(known_hosts, "known_hosts")
        self.identity_file = self._local_file(identity_file, "identity_file")
        if lifecycle_callback is not None and not callable(lifecycle_callback):
            raise ExecutionPolicyError("lifecycle_callback must be a trusted controller callable")
        self.lifecycle_callback = lifecycle_callback
        self.gpu_request = GPURequest.from_dict(gpu_request) if gpu_request is not None else None
        self._lock = threading.Lock()
        self._active = None
        self._cancel = threading.Event()
        self._last_request_id = None
        self._running = False

    @staticmethod
    def _local_file(value, label):
        path = Path(value)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1048576:
            raise ExecutionPolicyError(f"SSH {label} must be an explicit regular file no larger than 1 MiB")
        return str(path.resolve())

    def _command(self):
        if not shutil.which(self.binary):
            raise ExecutionUnavailable("SSH client is unavailable")
        command = [self.remote_python, "-m", "paper2bench.executors.worker"]
        if self.remote_policy is not None:
            command.extend(["--policy", self.remote_policy])
        return [self.binary, "-F", "/dev/null", "-T", "-p", str(self.port),
                "-i", self.identity_file, "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
                "-o", "UpdateHostKeys=no", "-o", "VerifyHostKeyDNS=no",
                "-o", "IdentityAgent=none", "-o", "UserKnownHostsFile=" + self.known_hosts,
                "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ClearAllForwardings=yes",
                "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no",
                "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "ControlMaster=no",
                "-o", "CanonicalizeHostname=no", "-o", "ConnectTimeout=" + str(self.connect_timeout_seconds),
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                "--", self.user + "@" + self.host, shlex.join(command)]

    def _selection(self):
        return {"image": self.image, "gpu_request": asdict(self.gpu_request) if self.gpu_request else None}

    def _rpc(self, operation, *, timeout_seconds, request=None, container_name=None, terminate=False,
             frame_limit=MAX_CONTROL_BYTES):
        from ..network import require_external_access
        require_external_access("Remote worker dispatch")
        if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ExecutionPolicyError("Remote deadline must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        identifier = uuid.uuid4().hex
        self._last_request_id = identifier
        message = {"protocol": PROTOCOL, "request_id": identifier, "operation": operation,
                   "executor": self._selection()}
        if request is not None:
            message["request"] = request
        if container_name is not None:
            message.update(container_name=container_name, terminate=terminate)
        payload = encode(message)
        try:
            # Raw, nonblocking pipes have no BufferedWriter lock. One selector
            # loop writes both the initial snapshot and every lifecycle ACK.
            process = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env=_client_environment(), start_new_session=True,
                                       bufsize=0)
        except OSError as error:
            raise ExecutionUnavailable("SSH worker transport could not start") from error
        selector = selectors.DefaultSelector()
        outgoing = deque([memoryview(payload)])
        stdout = bytearray()
        stderr = bytearray()
        result, count = None, 0
        stdin_registered = False
        stdout_eof = False

        def queue_ack(value):
            nonlocal stdin_registered
            outgoing.append(memoryview(encode(value)))
            if not stdin_registered:
                selector.register(process.stdin.fileno(), selectors.EVENT_WRITE, "stdin")
                stdin_registered = True

        def handle_frame(raw):
            nonlocal result, count
            frame = decode(raw, frame_limit)
            if frame.get("protocol") != PROTOCOL or frame.get("request_id") != identifier:
                raise RemoteProtocolError("SSH worker response identity mismatch")
            if frame.get("type") == "event":
                count += 1
                event, body = frame.get("event"), frame.get("payload")
                if (count > 8 or event not in {"job_intent", "job_created", "job_cleanup"}
                        or not isinstance(body, dict) or len(encode(body)) > MAX_CONTROL_BYTES
                        or not re.fullmatch(r"p2b-[a-f0-9]{32}", body.get("container_name", ""))):
                    raise RemoteProtocolError("Invalid remote execution lifecycle event")
                if event in {"job_intent", "job_created"}:
                    with self._lock:
                        self._active = body["container_name"]
                body = dict(body, worker_host=self.host, worker_user=self.user,
                            worker_port=self.port, remote_request_id=identifier)
                # The trusted controller persists before ACK. The ACK itself is
                # queued into the same nonblocking writer under the RPC deadline.
                if self.lifecycle_callback:
                    self.lifecycle_callback(event, body)
                queue_ack({"ack": event, "request_id": identifier})
            elif frame.get("type") == "error":
                raise RemoteProtocolError("SSH worker rejected the request: " + str(frame.get("error_type", "worker_error"))[:80])
            elif frame.get("type") == "result" and isinstance(frame.get("result"), dict):
                result = frame
            else:
                raise RemoteProtocolError("Malformed SSH worker response")

        try:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdout.fileno(), selectors.EVENT_READ, "stdout")
            selector.register(process.stderr.fileno(), selectors.EVENT_READ, "stderr")
            selector.register(process.stdin.fileno(), selectors.EVENT_WRITE, "stdin")
            stdin_registered = True
            while result is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteProtocolError("SSH worker deadline exceeded; remote outcome requires reconciliation")
                if stdout_eof:
                    raise RemoteProtocolError("SSH worker exited without a valid final result")
                for key, _ in selector.select(timeout=min(0.1, remaining)):
                    if time.monotonic() >= deadline:
                        raise RemoteProtocolError("SSH worker deadline exceeded; remote outcome requires reconciliation")
                    if key.data == "stdin":
                        try:
                            written = os.write(key.fd, outgoing[0][:65536])
                        except BlockingIOError:
                            continue
                        except OSError as error:
                            raise RemoteProtocolError("SSH request or lifecycle ACK could not be written") from error
                        if written <= 0:
                            raise RemoteProtocolError("SSH request writer stopped before completion")
                        outgoing[0] = outgoing[0][written:]
                        if not outgoing[0]:
                            outgoing.popleft()
                        if not outgoing:
                            selector.unregister(key.fd)
                            stdin_registered = False
                        continue
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    except OSError as error:
                        raise RemoteProtocolError("SSH response could not be read") from error
                    if not chunk:
                        selector.unregister(key.fd)
                        if key.data == "stdout":
                            stdout_eof = True
                        continue
                    if key.data == "stderr":
                        room = max(0, 16384 - len(stderr))
                        stderr.extend(chunk[:room])
                        continue
                    stdout.extend(chunk)
                    while result is None:
                        newline = stdout.find(b"\n")
                        if newline < 0:
                            if len(stdout) >= frame_limit:
                                raise RemoteProtocolError("Remote protocol frame exceeded its bound")
                            break
                        if newline + 1 > frame_limit:
                            raise RemoteProtocolError("Remote protocol frame exceeded its bound")
                        line = bytes(stdout[:newline + 1])
                        del stdout[:newline + 1]
                        handle_frame(line)
            if time.monotonic() >= deadline:
                raise RemoteProtocolError("SSH worker deadline exceeded while validating the response")
            return result
        finally:
            selector.close()
            # Terminate before closing any pipe. Even a nonreading peer holding
            # a full request pipe cannot delay this cleanup; no writer thread or
            # buffered stream lock survives the absolute RPC deadline.
            if process.poll() is None:
                if result is not None:
                    try:
                        process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        if process.poll() is None:
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass

    def preflight(self):
        response = self._rpc("preflight", timeout_seconds=self.connect_timeout_seconds + 30)["result"]
        identity = response.get("image_id")
        if response.get("worker_protocol") != PROTOCOL or not isinstance(identity, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", identity):
            raise RemoteProtocolError("Worker preflight did not provide a pinned execution image")
        self.image = identity
        return dict(response, backend="ssh", worker_host=self.host, worker_user=self.user,
                    worker_port=self.port, remote_request_id=self._last_request_id,
                    isolation="operator_controlled_remote_linux_container")

    def reconcile(self, container_name, terminate=False):
        if not isinstance(container_name, str) or not re.fullmatch(r"p2b-[a-f0-9]{32}", container_name):
            raise ExecutionPolicyError("Remote reconciliation requires an exact recorded job name")
        if type(terminate) is not bool:
            raise ExecutionPolicyError("terminate must be an explicit boolean")
        frame = self._rpc("reconcile", container_name=container_name, terminate=terminate,
                          timeout_seconds=self.connect_timeout_seconds + 30)
        response = frame["result"]
        if response.get("container_name") != container_name or type(response.get("cleanup_confirmed")) is not bool:
            raise RemoteProtocolError("Worker reconciliation identity or status is invalid")
        return dict(response, backend="ssh", worker_host=self.host, worker_user=self.user,
                    worker_port=self.port, remote_request_id=frame["request_id"])

    def cancel(self):
        self._cancel.set()
        with self._lock:
            name = self._active
        if not name:
            return False
        try:
            return self.reconcile(name, terminate=True).get("cleanup_confirmed") is True
        except ExecutionUnavailable:
            return False

    def execute(self, request: ExecutionRequest):
        with self._lock:
            if self._running or self._active:
                raise ExecutionPolicyError("SSH executor already has an active or unreconciled job")
            self._running = True
        try:
            return self._execute_remote(request)
        finally:
            with self._lock:
                self._running = False

    def _execute_remote(self, request: ExecutionRequest):
        if request.workspace_bytes > MAX_WORKSPACE_BYTES:
            raise ExecutionPolicyError("Remote workspace exceeds 64 MiB protocol admission limit")
        started = time.monotonic()
        self._cancel.clear()
        with self._lock:
            if self._active:
                raise ExecutionPolicyError("SSH executor already has an active or unreconciled job")
        admission = self.preflight()
        ceilings = admission.get("worker_resource_ceilings")
        if not isinstance(ceilings, dict):
            raise RemoteProtocolError("Worker did not declare resource admission ceilings")
        for key in ("timeout_seconds", "memory_bytes", "cpus", "pids_limit", "workspace_bytes", "output_limit_bytes", "max_files"):
            maximum = ceilings.get(key)
            if type(maximum) not in {int, float} or not math.isfinite(maximum) or getattr(request, key) > maximum:
                raise ExecutionPolicyError("Execution request exceeds the remote worker's admission ceiling")
        archive = _snapshot(request.workspace, request)
        payload = {k: v for k, v in asdict(request).items() if k != "workspace"}
        payload["snapshot_b64"] = base64.b64encode(archive).decode("ascii")
        frame_limit = min(MAX_FRAME_BYTES, ((request.workspace_bytes + request.max_files * 2048 + 1048576 + 2) // 3) * 4
                          + request.output_limit_bytes * 6 + 1048576)
        try:
            frame = self._rpc("execute", request=payload, timeout_seconds=request.timeout_seconds + self.connect_timeout_seconds + 90,
                              frame_limit=frame_limit)
            data = frame["result"]
            if (set(data) - {"status", "exit_code", "stdout", "stderr", "duration_seconds", "backend", "isolation", "evidence", "artifacts"}
                    or data.get("status") not in {"succeeded", "failed", "timed_out", "cancelled", "cancellation_pending", "execution_error"}
                    or (data.get("exit_code") is not None and type(data.get("exit_code")) is not int)
                    or not isinstance(data.get("stdout"), str) or not isinstance(data.get("stderr"), str)
                    or len(data["stdout"].encode()) + len(data["stderr"].encode()) > request.output_limit_bytes + 16384
                    or type(data.get("duration_seconds")) not in {int, float}
                    or not math.isfinite(data["duration_seconds"]) or data["duration_seconds"] < 0
                    or not isinstance(data.get("evidence"), dict)):
                raise RemoteProtocolError("Worker returned invalid typed execution evidence")
            evidence = data["evidence"]
            with self._lock:
                expected_name = self._active
            if (evidence.get("container_name") != expected_name or evidence.get("image_id") != self.image
                    or type(evidence.get("cleanup_confirmed")) is not bool):
                raise RemoteProtocolError("Worker execution identity or termination evidence is invalid")
            artifacts = _restore(archive_decode(frame.get("snapshot_b64")), request.workspace, request)
            evidence.update(worker_host=self.host, worker_user=self.user, worker_port=self.port,
                            remote_request_id=frame["request_id"], seed=request.seed,
                            seed_scope="python_random_and_PAPER2BENCH_SEED_only" if request.seed is not None else "unset")
            status = "cancelled" if self._cancel.is_set() and evidence["cleanup_confirmed"] else data["status"]
            result = ExecutionResult(status, data["exit_code"], data["stdout"], data["stderr"],
                                     time.monotonic() - started, backend="ssh",
                                     isolation="operator_controlled_remote_linux_container", evidence=evidence, artifacts=artifacts)
            if evidence["cleanup_confirmed"]:
                with self._lock:
                    self._active = None
            return result
        except Exception as error:
            with self._lock:
                name = self._active
            return ExecutionResult("execution_error", None, "", "SSH execution outcome requires reconciliation: " + type(error).__name__,
                                   time.monotonic() - started, backend="ssh", isolation="operator_controlled_remote_linux_container",
                                   evidence={"cleanup_confirmed": False, "termination_status": "unknown", "container_name": name,
                                             "worker_host": self.host, "worker_user": self.user, "worker_port": self.port,
                                             "image_id": self.image, "remote_request_id": self._last_request_id})
