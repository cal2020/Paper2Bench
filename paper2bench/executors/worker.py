"""One-request SSH worker. Install this package on an operator-controlled host.

The worker accepts data on stdin and streams acknowledged lifecycle events on
stdout. Its policy is an operator-owned local file, never a field in a task.
Candidate code is always passed to DockerExecutor and never executed here.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .docker import DockerExecutor, ExecutionPolicyError, ExecutionRequest, _restore, _snapshot
from .gpu import GPURequest
from .remote_protocol import (MAX_CONTROL_BYTES, MAX_FRAME_BYTES, MAX_WORKSPACE_BYTES,
                              PROTOCOL, archive_decode, decode, encode)


@dataclass
class WorkerPolicy:
    image: str = "python:3.11-slim"
    gpu_request: dict | None = None
    ceilings: dict = field(default_factory=lambda: {
        "timeout_seconds": 3600, "memory_bytes": 4294967296, "cpus": 4,
        "pids_limit": 256, "workspace_bytes": MAX_WORKSPACE_BYTES,
        "output_limit_bytes": 1048576, "max_files": 1024})

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"schema_version", "image", "gpu_request", "ceilings"}:
            raise ExecutionPolicyError("Unsupported worker policy")
        if value.get("schema_version", 1) != 1 or type(value.get("schema_version", 1)) is not int:
            raise ExecutionPolicyError("Unsupported worker policy version")
        policy = cls()
        if "image" in value:
            if not isinstance(value["image"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}", value["image"]):
                raise ExecutionPolicyError("Invalid worker image")
            policy.image = value["image"]
        if value.get("gpu_request") is not None:
            policy.gpu_request = asdict(GPURequest.from_dict(value["gpu_request"]))
        if "ceilings" in value:
            supplied = value["ceilings"]
            if not isinstance(supplied, dict) or set(supplied) - set(policy.ceilings):
                raise ExecutionPolicyError("Unknown worker resource ceiling")
            for key, amount in supplied.items():
                if type(amount) not in {int, float} or not 0 < amount <= policy.ceilings[key]:
                    raise ExecutionPolicyError("Worker policy may only tighten built-in ceilings")
            policy.ceilings.update(supplied)
        return policy

    def request(self, value, workspace):
        allowed = {"code", "timeout_seconds", "memory_bytes", "cpus", "pids_limit",
                   "workspace_bytes", "output_limit_bytes", "max_files", "seed", "snapshot_b64"}
        if not isinstance(value, dict) or set(value) - allowed or "snapshot_b64" not in value:
            raise ExecutionPolicyError("Invalid remote execution request")
        data = {k: v for k, v in value.items() if k != "snapshot_b64"}
        request = ExecutionRequest(workspace=workspace, **data)
        for key, maximum in self.ceilings.items():
            if getattr(request, key) > maximum:
                raise ExecutionPolicyError("Remote request exceeds worker resource policy")
        return request


def serve(*, policy=None, executor_factory=DockerExecutor, input_stream=None, output_stream=None):
    policy = policy or WorkerPolicy()
    source, target = input_stream or sys.stdin.buffer, output_stream or sys.stdout.buffer
    request_id = None

    def emit(value):
        target.write(encode(dict(value, protocol=PROTOCOL, request_id=request_id)))
        target.flush()

    try:
        raw = source.readline(MAX_FRAME_BYTES + 1)
        message = decode(raw)
        if (set(message) - {"protocol", "request_id", "operation", "executor", "request", "container_name", "terminate"}
                or message.get("protocol") != PROTOCOL or not isinstance(message.get("request_id"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", message["request_id"])):
            raise ExecutionPolicyError("Invalid remote protocol envelope")
        request_id = message["request_id"]
        selection = message.get("executor", {})
        if not isinstance(selection, dict) or set(selection) - {"image", "gpu_request"}:
            raise ExecutionPolicyError("Invalid remote executor selection")
        requested_gpu = selection.get("gpu_request")
        if requested_gpu is not None:
            requested_gpu = asdict(GPURequest.from_dict(requested_gpu))
        if requested_gpu != policy.gpu_request:
            raise ExecutionPolicyError("Requested accelerator differs from operator worker policy")

        def lifecycle(event, payload):
            emit({"type": "event", "event": event, "payload": payload})
            acknowledgement = decode(source.readline(MAX_CONTROL_BYTES + 1), MAX_CONTROL_BYTES)
            if acknowledgement != {"ack": event, "request_id": request_id}:
                raise ExecutionPolicyError("Controller did not acknowledge durable lifecycle event")

        executor = executor_factory(image=policy.image, gpu_request=policy.gpu_request, lifecycle_callback=lifecycle)
        operation = message.get("operation")
        if operation == "reconcile":
            result = executor.reconcile(message.get("container_name"), terminate=message.get("terminate", False))
            emit({"type": "result", "result": result})
            return 0
        if operation not in {"preflight", "execute"}:
            raise ExecutionPolicyError("Unsupported remote worker operation")
        metadata = executor.preflight()
        requested_image = selection.get("image", policy.image)
        if requested_image not in {policy.image, metadata["image_id"]}:
            raise ExecutionPolicyError("Requested image is not the worker's pinned allowed image")
        executor.image = metadata["image_id"]
        if operation == "preflight":
            emit({"type": "result", "result": dict(metadata, worker_protocol=PROTOCOL,
                                                      worker_resource_ceilings=policy.ceilings)})
            return 0
        with tempfile.TemporaryDirectory(prefix="p2b-remote-") as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            request = policy.request(message.get("request"), workspace)
            _restore(archive_decode(message["request"]["snapshot_b64"]), workspace, request)
            result = executor.execute(request)
            archive = _snapshot(workspace, request)
            emit({"type": "result", "result": result.to_dict(),
                  "snapshot_b64": base64.b64encode(archive).decode("ascii")})
        return 0
    except Exception as error:
        try:
            emit({"type": "error", "error_type": type(error).__name__,
                  "message": "Remote worker rejected or could not complete the request"})
        except (OSError, ValueError):
            pass
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paper2Bench bounded SSH worker")
    parser.add_argument("--policy", help="Operator-owned local worker policy JSON")
    args = parser.parse_args(argv)
    policy = WorkerPolicy()
    if args.policy:
        path = Path(args.policy)
        if path.is_symlink() or not path.is_file():
            parser.error("Worker policy must be a regular nonsymlink file")
        raw = path.read_bytes()
        if len(raw) > MAX_CONTROL_BYTES:
            parser.error("Worker policy exceeds limit")
        policy = WorkerPolicy.from_dict(decode(raw, MAX_CONTROL_BYTES))
    raise SystemExit(serve(policy=policy))


if __name__ == "__main__":
    main()
