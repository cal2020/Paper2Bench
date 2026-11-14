"""Operator-selected execution backends with persistent, reproducible selection."""
from __future__ import annotations

from dataclasses import asdict

from .executors import DockerExecutor, GPURequest, SSHExecutor
from .runtime_config import PolicyDenied, get_context

_FIELDS = {"backend", "image", "gpu", "ssh", "profile", "network", "limits", "max_parallel_jobs", "model_broker"}
_SSH_FIELDS = {"host", "user", "known_hosts", "identity_file", "port", "remote_python", "remote_policy",
               "connect_timeout_seconds"}


def executor_selection(executor, *, image=None):
    if isinstance(executor, SSHExecutor):
        result = {"backend": "ssh", "image": executor.image,
                  "ssh": {key: getattr(executor, key) for key in _SSH_FIELDS}}
    elif isinstance(executor, DockerExecutor):
        result = {"backend": "docker", "image": executor.image}
    else:
        result = {"backend": "injected", "image": image,
                  "class": type(executor).__module__ + "." + type(executor).__qualname__}
    gpu = getattr(executor, "gpu_request", None)
    if gpu is not None:
        result["gpu"] = asdict(gpu)
        result["gpu"]["device_ids"] = list(result["gpu"]["device_ids"])
    return result


def build_executor(*, image=None, selection=None):
    context = get_context()
    operator = dict(context.config.execution) if context else {}
    if set(operator) - _FIELDS:
        raise ValueError("Unsupported operator execution fields")
    if operator.get("network", "none") != "none" or operator.get("max_parallel_jobs", 1) != 1:
        raise ValueError("Candidate networking and parallel controller jobs are not supported")
    if selection is not None:
        if not isinstance(selection, dict) or set(selection) - {"backend", "image", "gpu", "ssh", "class"}:
            raise ValueError("Invalid frozen executor selection")
        requested_backend = operator.get("backend", selection["backend"])
        if requested_backend == "oci":
            requested_backend = "docker"
        if requested_backend != selection["backend"]:
            raise ValueError("Current operator backend differs from the recorded run")
        config = selection
    else:
        config = operator
    backend = config.get("backend", "docker")
    gpu = GPURequest.from_dict(config["gpu"]) if config.get("gpu") is not None else None
    selected_image = image or config.get("image") or "python:3.11-slim"
    if backend in {"docker", "oci"}:
        return DockerExecutor(image=selected_image, gpu_request=gpu)
    if backend == "ssh":
        if context and context.config.policy.get("data_egress") == "local_only":
            raise PolicyDenied("A local-only run cannot dispatch to a remote SSH worker")
        ssh = config.get("ssh")
        if not isinstance(ssh, dict) or set(ssh) - _SSH_FIELDS:
            raise ValueError("SSH backend requires explicit pinned operator configuration")
        return SSHExecutor(**ssh, image=selected_image, gpu_request=gpu)
    if backend == "injected":
        raise ValueError("This run requires the original caller-supplied executor")
    raise ValueError("Unsupported execution backend")
