"""Explicit NVIDIA device admission for a dedicated Linux worker.

Docker device selection does not provide a hard VRAM quota. The operator must
dedicate the requested devices; advisory locks coordinate this user's local
Paper2Bench controllers, not arbitrary processes or other system users.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .docker import ExecutionPolicyError, ExecutionUnavailable

_UUID = re.compile(r"GPU-[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}")


@dataclass(frozen=True)
class GPURequest:
    device_ids: tuple[str, ...]
    minimum_memory_bytes: int = 0
    dedicated: bool = True
    require_idle: bool = True
    hard_memory_limit: bool = False

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            result = value
        else:
            if not isinstance(value, dict) or set(value) - {"device_ids", "minimum_memory_bytes", "dedicated", "require_idle", "hard_memory_limit"}:
                raise ExecutionPolicyError("GPU request contains unsupported fields")
            if value.get("dedicated") is not True:
                raise ExecutionPolicyError("GPU configuration requires an explicit dedicated-device assertion")
            data = dict(value)
            ids = data.get("device_ids")
            if not isinstance(ids, (list, tuple)):
                raise ExecutionPolicyError("GPU device_ids must be explicit NVIDIA UUIDs")
            data["device_ids"] = tuple(ids)
            result = cls(**data)
        if (not result.device_ids or len(result.device_ids) > 8
                or any(not isinstance(x, str) or not _UUID.fullmatch(x) for x in result.device_ids)
                or len(set(result.device_ids)) != len(result.device_ids)):
            raise ExecutionPolicyError("Select one to eight distinct NVIDIA GPU UUIDs")
        if type(result.minimum_memory_bytes) is not int or result.minimum_memory_bytes < 0:
            raise ExecutionPolicyError("GPU minimum_memory_bytes must be a nonnegative integer")
        for name in ("dedicated", "require_idle", "hard_memory_limit"):
            if type(getattr(result, name)) is not bool:
                raise ExecutionPolicyError(f"GPU {name} must be a boolean")
        if not result.dedicated:
            raise ExecutionPolicyError("GPU execution requires an operator assertion of dedicated devices")
        if result.hard_memory_limit:
            raise ExecutionPolicyError("This Docker profile cannot enforce a hard VRAM quota")
        return result


def preflight_gpu(request, docker_binary, command):
    request = GPURequest.from_dict(request)
    if not shutil.which("nvidia-smi"):
        raise ExecutionUnavailable("GPU profile requires nvidia-smi on the dedicated Linux worker")
    runtime = command([docker_binary, "info", "--format", "{{json .Runtimes}}"], timeout=10)
    try:
        runtimes = json.loads(runtime.stdout)
    except (ValueError, UnicodeError) as error:
        raise ExecutionUnavailable("Cannot inspect Docker accelerator runtimes") from error
    if runtime.returncode != 0 or not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        raise ExecutionUnavailable("Docker NVIDIA runtime is not configured")
    inventory = command(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
                        timeout=10, limit=65536)
    if inventory.returncode != 0 or inventory.output_exceeded or inventory.timed_out:
        raise ExecutionUnavailable("NVIDIA driver/device inventory is unavailable")
    devices = {}
    try:
        for line in inventory.stdout.decode("utf-8").splitlines():
            uuid, total, used, driver = [part.strip() for part in line.split(",")]
            if not _UUID.fullmatch(uuid) or uuid in devices:
                raise ValueError("invalid UUID")
            total, used = int(total) * 1048576, int(used) * 1048576
            if total <= 0 or not 0 <= used <= total or not re.fullmatch(r"[0-9.]{1,40}", driver):
                raise ValueError("invalid device metadata")
            devices[uuid] = {"uuid": uuid, "memory_total_bytes": total, "memory_used_bytes": used, "driver_version": driver}
    except (ValueError, UnicodeError) as error:
        raise ExecutionUnavailable("Malformed NVIDIA device inventory") from error
    selected = []
    for identity in request.device_ids:
        device = devices.get(identity)
        if device is None or device["memory_total_bytes"] < request.minimum_memory_bytes:
            raise ExecutionUnavailable("Requested GPU is missing or has insufficient total memory")
        selected.append(device)
    if request.require_idle:
        activity = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], timeout=10, limit=65536)
        if activity.returncode != 0 or activity.output_exceeded or activity.timed_out:
            raise ExecutionUnavailable("GPU process inventory could not be verified")
        for line in activity.stdout.decode(errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not _UUID.fullmatch(fields[0]) or not fields[1].isdigit():
                raise ExecutionUnavailable("Malformed GPU activity inventory")
            if fields[0] in request.device_ids:
                raise ExecutionUnavailable("Requested dedicated GPU has an active compute process")
    return {"kind": "nvidia_gpu", "devices": selected, "hard_vram_quota": False,
            "allocation_assurance": "operator_dedicated_with_per_user_advisory_lock",
            "idle_compute_inventory_checked": request.require_idle}


class GPUDeviceLease:
    def __init__(self, device_ids):
        self.devices = tuple(sorted(device_ids))
        self.handles = []

    def __enter__(self):
        try:
            import fcntl
        except ImportError as error:
            raise ExecutionUnavailable("GPU advisory admission requires a POSIX worker") from error
        root = Path(tempfile.gettempdir()) / ("paper2bench-gpu-" + str(os.getuid()))
        root.mkdir(mode=0o700, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ExecutionPolicyError("GPU lease directory is not private to the worker user")
        try:
            for uuid in self.devices:
                if not _UUID.fullmatch(uuid):
                    raise ExecutionPolicyError("Invalid GPU lease identity")
                fd = os.open(root / uuid, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
                self.handles.append(fd)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                    raise ExecutionPolicyError("Invalid GPU lease file")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ExecutionUnavailable("GPU is already leased by another Paper2Bench controller") from error
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for fd in self.handles:
            os.close(fd)
        self.handles.clear()
