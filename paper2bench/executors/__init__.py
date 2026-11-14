"""Isolated candidate execution. There is deliberately no implicit host runner."""

from .docker import (
    DockerExecutor,
    ExecutionPolicyError,
    ExecutionRequest,
    ExecutionResult,
    ExecutionUnavailable,
)
from .gpu import GPURequest
from .ssh import SSHExecutor, RemoteProtocolError

__all__ = [
    "DockerExecutor", "ExecutionRequest", "ExecutionResult",
    "ExecutionUnavailable", "ExecutionPolicyError",
    "GPURequest", "SSHExecutor", "RemoteProtocolError",
]
