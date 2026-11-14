"""Data-only bounded protocol shared by SSH controller and worker."""
from __future__ import annotations

import base64
import json
import math

from .docker import ExecutionPolicyError

PROTOCOL = "paper2bench.remote/1"
MAX_FRAME_BYTES = 192 * 1024 * 1024
MAX_CONTROL_BYTES = 65536
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_WORKSPACE_BYTES + 1024 * 2048 + 1048576


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionPolicyError("Duplicate remote protocol key")
        result[key] = value
    return result


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ExecutionPolicyError("Nonfinite remote protocol number")
    return value


def decode(raw, limit=MAX_FRAME_BYTES):
    if not isinstance(raw, bytes) or len(raw) > limit:
        raise ExecutionPolicyError("Remote protocol frame exceeds its limit")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_float=_float,
                           parse_constant=lambda _: (_ for _ in ()).throw(ExecutionPolicyError("Nonfinite remote protocol number")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ExecutionPolicyError("Invalid data-only remote protocol frame") from error
    if not isinstance(value, dict):
        raise ExecutionPolicyError("Remote protocol frame must be an object")
    return value


def encode(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(raw) > MAX_FRAME_BYTES:
        raise ExecutionPolicyError("Remote protocol response exceeds its limit")
    return raw


def archive_decode(text):
    if not isinstance(text, str) or len(text) > ((MAX_ARCHIVE_BYTES + 2) // 3) * 4:
        raise ExecutionPolicyError("Remote snapshot exceeds its limit")
    try:
        value = base64.b64decode(text, validate=True)
    except (ValueError, UnicodeError) as error:
        raise ExecutionPolicyError("Invalid snapshot encoding") from error
    if len(value) > MAX_ARCHIVE_BYTES:
        raise ExecutionPolicyError("Remote snapshot exceeds its limit")
    return value
