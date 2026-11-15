"""Verified artifacts and transactional SQLite run ownership.

Exceptions: InvalidArtifactHash rejects malformed references; ArtifactCorruption
means stored bytes failed verification or JSON decoding; ArtifactNotFound means a
valid reference is absent; RunNotFound means an unknown run; IdempotencyConflict
means a creation key was reused with a different specification. No operation
replaces an existing artifact or a terminal run. Callers must pass sanitized
errors; obvious credential assignments are additionally redacted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import canonical_bytes


class StorageError(Exception):
    """Base storage exception."""


class InvalidArtifactHash(StorageError, ValueError):
    """An artifact reference is not a lowercase SHA-256 digest."""


class ArtifactCorruption(StorageError):
    """Artifact content does not match its reference or expected encoding."""


class ArtifactNotFound(StorageError, FileNotFoundError):
    """The requested, well-formed artifact is absent."""


class RunNotFound(StorageError, KeyError):
    """The requested run does not exist."""


class IdempotencyConflict(StorageError, ValueError):
    """An idempotency key already identifies another specification."""


_HASH = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL = frozenset(
    {
        "completed",
        "invalid",
        "failed",
        "unsupported",
        "resource_exhausted",
        "cancelled",
        "outcome_unknown",
    }
)


def _validate_hash(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise InvalidArtifactHash("Expected a lowercase SHA-256 artifact reference.")
    return value


def _lease_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Lease duration must be a finite positive number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Lease duration must be a finite positive number.")
    return float(value)


def _safe_error(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Error must be a sanitized string or None.")
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", value)
    value = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret|authorization)\b\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        value,
    )
    return value[:1000]


_CALL_FIELDS = frozenset(
    {
        "phase",
        "call_id",
        "test_id",
        "requested_model",
        "provider_profile",
        "input_sha256",
        "ordinal",
        "output_sha256",
        "model",
        "request_id",
        "usage",
        "fingerprint",
        "finish_reason",
        "wall_seconds",
        "error",
    }
)
_CALL_PHASES = frozenset({"dispatched", "completed", "outcome_unknown", "failed"})
_CALL_HASH_FIELDS = frozenset({"provider_profile", "input_sha256", "output_sha256"})


def _call_metadata(call: dict) -> dict:
    """Bound the ledger to execution metadata; never accept raw prompts or secrets."""
    if not isinstance(call, dict):
        raise TypeError("Provider call metadata must be a JSON object.")
    if set(call) - _CALL_FIELDS:
        raise ValueError("Provider call contains unsupported metadata fields.")
    if call.get("phase") not in _CALL_PHASES:
        raise ValueError("Provider call phase is invalid.")
    if not isinstance(call.get("call_id"), str) or not call["call_id"]:
        raise ValueError("Provider call requires a call identifier.")
    for key, value in call.items():
        if key in _CALL_HASH_FIELDS:
            _validate_hash(value)
        elif key == "usage":
            if not isinstance(value, dict) or set(value) - {
                "input_tokens",
                "output_tokens",
                "total_tokens",
            }:
                raise ValueError("Provider usage must contain only token counts.")
            if any(v is not None and (type(v) is not int or v < 0) for v in value.values()):
                raise ValueError("Provider token counts must be nonnegative integers or unknown.")
        elif key == "wall_seconds":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Provider wall time must be finite and nonnegative.")
        elif key == "ordinal":
            if type(value) is not int or value < 0:
                raise ValueError("Provider call ordinal must be a nonnegative integer.")
        elif value is not None and (not isinstance(value, str) or len(value) > 4096):
            raise ValueError("Provider metadata strings must be at most 4096 characters.")
    error = call.get("error")
    if error is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error) is None:
        raise ValueError("Provider errors must be safe diagnostic codes.")
    encoded = canonical_bytes(call)
    if len(encoded) > 64 * 1024:
        raise ValueError("Provider call metadata exceeds the 64 KiB limit.")
    return json.loads(encoded)


class ArtifactStore:
    """Immutable content-addressed files, published atomically after fsync."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str) -> Path:
        value = _validate_hash(sha256)
        return self.root / value[:2] / value[2:]

    def put(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("Artifact data must be bytes.")
        sha256 = hashlib.sha256(data).hexdigest()
        path = self._path(sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ArtifactCorruption("Artifact directory must not be a symbolic link.")
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Linking publishes without replacing an already-existing reference.
                os.link(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except FileExistsError:
                self.get(sha256)  # Existing corrupt content is never silently repaired.
        finally:
            os.unlink(temporary)
        return sha256

    def get(self, sha256: str) -> bytes:
        path = self._path(sha256)
        if path.parent.is_symlink():
            raise ArtifactCorruption("Artifact directory must not be a symbolic link.")
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            raise ArtifactNotFound("Artifact does not exist.") from None
        except OSError:
            raise ArtifactCorruption("Artifact cannot be read safely.") from None
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ArtifactCorruption("Artifact is not a regular file.")
            data = handle.read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ArtifactCorruption("Artifact content failed SHA-256 verification.")
        return data

    def put_json(self, value: Any) -> str:
        return self.put(canonical_bytes(value))

    def get_json(self, sha256: str) -> Any:
        def reject_constant(_: str) -> None:
            raise ValueError("Nonfinite JSON constant")

        data = self.get(sha256)
        try:
            return json.loads(data, parse_constant=reject_constant)
        except (ValueError, UnicodeDecodeError):
            raise ArtifactCorruption("Artifact is not valid finite JSON.") from None


class RunStore:
    """One transaction per mutation; expired leases cannot publish or renew.

    Times are UTC Unix seconds. SQLite ownership decisions use the transaction's
    wall-clock time, so hosts accessing this database must have a trustworthy
    clock. Keep the database on a local filesystem with working SQLite locks.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
                    token INTEGER NOT NULL DEFAULT 0, attempt INTEGER NOT NULL DEFAULT 0,
                    worker TEXT, lease_expires_at REAL, result_hash TEXT, error TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    run_id TEXT NOT NULL REFERENCES runs(id), attempt INTEGER NOT NULL,
                    token INTEGER NOT NULL, worker TEXT NOT NULL, state TEXT NOT NULL,
                    started_at REAL NOT NULL, lease_expires_at REAL NOT NULL,
                    finished_at REAL, result_hash TEXT, error TEXT,
                    PRIMARY KEY (run_id, attempt)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id), at REAL NOT NULL,
                    kind TEXT NOT NULL, token INTEGER NOT NULL, attempt INTEGER NOT NULL,
                    details TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFound("Run does not exist.")
        return row

    @staticmethod
    def _event(
        connection: sqlite3.Connection, row: sqlite3.Row, kind: str, at: float, **details: Any
    ) -> None:
        connection.execute(
            "INSERT INTO events(run_id,at,kind,token,attempt,details) VALUES(?,?,?,?,?,?)",
            (
                row["id"],
                at,
                kind,
                row["token"],
                row["attempt"],
                json.dumps(details, sort_keys=True, allow_nan=False),
            ),
        )

    @staticmethod
    def _render(connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
        result = dict(row)
        result["run_id"] = row["id"]
        result["attempts"] = [
            dict(item)
            for item in connection.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY attempt", (row["id"],)
            )
        ]
        return result

    def create_run(self, spec_hash: str, idempotency_key: str) -> dict:
        _validate_hash(spec_hash)
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("A nonempty idempotency key is required.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM runs WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["spec_hash"] != spec_hash:
                    raise IdempotencyConflict(
                        "Idempotency key identifies a different specification."
                    )
                return self._render(connection, existing)
            now = time.time()
            run_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO runs(id,spec_hash,idempotency_key,state,created_at,updated_at) "
                "VALUES(?,?,?,'queued',?,?)",
                (run_id, spec_hash, idempotency_key, now, now),
            )
            row = self._row(connection, run_id)
            self._event(connection, row, "created", now)
            return self._render(connection, row)

    def get_run(self, run_id: str) -> dict:
        with self._connect() as connection:
            connection.execute("BEGIN")
            return self._render(connection, self._row(connection, run_id))

    def claim(self, run_id: str, worker: str, lease_seconds: float = 30) -> dict | None:
        duration = _lease_seconds(lease_seconds)
        if not isinstance(worker, str) or not worker.strip():
            raise ValueError("A nonempty worker identifier is required.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, run_id)
            now = time.time()
            if row["state"] == "running":
                if row["lease_expires_at"] > now:
                    return None
                connection.execute(
                    "UPDATE attempts SET state='expired',finished_at=? "
                    "WHERE run_id=? AND attempt=? AND state='running'",
                    (now, run_id, row["attempt"]),
                )
                self._event(connection, row, "lease_expired", now)
            elif row["state"] != "queued":
                return None
            token, attempt = row["token"] + 1, row["attempt"] + 1
            expires = now + duration
            connection.execute(
                "UPDATE runs SET state='running',token=?,attempt=?,worker=?,"
                "lease_expires_at=?,updated_at=? WHERE id=?",
                (token, attempt, worker, expires, now, run_id),
            )
            connection.execute(
                "INSERT INTO attempts(run_id,attempt,token,worker,state,started_at,lease_expires_at) "
                "VALUES(?,?,?,?,'running',?,?)",
                (run_id, attempt, token, worker, now, expires),
            )
            row = self._row(connection, run_id)
            self._event(connection, row, "claimed", now, worker=worker)
            return self._render(connection, row)

    def heartbeat(self, run_id: str, token: int, lease_seconds: float = 30) -> bool:
        duration = _lease_seconds(lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, run_id)
            now = time.time()
            if not self._live(row, token, now):
                return False
            expires = now + duration
            connection.execute(
                "UPDATE runs SET lease_expires_at=?,updated_at=? WHERE id=?",
                (expires, now, run_id),
            )
            connection.execute(
                "UPDATE attempts SET lease_expires_at=? WHERE run_id=? AND attempt=?",
                (expires, run_id, row["attempt"]),
            )
            self._event(connection, row, "heartbeat", now, lease_expires_at=expires)
            return True

    @staticmethod
    def _live(row: sqlite3.Row, token: int, now: float) -> bool:
        return (
            row["state"] == "running"
            and type(token) is int
            and row["token"] == token
            and row["lease_expires_at"] > now
        )

    def is_active(self, run_id: str, token: int) -> bool:
        with self._connect() as connection:
            return self._live(self._row(connection, run_id), token, time.time())

    def finish(
        self,
        run_id: str,
        token: int,
        state: str,
        result_hash: str | None = None,
        error: str | None = None,
    ) -> bool:
        if state not in _TERMINAL or state == "cancelled":
            raise ValueError("Finish requires a terminal state; use cancel for cancellation.")
        if state == "completed" and result_hash is None:
            raise ValueError("Completed runs require a result artifact reference.")
        if result_hash is not None:
            _validate_hash(result_hash)
        safe_error = _safe_error(error)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, run_id)
            now = time.time()
            if not self._live(row, token, now):
                return False
            connection.execute(
                "UPDATE runs SET state=?,result_hash=?,error=?,lease_expires_at=NULL,updated_at=? "
                "WHERE id=?",
                (state, result_hash, safe_error, now, run_id),
            )
            connection.execute(
                "UPDATE attempts SET state=?,finished_at=?,result_hash=?,error=? "
                "WHERE run_id=? AND attempt=?",
                (state, now, result_hash, safe_error, run_id, row["attempt"]),
            )
            self._event(connection, row, "finished", now, state=state, result_hash=result_hash)
            return True

    def cancel_if_active(
        self, run_id: str, token: int, *, diagnostic_hash: str | None = None
    ) -> bool:
        """Cancel only the caller's live attempt, optionally recording its receipt.

        The ownership check, fencing-token increment, attempt transition and
        optional diagnostic attachment commit together. A late worker cannot
        cancel a successor or change a terminal run. Diagnostic events retain
        the original attempt token; cancellation events carry the new fence.
        """
        if diagnostic_hash is not None:
            _validate_hash(diagnostic_hash)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, run_id)
            now = time.time()
            if not self._live(row, token, now):
                return False
            connection.execute(
                "UPDATE attempts SET state='cancelled',finished_at=? "
                "WHERE run_id=? AND attempt=? AND state='running'",
                (now, run_id, row["attempt"]),
            )
            connection.execute(
                "UPDATE runs SET state='cancelled',token=token+1,lease_expires_at=NULL,"
                "updated_at=? WHERE id=?",
                (now, run_id),
            )
            self._event(connection, self._row(connection, run_id), "cancelled", now)
            if diagnostic_hash is not None:
                existing = connection.execute(
                    "SELECT details FROM events "
                    "WHERE run_id=? AND token=? AND kind='attempt_diagnostic'",
                    (run_id, token),
                )
                if not any(
                    json.loads(item["details"])["artifact_sha256"] == diagnostic_hash
                    for item in existing
                ):
                    self._event(
                        connection,
                        row,
                        "attempt_diagnostic",
                        now,
                        artifact_sha256=diagnostic_hash,
                    )
            return True

    def cancel(self, run_id: str) -> dict:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, run_id)
            if row["state"] in _TERMINAL:
                return self._render(connection, row)
            now = time.time()
            if row["state"] == "running":
                connection.execute(
                    "UPDATE attempts SET state='cancelled',finished_at=? "
                    "WHERE run_id=? AND attempt=? AND state='running'",
                    (now, run_id, row["attempt"]),
                )
            connection.execute(
                "UPDATE runs SET state='cancelled',token=token+1,lease_expires_at=NULL,"
                "updated_at=? WHERE id=?",
                (now, run_id),
            )
            row = self._row(connection, run_id)
            self._event(connection, row, "cancelled", now)
            return self._render(connection, row)

    def events(self, run_id: str) -> list[dict]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._row(connection, run_id)
            events = []
            for row in connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)
            ):
                event = dict(row)
                event["details"] = json.loads(event["details"])
                events.append(event)
            return events

    def record_call(self, run_id: str, token: int, call: dict) -> bool:
        """Persist one call transition before/after dispatch, fenced by the live lease.

        Records contain bounded metadata and artifact hashes, never raw prompts,
        response text, HTTP headers, or credentials. The ledger survives process
        failure and cancellation independently of result publication.
        """
        metadata = _call_metadata(call)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, run_id)
            now = time.time()
            if not self._live(row, token, now):
                return False
            self._event(connection, row, "provider_call", now, call=metadata)
            return True

    def calls(self, run_id: str) -> list[dict]:
        """Return ordered, durable call transitions, including cancelled attempts."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._row(connection, run_id)
            return [
                json.loads(row["details"])["call"]
                for row in connection.execute(
                    "SELECT details FROM events WHERE run_id=? AND kind='provider_call' ORDER BY id",
                    (run_id,),
                )
            ]

    def attach_diagnostic(self, run_id: str, token: int, artifact_sha256: str) -> bool:
        """Attach immutable diagnostics to a known attempt, including after fencing.

        This append-only audit operation cannot publish a scientific result or
        change execution ownership/state. An old worker may report its cleanup
        failure after cancellation while remaining unable to overwrite a newer
        attempt. Repeated identical attachments are idempotent.
        """
        _validate_hash(artifact_sha256)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._row(connection, run_id)
            if type(token) is not int:
                return False
            attempt = connection.execute(
                "SELECT attempt FROM attempts WHERE run_id=? AND token=?", (run_id, token)
            ).fetchone()
            if attempt is None:
                return False
            existing = connection.execute(
                "SELECT details FROM events "
                "WHERE run_id=? AND token=? AND kind='attempt_diagnostic'",
                (run_id, token),
            )
            if any(
                json.loads(row["details"])["artifact_sha256"] == artifact_sha256 for row in existing
            ):
                return True
            connection.execute(
                "INSERT INTO events(run_id,at,kind,token,attempt,details) VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    time.time(),
                    "attempt_diagnostic",
                    token,
                    attempt["attempt"],
                    json.dumps({"artifact_sha256": artifact_sha256}, sort_keys=True),
                ),
            )
            return True

    def diagnostics(self, run_id: str) -> list[dict]:
        """Return diagnostic references without granting them result authority."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._row(connection, run_id)
            return [
                {
                    "attempt": row["attempt"],
                    "token": row["token"],
                    "artifact_sha256": json.loads(row["details"])["artifact_sha256"],
                }
                for row in connection.execute(
                    "SELECT attempt,token,details FROM events "
                    "WHERE run_id=? AND kind='attempt_diagnostic' ORDER BY id",
                    (run_id,),
                )
            ]

    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            return [
                self._render(connection, row)
                for row in connection.execute("SELECT * FROM runs ORDER BY created_at,id")
            ]
