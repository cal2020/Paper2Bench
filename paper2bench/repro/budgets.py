"""Transactional call/token quotas shared across experiments.

Counters include both outstanding reservations and settled usage. They are not
currency charges. Reservations never expire implicitly: a process crash or an
ambiguous request retains its full allocation until explicitly reconciled.

Ledger idempotence does not make a provider request idempotent. A caller must
not dispatch again simply because an existing reservation can be read again.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .providers import Generation, ProviderClient, ProviderProfile


class BudgetExceeded(RuntimeError):
    """A new reservation would exceed quota, or the budget is blocked."""


class BudgetConflict(RuntimeError):
    """An identifier was reused with conflicting limits or reconciliation."""


class BudgetNotFound(LookupError):
    """The requested budget or reservation does not exist."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
# At most 1e6 non-rejected reservations can consume calls, and each reported
# output count is at most 1e12. Even worst-case overages therefore fit SQLite's
# signed 64-bit integer counters without silent promotion to floating point.
_MAX_COUNTER = 10**12
_MAX_CALLS = 10**6


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier of 1..128 characters")
    return value


def _counter(value: int, field: str, *, positive: bool = False, maximum: int = _MAX_COUNTER) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


class BudgetStore:
    """SQLite quota ledger with serialized reservations and durable reconciliation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("budget ledger requires a durable filesystem path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS p2b_budgets (
                    budget_id TEXT PRIMARY KEY,
                    max_calls INTEGER NOT NULL,
                    max_output_tokens INTEGER NOT NULL,
                    used_calls INTEGER NOT NULL DEFAULT 0 CHECK(used_calls >= 0),
                    used_output_tokens INTEGER NOT NULL DEFAULT 0 CHECK(used_output_tokens >= 0),
                    blocked INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS p2b_budget_reservations (
                    budget_id TEXT NOT NULL REFERENCES p2b_budgets(budget_id),
                    reservation_id TEXT NOT NULL,
                    calls INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    run_id TEXT,
                    state TEXT NOT NULL,
                    actual_output_tokens INTEGER,
                    accounted_calls INTEGER NOT NULL,
                    accounted_output_tokens INTEGER NOT NULL,
                    overage INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(budget_id, reservation_id)
                );
                CREATE TABLE IF NOT EXISTS p2b_budget_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    budget_id TEXT NOT NULL REFERENCES p2b_budgets(budget_id),
                    reservation_id TEXT,
                    kind TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _budget(connection, budget_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM p2b_budgets WHERE budget_id=?", (budget_id,)
        ).fetchone()
        if row is None:
            raise BudgetNotFound("budget not found")
        return row

    @staticmethod
    def _reservation(connection, budget_id: str, reservation_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM p2b_budget_reservations WHERE budget_id=? AND reservation_id=?",
            (budget_id, reservation_id),
        ).fetchone()
        if row is None:
            raise BudgetNotFound("reservation not found")
        return row

    @staticmethod
    def _budget_dict(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["blocked"] = bool(result["blocked"])
        result["remaining_calls"] = max(0, result["max_calls"] - result["used_calls"])
        result["remaining_output_tokens"] = max(
            0, result["max_output_tokens"] - result["used_output_tokens"]
        )
        result["accounting"] = "outstanding_reservations_plus_settled_usage"
        return result

    @staticmethod
    def _event(connection, budget_id, reservation_id, kind, data, timestamp):
        connection.execute(
            """INSERT INTO p2b_budget_events
               (budget_id,reservation_id,kind,data,created_at) VALUES(?,?,?,?,?)""",
            (budget_id, reservation_id, kind, json.dumps(data, sort_keys=True), timestamp),
        )

    def create(self, budget_id: str, max_calls: int, max_output_tokens: int) -> dict:
        _identifier(budget_id, "budget_id")
        _counter(max_calls, "max_calls", maximum=_MAX_CALLS)
        _counter(max_output_tokens, "max_output_tokens")
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM p2b_budgets WHERE budget_id=?", (budget_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["max_calls"] != max_calls
                    or existing["max_output_tokens"] != max_output_tokens
                ):
                    raise BudgetConflict("budget identifier already has different limits")
                return self._budget_dict(existing)
            timestamp = time.time()
            connection.execute(
                """INSERT INTO p2b_budgets
                   (budget_id,max_calls,max_output_tokens,created_at,updated_at)
                   VALUES(?,?,?,?,?)""",
                (budget_id, max_calls, max_output_tokens, timestamp, timestamp),
            )
            self._event(
                connection,
                budget_id,
                None,
                "created",
                {"max_calls": max_calls, "max_output_tokens": max_output_tokens},
                timestamp,
            )
            return self._budget_dict(self._budget(connection, budget_id))

    def get(self, budget_id: str) -> dict:
        _identifier(budget_id, "budget_id")
        with self._connection() as connection:
            return self._budget_dict(self._budget(connection, budget_id))

    def reserve(
        self,
        budget_id: str,
        reservation_id: str,
        *,
        calls: int = 1,
        output_tokens: int,
        run_id: str | None = None,
    ) -> dict:
        _identifier(budget_id, "budget_id")
        _identifier(reservation_id, "reservation_id")
        if run_id is not None:
            _identifier(run_id, "run_id")
        _counter(calls, "calls", positive=True, maximum=_MAX_CALLS)
        _counter(output_tokens, "output_tokens")
        with self._transaction() as connection:
            budget = self._budget(connection, budget_id)
            existing = connection.execute(
                "SELECT * FROM p2b_budget_reservations WHERE budget_id=? AND reservation_id=?",
                (budget_id, reservation_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["calls"] != calls
                    or existing["output_tokens"] != output_tokens
                    or existing["run_id"] != run_id
                ):
                    raise BudgetConflict("reservation identifier already has different parameters")
                return dict(existing)
            if budget["blocked"]:
                raise BudgetExceeded("budget is blocked after provider-reported usage overage")
            if (
                budget["used_calls"] + calls > budget["max_calls"]
                or budget["used_output_tokens"] + output_tokens > budget["max_output_tokens"]
            ):
                raise BudgetExceeded("reservation would exceed call or output-token quota")
            timestamp = time.time()
            connection.execute(
                """INSERT INTO p2b_budget_reservations
                   (budget_id,reservation_id,calls,output_tokens,run_id,state,
                    accounted_calls,accounted_output_tokens,created_at,updated_at)
                   VALUES(?,?,?,?,?,'reserved',?,?,?,?)""",
                (
                    budget_id,
                    reservation_id,
                    calls,
                    output_tokens,
                    run_id,
                    calls,
                    output_tokens,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """UPDATE p2b_budgets SET used_calls=used_calls+?,
                   used_output_tokens=used_output_tokens+?,updated_at=? WHERE budget_id=?""",
                (calls, output_tokens, timestamp, budget_id),
            )
            self._event(
                connection,
                budget_id,
                reservation_id,
                "reserved",
                {"calls": calls, "output_tokens": output_tokens, "run_id": run_id},
                timestamp,
            )
            return dict(self._reservation(connection, budget_id, reservation_id))

    def reconcile(
        self,
        budget_id: str,
        reservation_id: str,
        *,
        actual_output_tokens: int | None,
        outcome: Literal["completed", "rejected", "unknown"],
    ) -> dict:
        _identifier(budget_id, "budget_id")
        _identifier(reservation_id, "reservation_id")
        if actual_output_tokens is not None:
            _counter(actual_output_tokens, "actual_output_tokens")
        if not isinstance(outcome, str) or outcome not in {"completed", "rejected", "unknown"}:
            raise ValueError("outcome must be completed, rejected, or unknown")
        if outcome == "unknown" and actual_output_tokens is not None:
            raise ValueError("unknown outcomes must retain an unknown actual token count")
        if outcome == "rejected" and actual_output_tokens not in (None, 0):
            raise ValueError("definitely-not-dispatched rejection cannot report token usage")
        actual = 0 if outcome == "rejected" else actual_output_tokens
        with self._transaction() as connection:
            budget = self._budget(connection, budget_id)
            reservation = self._reservation(connection, budget_id, reservation_id)
            old_state = reservation["state"]
            if old_state == outcome and reservation["actual_output_tokens"] == actual:
                return dict(reservation)
            # Unknown completion usage can be refined once. A known completion
            # and a definite rejection are terminal, so repeated/refused calls
            # cannot produce double refunds or rewrite provider usage.
            if old_state == "rejected":
                raise BudgetConflict("rejected reservation cannot be reconciled differently")
            if old_state == "completed" and not (
                outcome == "completed"
                and reservation["actual_output_tokens"] is None
                and actual is not None
            ):
                raise BudgetConflict("completed reservation cannot be reconciled differently")
            calls = 0 if outcome == "rejected" else reservation["calls"]
            tokens = (
                0
                if outcome == "rejected"
                else reservation["output_tokens"]
                if actual is None
                else actual
            )
            overage = max(0, tokens - reservation["output_tokens"])
            timestamp = time.time()
            connection.execute(
                """UPDATE p2b_budget_reservations SET state=?,actual_output_tokens=?,
                   accounted_calls=?,accounted_output_tokens=?,overage=?,updated_at=?
                   WHERE budget_id=? AND reservation_id=?""",
                (outcome, actual, calls, tokens, overage, timestamp, budget_id, reservation_id),
            )
            connection.execute(
                """UPDATE p2b_budgets SET used_calls=used_calls+?,
                   used_output_tokens=used_output_tokens+?,blocked=?,updated_at=?
                   WHERE budget_id=?""",
                (
                    calls - reservation["accounted_calls"],
                    tokens - reservation["accounted_output_tokens"],
                    int(bool(budget["blocked"] or overage)),
                    timestamp,
                    budget_id,
                ),
            )
            self._event(
                connection,
                budget_id,
                reservation_id,
                "reconciled",
                {
                    "outcome": outcome,
                    "actual_output_tokens": actual,
                    "accounted_calls": calls,
                    "accounted_output_tokens": tokens,
                    "overage": overage,
                },
                timestamp,
            )
            return dict(self._reservation(connection, budget_id, reservation_id))

    def events(self, budget_id: str) -> list[dict]:
        _identifier(budget_id, "budget_id")
        with self._connection() as connection:
            self._budget(connection, budget_id)
            rows = connection.execute(
                "SELECT * FROM p2b_budget_events WHERE budget_id=? ORDER BY event_id", (budget_id,)
            ).fetchall()
            return [{**dict(row), "data": json.loads(row["data"])} for row in rows]


class BudgetedProviderClient:
    """Synchronous provider wrapper with conservative, cross-run reservations.

    Every generation uses a fresh reservation and forwards to the underlying
    client exactly once. Errors, interruptions, and unknown usage never produce
    an implicit refund. This controls requested calls/output tokens, not money
    or a provider's actual compliance with its requested generation limit.
    """

    def __init__(
        self,
        client: ProviderClient,
        ledger: BudgetStore,
        budget_id: str,
        run_id: str | None = None,
    ):
        _identifier(budget_id, "budget_id")
        if run_id is not None:
            _identifier(run_id, "run_id")
        self.client = client
        self.ledger = ledger
        self.budget_id = budget_id
        self.run_id = run_id
        self.last_reservation_id: str | None = None

    @property
    def profile(self) -> ProviderProfile:
        return self.client.profile

    def generate(self, **kwargs) -> Generation:
        requested = _counter(kwargs.get("max_output_tokens"), "max_output_tokens", positive=True)
        reservation_id = uuid.uuid4().hex
        self.ledger.reserve(
            self.budget_id,
            reservation_id,
            output_tokens=requested,
            run_id=self.run_id,
        )
        self.last_reservation_id = reservation_id
        try:
            generation = self.client.generate(**kwargs)
        except BaseException as error:
            # KeyboardInterrupt/SystemExit can arrive after provider acceptance,
            # just as a network timeout can. Preserve the allocation even if
            # recording the unknown state itself fails; reserve is durable.
            try:
                self.ledger.reconcile(
                    self.budget_id,
                    reservation_id,
                    actual_output_tokens=None,
                    outcome="unknown",
                )
            except BaseException:
                error.add_note(
                    "Quota reservation remains allocated; usage reconciliation is required."
                )
            raise
        settled = self.ledger.reconcile(
            self.budget_id,
            reservation_id,
            actual_output_tokens=generation.usage.output_tokens,
            outcome="completed",
        )
        if settled["overage"]:
            raise BudgetExceeded(
                "provider reported output usage above its reservation; budget blocked"
            )
        return generation
