"""Transactional local run state, submission intents and resource reservations.

This journal coordinates one machine. It does not claim exactly-once delivery
to external providers: an interrupted running attempt is deliberately unknown.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class JournalError(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RunJournal:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "runs.sqlite3"
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, created REAL NOT NULL,
                    updated REAL NOT NULL, spec TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, state TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL,
                    request_key TEXT UNIQUE NOT NULL, result TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    at REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS budgets (
                    run_id TEXT PRIMARY KEY, limits TEXT NOT NULL, used TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reservations (
                    key TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    amounts TEXT NOT NULL, state TEXT NOT NULL, actual TEXT);
            """)
        self.path.chmod(0o600)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            if db.in_transaction:
                db.commit()
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _event(db, run_id, kind, payload):
        db.execute("INSERT INTO events(run_id,at,kind,payload) VALUES (?,?,?,?)",
                   (run_id, time.time(), kind, _json(payload)))

    def create(self, spec, *, limits=None, parent_run=None):
        run_id = uuid.uuid4().hex
        spec = dict(spec, parent_run=parent_run)
        limits = self._amounts(limits or {})
        now = time.time()
        with self._transaction() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                       (run_id, "pending", now, now, _json(spec)))
            db.execute("INSERT INTO budgets VALUES (?,?,?)", (run_id, _json(limits), "{}"))
            self._event(db, run_id, "created", {"parent_run": parent_run})
        return run_id

    def get(self, run_id):
        with self._transaction() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise JournalError("Unknown run ID")
            result = dict(row)
            result["spec"] = json.loads(result["spec"])
            result["attempts"] = []
            for a in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY created", (run_id,)):
                item = dict(a)
                item["result"] = json.loads(item["result"]) if item["result"] else None
                result["attempts"].append(item)
            b = db.execute("SELECT * FROM budgets WHERE run_id=?", (run_id,)).fetchone()
            result["budget"] = {"limits": json.loads(b["limits"]), "used": json.loads(b["used"]),
                                "reserved": self._reserved(db, run_id)}
            return result

    def event(self, run_id, kind, payload):
        with self._transaction() as db:
            self._event(db, run_id, kind, payload)

    def events(self, run_id):
        with self._transaction() as db:
            return [dict(r, payload=json.loads(r["payload"])) for r in
                    db.execute("SELECT * FROM events WHERE run_id=? ORDER BY sequence", (run_id,))]

    def children(self, parent_run):
        """Find durable child identities, including interrupted submission intents."""
        with self._transaction() as db:
            ids = [row["id"] for row in db.execute("SELECT id,spec FROM runs ORDER BY created")
                   if json.loads(row["spec"]).get("parent_run") == parent_run]
            if len(ids) > 10000:
                raise JournalError("Child run count exceeds supported orchestration bound")
        return [self.get(run_id) for run_id in ids]

    def recover_orchestration(self, run_id):
        """Only a scientific controller with no active children can restart its loop."""
        with self._transaction() as db:
            row = db.execute("SELECT state,spec FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or json.loads(row["spec"]).get("kind") != "scientific":
                raise JournalError("Only scientific orchestration can use this recovery path")
            if row["state"] in {"succeeded", "cancelled", "cancellation_requested"}:
                raise JournalError("Scientific run is completed or cancelled")
            for child in db.execute("SELECT id,state,spec FROM runs"):
                child_spec = json.loads(child["spec"])
                if child_spec.get("parent_run") == run_id and child_spec.get("kind") == "execution" and child["state"] in {
                        "pending", "running", "reconciling", "cancellation_requested"}:
                    if child["state"] == "pending" and not db.execute("SELECT 1 FROM attempts WHERE run_id=?", (child["id"],)).fetchone():
                        continue
                    raise JournalError("A child outcome must be reconciled before orchestration resumes")
            db.execute("UPDATE attempts SET state='blocked',updated=? WHERE run_id=? AND state IN ('submission_intent','running','reconciling')",
                       (time.time(), run_id))
            db.execute("UPDATE runs SET state='blocked',updated=? WHERE id=?", (time.time(), run_id))
            self._event(db, run_id, "orchestration_recovery", {})

    def recover_authoring(self, run_id):
        """Recover an interrupted, fully settled authoring controller.

        The generation controller must hold its output lock and validate that
        no checkpoint stage has a running or unknown external outcome before
        using this path. The transaction additionally rejects unsettled usage,
        execution runs, completed work and cancellation.
        """
        with self._transaction() as db:
            row = db.execute("SELECT state,spec FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or json.loads(row["spec"]).get("kind") != "authoring":
                raise JournalError("Only authoring runs can use checkpoint recovery")
            if row["state"] not in {"running", "reconciling"}:
                raise JournalError("Authoring run is not awaiting checkpoint recovery")
            if any(self._reserved(db, run_id).values()):
                raise JournalError("Authoring usage must be reconciled before checkpoint recovery")
            active = [item["id"] for item in db.execute(
                "SELECT id FROM attempts WHERE run_id=? AND state IN ('submission_intent','running')",
                (run_id,))]
            now = time.time()
            db.execute("UPDATE attempts SET state='failed',updated=?,result=? "
                       "WHERE run_id=? AND state IN ('submission_intent','running')",
                       (now, _json({"reason": "authoring_checkpoint_recovered"}), run_id))
            db.execute("UPDATE runs SET state='failed',updated=? WHERE id=?", (now, run_id))
            self._event(db, run_id, "authoring_recovery", {
                "previous_state": row["state"], "attempt_ids": active,
                "usage_status": "fully_settled", "checkpoint_verified_by_controller": True})
        return "failed"

    def begin_attempt(self, run_id):
        """Persist submission intent before any side effect. Never retry an unknown attempt."""
        attempt_id = uuid.uuid4().hex
        now = time.time()
        with self._transaction() as db:
            row = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or row[0] not in {"pending", "failed", "blocked"}:
                raise JournalError("Run is complete, active, cancelled, or requires reconciliation")
            db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?)",
                       (attempt_id, run_id, "submission_intent", now, now, attempt_id, None))
            db.execute("UPDATE runs SET state='running',updated=? WHERE id=?", (now, run_id))
            self._event(db, run_id, "submission_intent", {"attempt_id": attempt_id,
                                                        "idempotency_key": attempt_id})
        return attempt_id

    def finish(self, run_id, attempt_id, state, result):
        """Commit the final state and result together.

        A trusted result factory may receive the resolved state so evidence
        remains consistent if cancellation wins the finalization race.
        """
        if state not in {"succeeded", "failed", "blocked", "cancelled", "reconciling"}:
            raise JournalError("Invalid terminal state")
        now = time.time()
        with self._transaction() as db:
            row = db.execute("SELECT state FROM attempts WHERE id=? AND run_id=?",
                             (attempt_id, run_id)).fetchone()
            if not row or row[0] not in {"submission_intent", "running"}:
                raise JournalError("Attempt already finalized or unknown")
            # A confirmed failure or preparation block must not erase a user's
            # cancellation and make the same run retryable. Ambiguous work still
            # needs reconciliation before termination can be asserted.
            if self._cancelled(db, run_id) and state != "reconciling":
                state = "cancelled"
            if callable(result):
                result = result(state)
            db.execute("UPDATE attempts SET state=?,updated=?,result=? WHERE id=?",
                       (state, now, _json(result), attempt_id))
            db.execute("UPDATE runs SET state=?,updated=? WHERE id=?", (state, now, run_id))
            self._event(db, run_id, state, {"attempt_id": attempt_id})
        return state

    @staticmethod
    def _cancelled(db, run_id):
        row = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
        return row is not None and row[0] in {"cancellation_requested", "cancelled"}

    def request_cancel(self, run_id):
        with self._transaction() as db:
            row = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise JournalError("Unknown run ID")
            if row[0] in {"succeeded", "failed", "blocked", "cancelled"}:
                return row[0]
            state = "cancelled" if row[0] == "pending" else "cancellation_requested"
            db.execute("UPDATE runs SET state=?,updated=? WHERE id=?", (state, time.time(), run_id))
            self._event(db, run_id, state, {})
            return state

    def confirm_orphan_termination(self, run_id, attempt_id, evidence):
        """Confirm termination, preserving unknown resource reservations for review."""
        with self._transaction() as db:
            row = db.execute("SELECT state FROM attempts WHERE id=? AND run_id=?", (attempt_id, run_id)).fetchone()
            if not row or row[0] not in {"submission_intent", "running", "reconciling"}:
                raise JournalError("Attempt is not awaiting termination reconciliation")
            db.execute("UPDATE attempts SET state='cancelled',updated=?,result=? WHERE id=?",
                       (time.time(), _json(evidence), attempt_id))
            db.execute("UPDATE runs SET state='cancelled',updated=? WHERE id=?", (time.time(), run_id))
            self._event(db, run_id, "orphan_termination_confirmed", {"attempt_id": attempt_id, **evidence})

    @staticmethod
    def _amounts(amounts):
        result = {}
        for key, value in amounts.items():
            if (not isinstance(key, str) or type(value) not in {int, float}
                    or not math.isfinite(value) or value < 0):
                raise JournalError("Budget values must be finite nonnegative numbers")
            result[key] = value
        return result

    @staticmethod
    def _reserved(db, run_id):
        reserved = {}
        for r in db.execute("SELECT amounts FROM reservations WHERE run_id=? AND state='reserved'", (run_id,)):
            for key, value in json.loads(r[0]).items():
                reserved[key] = reserved.get(key, 0) + value
        return reserved

    def reserve(self, run_id, key, amounts):
        amounts = self._amounts(amounts)
        with self._transaction() as db:
            existing = db.execute("SELECT * FROM reservations WHERE key=?", (key,)).fetchone()
            if existing:
                if existing["run_id"] != run_id or json.loads(existing["amounts"]) != amounts:
                    raise JournalError("Idempotency key reused for different reservation")
                return
            budget = db.execute("SELECT * FROM budgets WHERE run_id=?", (run_id,)).fetchone()
            if budget is None:
                raise JournalError("Unknown run")
            state = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()[0]
            if state not in {"pending", "running"}:
                raise JournalError("Run no longer accepts new budget reservations")
            limits, used = json.loads(budget["limits"]), json.loads(budget["used"])
            reserved = self._reserved(db, run_id)
            for dimension, value in amounts.items():
                if dimension not in limits or used.get(dimension, 0) + reserved.get(dimension, 0) + value > limits[dimension]:
                    raise JournalError(f"Budget exhausted or unspecified: {dimension}")
            db.execute("INSERT INTO reservations VALUES (?,?,?,'reserved',NULL)",
                       (key, run_id, _json(amounts)))
            self._event(db, run_id, "budget_reserved", {"key": key, "amounts": amounts})

    def settle(self, run_id, key, actual):
        """Only call after a confirmed outcome; unknown work retains its reservation."""
        actual = self._amounts(actual)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM reservations WHERE key=? AND run_id=?", (key, run_id)).fetchone()
            if row is None:
                raise JournalError("Unknown reservation")
            if row["state"] == "settled":
                if json.loads(row["actual"]) != actual:
                    raise JournalError("Reservation already settled differently")
                return
            reserved = json.loads(row["amounts"])
            if set(actual) != set(reserved):
                raise JournalError("Every reserved dimension needs a known actual value before settlement")
            budget = db.execute("SELECT used FROM budgets WHERE run_id=?", (run_id,)).fetchone()
            used = json.loads(budget[0])
            for dimension, value in actual.items():
                used[dimension] = used.get(dimension, 0) + value
            db.execute("UPDATE budgets SET used=? WHERE run_id=?", (_json(used), run_id))
            db.execute("UPDATE reservations SET state='settled',actual=? WHERE key=?", (_json(actual), key))
            self._event(db, run_id, "budget_settled", {"key": key, "actual": actual,
                         "overrun": any(actual.get(k, 0) > v for k, v in reserved.items())})
