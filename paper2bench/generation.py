"""Versioned authoring checkpoints with explicit unknown-outcome handling.

Completed stage values are historical artifacts. They are never represented as a
fresh model trial. An interrupted/unsaved external call is not blindly repeated.
"""
from __future__ import annotations

import re
import time
import os
from contextlib import contextmanager
from pathlib import Path

from .bundles import BundleError, atomic_bytes, atomic_json, canonical_json, digest, read_json

MAX_STAGE_BYTES = 8 * 1024 * 1024


class GenerationCheckpointError(ValueError):
    pass


@contextmanager
def task_output_lock(path):
    """Prevent separate authoring runs contaminating the same task output."""
    import fcntl
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = path / ".generation.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GenerationCheckpointError("Another authoring run owns this task output") from error
        yield
    finally:
        os.close(fd)


def pipeline_identity():
    root = Path(__file__).parent
    identities = {path.relative_to(root).as_posix(): digest(path.read_bytes())
                  for path in sorted(root.rglob("*.py"))}
    for path in sorted((root / "templates").glob("*.txt")):
        identities["templates/" + path.name] = digest(path.read_bytes())
    return digest(canonical_json(identities))


def read_verification_data(root):
    """Freeze the exact bounded public inputs admitted to legacy verification."""
    from .bundles import read_artifact
    from .verify import MAX_DATA_BYTES, MAX_DATA_FILES, _RESTRICTED_NAMES
    root = Path(root)
    if root.is_symlink():
        raise GenerationCheckpointError("Verification data must not be a symlink")
    if not root.exists():
        return None
    if not root.is_dir():
        raise GenerationCheckpointError("Verification data must be a directory")
    files, total = {}, 0
    for index, path in enumerate(root.rglob("*")):
        if index >= MAX_DATA_FILES * 4:
            raise GenerationCheckpointError("Verification data directory count exceeds its bound")
        if path.is_symlink():
            raise GenerationCheckpointError("Verification data must not contain symlinks")
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") or part.lower() in _RESTRICTED_NAMES for part in relative.parts):
            raise GenerationCheckpointError("Restricted verification data path")
        if len(files) >= MAX_DATA_FILES:
            raise GenerationCheckpointError("Verification data file count exceeds its bound")
        raw = read_artifact(root, relative.as_posix(), limit=max(1, MAX_DATA_BYTES - total))
        total += len(raw)
        if total > MAX_DATA_BYTES:
            raise GenerationCheckpointError("Verification data exceeds its byte bound")
        files[relative.as_posix()] = raw
    return files


class GenerationCheckpoint:
    def __init__(self, root, run_id, plan, *, resume=False, max_attempts=3):
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise GenerationCheckpointError("Invalid generation run ID")
        self.root = Path(root).resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "generation.json"
        self.plan = plan
        self.plan_hash = digest(canonical_json(plan))
        self.resume = resume
        if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
            raise GenerationCheckpointError("Invalid authoring stage attempt limit")
        self.max_attempts = max_attempts
        self.cache_hits = []
        if self.path.exists():
            self.state = read_json(self.path)
            if (not isinstance(self.state, dict) or type(self.state.get("schema_version")) is not int
                    or self.state["schema_version"] != 1 or self.state.get("run_id") != run_id
                    or self.state.get("state") not in {"running", "failed", "reconciling", "succeeded"}
                    or not isinstance(self.state.get("stages"), dict)):
                raise GenerationCheckpointError("Invalid checkpoint schema or run identity")
            if self.state.get("plan_sha256") != self.plan_hash or self.state.get("plan") != plan:
                raise GenerationCheckpointError("Paper, prompts, model configuration, template or task identity changed")
            if not resume:
                raise GenerationCheckpointError("Existing generation requires explicit resume")
        else:
            if resume:
                raise GenerationCheckpointError("Generation checkpoint is missing")
            self.state = {"schema_version": 1, "run_id": run_id, "plan": plan,
                          "plan_sha256": self.plan_hash, "state": "running", "stages": {},
                          "created_at": time.time(), "cache_semantics": "historical_artifact_reuse"}
            self._save()

    def _save(self):
        atomic_json(self.path, self.state)

    def ensure_resumable(self):
        """Require durable outcomes for every dispatched stage before recovery."""
        for name, stage in self.state["stages"].items():
            if (not isinstance(name, str) or not re.fullmatch(r"[a-z_]{1,48}", name)
                    or not isinstance(stage, dict)
                    or stage.get("state") not in {"succeeded", "failed", "running", "unknown"}):
                raise GenerationCheckpointError("Invalid authoring stage checkpoint")
            if stage["state"] in {"running", "unknown"}:
                raise GenerationCheckpointError("Stage outcome is unknown; reconcile it before any paid replay")
            if stage["state"] == "succeeded":
                receipt = read_json(self.root / (name + ".json"), limit=MAX_STAGE_BYTES)
                if (not isinstance(receipt, dict) or "value" not in receipt
                        or digest(canonical_json(receipt)) != stage.get("sha256")):
                    raise GenerationCheckpointError("Completed stage integrity check failed")

    def run_stage(self, name, operation):
        if not re.fullmatch(r"[a-z_]{1,48}", name):
            raise GenerationCheckpointError("Invalid stage name")
        previous = self.state["stages"].get(name)
        receipt = self.root / (name + ".json")
        if previous and previous["state"] == "succeeded":
            value = read_json(receipt, limit=MAX_STAGE_BYTES)
            if digest(canonical_json(value)) != previous["sha256"]:
                raise GenerationCheckpointError("Completed stage integrity check failed")
            self.cache_hits.append(name)
            return value["value"]
        if previous and previous["state"] in {"running", "unknown"}:
            raise GenerationCheckpointError("Stage outcome is unknown; reconcile it before any paid replay")
        if previous and not self.resume:
            raise GenerationCheckpointError("Failed stages require explicit resume")
        if previous and previous.get("attempt", 0) >= self.max_attempts:
            raise GenerationCheckpointError("Authoring stage attempt budget exhausted")
        self.state["stages"][name] = {"state": "running", "started_at": time.time(),
                                       "attempt": previous.get("attempt", 0) + 1 if previous else 1}
        self._save()  # Dispatch intent is durable first.
        try:
            result = operation()
        except BaseException as error:
            known_validation = isinstance(error, ValueError) and not getattr(error, "outcome_unknown", False)
            self.state["stages"][name].update(state="failed" if known_validation else "unknown",
                                                error_type=type(error).__name__)
            self.state["state"] = "failed" if known_validation else "reconciling"
            self._save()
            raise
        record = {"value": result}
        raw = canonical_json(record)
        if len(raw) > MAX_STAGE_BYTES:
            self.state["stages"][name]["state"] = "unknown"
            self._save()
            raise GenerationCheckpointError("Stage output exceeded its receipt limit; no automatic replay")
        # If this write fails, the durable running intent remains unknown.
        atomic_bytes(receipt, raw)
        self.state["stages"][name].update(state="succeeded", sha256=digest(raw), finished_at=time.time())
        self._save()
        return result

    def complete(self, result):
        record = {"value": result}
        raw = canonical_json(record)
        if len(raw) > MAX_STAGE_BYTES:
            raise GenerationCheckpointError("Final generation result exceeds its bounded reader limit")
        atomic_bytes(self.root / "result.json", raw)
        artifacts = {}
        for key in ("tree_path", "config_path", "instruction_path", "instruction_gt_path"):
            if key in result:
                path = Path(result[key])
                if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_STAGE_BYTES:
                    raise GenerationCheckpointError("Final artifact exceeds supported bounds")
                artifacts[str(path.resolve())] = digest(path.read_bytes())
        if result.get("verify_metrics") is not None and "instruction_gt_path" in result:
            path = Path(result["instruction_gt_path"]).parent / "verify_results.json"
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_STAGE_BYTES:
                raise GenerationCheckpointError("Verification artifact is unavailable or exceeds bounds")
            artifacts[str(path.resolve())] = digest(path.read_bytes())
        extras = result.get("extras", {})
        candidate_manifest = extras.get("task_manifest_path")
        if candidate_manifest:
            from .bundles import validate_task, safe_file
            candidate = Path(candidate_manifest)
            manifest = validate_task(candidate)
            artifacts[str(candidate.resolve())] = digest(candidate.read_bytes())
            for name in manifest["files"]:
                path = safe_file(candidate.parent, name)
                artifacts[str(path.resolve())] = digest(path.read_bytes())
        source_snapshot = extras.get("source_snapshot")
        if source_snapshot:
            source = Path(source_snapshot)
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
                raise GenerationCheckpointError("Source snapshot artifact is unavailable")
            artifacts[str(source.resolve())] = digest(source.read_bytes())
        template_snapshot = extras.get("template_snapshot")
        if template_snapshot:
            from .bundles import _read_regular_file
            template = Path(template_snapshot)
            artifacts[str(template.resolve())] = digest(_read_regular_file(template, MAX_STAGE_BYTES))
        self.state.update(state="succeeded", result_sha256=digest(canonical_json(record)),
                          finished_at=time.time(), cache_hits=list(self.cache_hits), artifact_sha256=artifacts)
        self._save()

    def completed_result(self):
        if self.state["state"] != "succeeded":
            return None
        record = read_json(self.root / "result.json", limit=MAX_STAGE_BYTES)
        if digest(canonical_json(record)) != self.state["result_sha256"]:
            raise GenerationCheckpointError("Generation result integrity check failed")
        for name, expected in self.state.get("artifact_sha256", {}).items():
            path = Path(name)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024 or digest(path.read_bytes()) != expected:
                raise GenerationCheckpointError("Completed generation artifact changed or is unavailable")
        extras = record["value"].get("extras", {})
        if extras.get("verification_data_snapshot"):
            current = read_verification_data(extras["verification_data_snapshot"])
            if current is None or {name: digest(raw) for name, raw in current.items()} != extras["verification_data_sha256"]:
                raise GenerationCheckpointError("Completed verification data identity changed")
        return record["value"]
