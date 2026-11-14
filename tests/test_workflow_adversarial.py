"""Adversarial contract tests for staging, budgets and bounded JSON parsing."""

import json
from pathlib import Path

import pytest

from paper2bench.bundles import BundleError, atomic_json, digest, read_json
from paper2bench.execution_policy import admit_limits
from paper2bench.executors import ExecutionResult
from paper2bench.executors import ExecutionPolicyError
from paper2bench.journal import JournalError, RunJournal
from paper2bench.runtime_config import runtime_scope
from paper2bench import workflow


class RecordingExecutor:
    def __init__(self, status="succeeded", exit_code=0):
        self.status = status
        self.exit_code = exit_code
        self.requests = []

    def preflight(self):
        return {"backend": "fake", "image_id": "sha256:" + "a" * 64}

    def execute(self, request):
        self.requests.append(request)
        return ExecutionResult(self.status, self.exit_code, "", "", 0.01,
                               backend="fake", isolation="test_double",
                               evidence={"cleanup_confirmed": True})

    def cancel(self):
        return True


def make_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "main.py").write_text("print('pinned source')\n")
    atomic_json(bundle / "inputs.json", {"test": [1, 2]})
    manifest = {
        "schema_version": 1, "task_id": "adversarial", "revision": 1,
        "task_mode": "from_scratch", "readiness": "ready",
        "entrypoint": "main.py", "inputs": "inputs.json", "predictions": "predictions.json",
        "files": {name: digest((bundle / name).read_bytes()) for name in ("main.py", "inputs.json")},
    }
    atomic_json(bundle / "task_manifest.json", manifest)
    return bundle / "task_manifest.json"


def test_executed_code_comes_from_verified_staged_submission(tmp_path, monkeypatch):
    """Simulate a file changing during a pre-stage read, then being restored.

    No hostile code executes: the injected executor only records the request.
    Once staging hashes pass, controller code must read that staged snapshot.
    """
    manifest_path = make_bundle(tmp_path)
    original_safe_file = workflow.safe_file

    class TransientRead:
        def __init__(self, actual):
            self.actual = actual

        def read_text(self, *args, **kwargs):
            return "print('unhashed transient code')\n"

        def __getattr__(self, name):
            return getattr(self.actual, name)

    def race(root, name):
        actual = original_safe_file(root, name)
        if Path(root) == manifest_path.parent and name == "main.py":
            return TransientRead(actual)
        return actual

    monkeypatch.setattr(workflow, "safe_file", race)
    executor = RecordingExecutor()
    report = workflow.execute(manifest_path, executor=executor, runs_dir=tmp_path / "runs")
    assert report["pass"] is True
    assert executor.requests[0].code == "print('pinned source')\n"


def test_known_failed_attempt_can_resume_with_remaining_time_budget(tmp_path):
    manifest_path = make_bundle(tmp_path)
    runs = tmp_path / "runs"
    first = workflow.execute(manifest_path, executor=RecordingExecutor("failed", 1), runs_dir=runs)
    assert first["run_state"] == "failed"
    next_executor = RecordingExecutor()
    report = workflow.resume(first["run_id"], executor=next_executor, runs_dir=runs)
    assert report["run_state"] == "succeeded"
    assert len(next_executor.requests) == 1
    assert next_executor.requests[0].timeout_seconds <= 300


def resource_profile():
    return {
        "schema_version": 1,
        "models": {"local": {"provider": "ollama", "model": "operator-model"}},
        "roles": {"agent": "local"},
        "budgets": {"experiments": {"wall_seconds": 1, "memory_bytes": 67108864,
                                       "disk_bytes": 16777216, "cpu_seconds": 1}},
    }


@pytest.mark.parametrize("requested", [
    {"timeout_seconds": 2, "memory_bytes": 67108864, "workspace_bytes": 16777216},
    {"timeout_seconds": 1, "memory_bytes": 134217728, "workspace_bytes": 16777216},
])
def test_operator_experiment_resource_budgets_are_admission_limits(requested):
    with runtime_scope(resource_profile()):
        with pytest.raises(ValueError):
            admit_limits(requested)


def test_bundle_json_cannot_hide_float_overflow_as_a_finite_value(tmp_path):
    path = tmp_path / "overflow.json"
    path.write_text('{"value": 1e999}')
    with pytest.raises(BundleError):
        read_json(path)


@pytest.mark.parametrize("limits", [
    {"memory_bytes": 1024},
    {"pids_limit": 2},
    {"workspace_bytes": 32 * 1024 * 1024, "memory_bytes": 16 * 1024 * 1024},
])
def test_invalid_backend_request_is_blocked_before_reservation(tmp_path, monkeypatch, limits):
    manifest_path = make_bundle(tmp_path)
    manifest = read_json(manifest_path)
    manifest["limits"] = limits
    atomic_json(manifest_path, manifest)
    run_ids = []
    original_create = RunJournal.create

    def record_create(self, *args, **kwargs):
        run_id = original_create(self, *args, **kwargs)
        run_ids.append(run_id)
        return run_id

    monkeypatch.setattr(RunJournal, "create", record_create)
    executor = RecordingExecutor()
    runs = tmp_path / "runs"
    with pytest.raises(ExecutionPolicyError):
        workflow.execute(manifest_path, executor=executor, runs_dir=runs)
    run = RunJournal(runs).get(run_ids[0])
    assert run["state"] == "blocked"
    assert run["budget"]["reserved"] == {}
    assert executor.requests == []


@pytest.mark.parametrize("worker_status, exit_code", [("succeeded", 0), ("failed", 1)])
def test_cancellation_during_execution_is_terminal(tmp_path, worker_status, exit_code):
    manifest_path = make_bundle(tmp_path)
    runs = tmp_path / "runs"

    class CancelledWorker(RecordingExecutor):
        def execute(self, request):
            journal = RunJournal(runs)
            # The private workspace belongs to this exact run and attempt.
            run_id = request.workspace.parents[2].name
            journal.request_cancel(run_id)
            return super().execute(request)

    report = workflow.execute(manifest_path, executor=CancelledWorker(worker_status, exit_code), runs_dir=runs)
    assert report["run_state"] == "cancelled"
    assert report["evaluation_status"] == "cancelled"
    assert report["pass"] is False
    committed = RunJournal(runs).get(report["run_id"])["attempts"][0]["result"]
    assert committed["report_data"] == report
    with pytest.raises(JournalError):
        workflow.resume(report["run_id"], executor=RecordingExecutor(), runs_dir=runs)


@pytest.mark.parametrize("filename", ["main.py", "inputs.json"])
def test_completed_resume_rejects_changed_pinned_payload(tmp_path, filename):
    manifest_path = make_bundle(tmp_path)
    runs = tmp_path / "runs"
    executor = RecordingExecutor()
    report = workflow.execute(manifest_path, executor=executor, runs_dir=runs)
    (manifest_path.parent / filename).write_text("changed pinned input")
    with pytest.raises(JournalError, match="Pinned task files changed"):
        workflow.resume(report["run_id"], executor=executor, runs_dir=runs)
    assert len(executor.requests) == 1
