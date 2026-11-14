"""Crash-boundary regressions found by independent integration review."""
import shutil
import sqlite3
from pathlib import Path

import pytest

from paper2bench import workflow
from paper2bench.bundles import BundleError, atomic_json, canonical_json, digest, export_bundle, read_json
from paper2bench.executors import ExecutionResult
from paper2bench.journal import RunJournal


class Worker:
    def __init__(self):
        self.calls = 0

    def preflight(self):
        return {"image_id": "sha256:" + "a" * 64}

    def execute(self, request):
        self.calls += 1
        return ExecutionResult("succeeded", 0, "", "", 0.01,
                               evidence={"cleanup_confirmed": True})

    def cancel(self):
        return False


def bundle(tmp_path, input_text="{}"):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "main.py").write_text("print('test')")
    (root / "input.json").write_text(input_text)
    data = {"schema_version": 1, "task_id": "review", "revision": 1,
            "readiness": "ready", "task_mode": "from_scratch", "entrypoint": "main.py",
            "inputs": "input.json", "predictions": "predictions.json",
            "files": {name: digest((root / name).read_bytes()) for name in ("main.py", "input.json")}}
    atomic_json(root / "task_manifest.json", data)
    return root / "task_manifest.json"


def only_run(runs_dir):
    journal = RunJournal(runs_dir)
    with sqlite3.connect(journal.path) as db:
        rows = db.execute("SELECT id FROM runs").fetchall()
    assert len(rows) == 1
    return journal.get(rows[0][0])


def test_invalid_staged_input_does_not_strand_an_active_attempt(tmp_path):
    task = bundle(tmp_path, input_text="not valid json")
    worker = Worker()
    runs = tmp_path / "runs"
    with pytest.raises(BundleError):
        workflow.execute(task, executor=worker, runs_dir=runs)
    assert worker.calls == 0
    run = only_run(runs)
    assert run["state"] in {"blocked", "failed"}
    assert not run["budget"]["reserved"]


def test_finished_attempt_recovers_missing_convenience_report_without_reexecution(tmp_path, monkeypatch):
    task = bundle(tmp_path)
    runs = tmp_path / "runs"
    worker = Worker()
    original = workflow.atomic_json

    def crash(path, data):
        if path.name == "evaluation_report.json" and path.parent.parent == runs:
            raise OSError("simulated summary-file write failure")
        return original(path, data)

    monkeypatch.setattr(workflow, "atomic_json", crash)
    with pytest.raises(OSError):
        workflow.execute(task, executor=worker, runs_dir=runs)
    monkeypatch.setattr(workflow, "atomic_json", original)
    run = only_run(runs)
    report = workflow.resume(run["id"], executor=worker, runs_dir=runs)
    assert report["execution"] == "pass"
    assert worker.calls == 1


def test_candidate_and_curator_exports_share_projected_task_identity(tmp_path):
    source = tmp_path / "source"
    fixture = Path(__file__).resolve().parents[1] / "examples/tasks/linear-fit"
    shutil.copytree(fixture, source)
    config = source / "task_config.yaml"
    config.write_text("research_question: Test\nreference_implementation: private.py\n")
    original = read_json(source / "task_manifest.json")
    original["files"]["task_config.yaml"] = digest(config.read_bytes())
    original["private_annotations"] = {"answer": "not public"}
    original["source"] = {"paper_id": "demo", "private_answer": "not public"}
    atomic_json(source / "task_manifest.json", original)
    curator = read_json(source / "curator_manifest.json")
    source_identity = digest(canonical_json(original))
    curator["task_manifest_sha256"] = source_identity
    atomic_json(source / "curator_manifest.json", curator)
    public_path = export_bundle(source, tmp_path / "candidate")
    private_path = export_bundle(source, tmp_path / "protected", audience="curator")
    public = read_json(public_path)
    protected = read_json(private_path)
    assert protected["task_manifest_sha256"] == digest(canonical_json(public))
    assert protected["export_source_task_manifest_sha256"] == source_identity
    assert "private_annotations" not in public
    assert public["source"] == {"paper_id": "demo"}
    assert "reference_implementation" not in (public_path.parent / "task_config.yaml").read_text()
    assert read_json(source / "task_manifest.json") == original


def test_curator_export_cannot_rebind_an_unrelated_evaluator(tmp_path):
    task = bundle(tmp_path)
    atomic_json(task.parent / "curator_manifest.json", {"schema_version": 1, "task_manifest_sha256": "0" * 64})
    destination = tmp_path / "protected"
    with pytest.raises(BundleError, match="does not match"):
        export_bundle(task.parent, destination, audience="curator")
    assert not destination.exists()
