import json
import os
import shutil
from pathlib import Path

import pytest

from paper2bench.bundles import BundleError, atomic_json, read_json
from paper2bench.executors import ExecutionResult
from paper2bench.journal import JournalError, RunJournal
from paper2bench.workflow import execute, reproduce, resume

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tasks/linear-fit"


class FakeExecutor:
    def __init__(self, predictions=None, status="succeeded", cleanup=True):
        self.predictions = predictions or {"predictions": [9, 11, 21]}
        self.status = status
        self.cleanup = cleanup
        self.calls = []

    def preflight(self):
        return {"image_id": "sha256:" + "1" * 64}

    def cancel(self):
        return True

    def execute(self, request):
        self.calls.append(request)
        assert not (request.workspace / "curator_manifest.json").exists()
        atomic_json(request.workspace / "predictions.json", self.predictions)
        return ExecutionResult(self.status, 0 if self.status == "succeeded" else 1, "", "", 0.1,
                               evidence={"cleanup_confirmed": self.cleanup})


def run_example(tmp_path, executor=None):
    return execute(EXAMPLE / "task_manifest.json", evaluation_bundle=EXAMPLE / "curator_manifest.json",
                   runs_dir=tmp_path, executor=executor or FakeExecutor(), evaluation_profile="method_output/v1")


def test_trusted_scoring_durable_resume_and_new_reproduction(tmp_path):
    worker = FakeExecutor()
    report = run_example(tmp_path, worker)
    assert report["pass"] and report["metric"]["value"] == 0
    assert report["scientific_reproduction"] == "not_assessed"
    assert resume(report["run_id"], runs_dir=tmp_path, executor=worker) == report
    assert len(worker.calls) == 1
    copied = reproduce(report["run_id"], runs_dir=tmp_path, executor=worker)
    assert copied["run_id"] != report["run_id"]
    assert RunJournal(tmp_path).get(copied["run_id"])["spec"]["parent_run"] == report["run_id"]
    assert (tmp_path / report["run_id"] / "evidence.jsonl").exists()


def test_printed_gold_metric_cannot_override_bad_predictions(tmp_path):
    worker = FakeExecutor({"predictions": [0, 0, 0], "mse": 0, "pass": True})
    report = run_example(tmp_path, worker)
    assert report["execution"] == "pass" and report["method_output"] == "fail"
    assert not report["pass"] and report["metric"]["value"] > 0


def test_unknown_cleanup_retains_budget_and_blocks_retry(tmp_path):
    report = run_example(tmp_path, FakeExecutor(cleanup=False))
    run = RunJournal(tmp_path).get(report["run_id"])
    assert run["state"] == "reconciling" and not report["pass"]
    assert run["budget"]["reserved"]["wall_seconds"] > 0
    with pytest.raises(JournalError):
        resume(report["run_id"], runs_dir=tmp_path, executor=FakeExecutor())


def test_confirmed_failure_can_retry_within_total_budget(tmp_path):
    report = run_example(tmp_path, FakeExecutor(status="failed"))
    second = resume(report["run_id"], runs_dir=tmp_path, executor=FakeExecutor())
    assert second["pass"] and second["attempt_id"] != report["attempt_id"]
    assert len(RunJournal(tmp_path).get(report["run_id"])["attempts"]) == 2


@pytest.mark.parametrize("operation", [resume, reproduce])
def test_installed_scorer_protocol_is_pinned_before_reuse(tmp_path, monkeypatch, operation):
    from paper2bench import workflow
    worker = FakeExecutor()
    report = run_example(tmp_path, worker)
    stored = RunJournal(tmp_path).get(report["run_id"])
    assert stored["spec"]["scorer_protocol"] == report["scorer_protocol"]
    monkeypatch.setitem(workflow.SCORER_PROTOCOLS, "linear_predictions/v1", "linear_predictions/future")
    with pytest.raises(JournalError, match="scorer protocol changed"):
        operation(report["run_id"], runs_dir=tmp_path, executor=worker)
    assert len(worker.calls) == 1


def test_mismatched_evaluator_or_missing_protocol_rejected(tmp_path):
    curator = read_json(EXAMPLE / "curator_manifest.json")
    curator["task_manifest_sha256"] = "0" * 64
    atomic_json(tmp_path / "curator.json", curator)
    with pytest.raises(BundleError):
        execute(EXAMPLE / "task_manifest.json", evaluation_bundle=tmp_path / "curator.json", executor=FakeExecutor())
    with pytest.raises(BundleError, match="scientific evaluation bundle"):
        execute(EXAMPLE / "task_manifest.json", evaluation_profile="reproduction/v1", executor=FakeExecutor())


@pytest.mark.skipif(os.getenv("P2B_DOCKER_TESTS") != "1", reason="opt-in real isolated method execution")
@pytest.mark.parametrize("name", ["linear-fit", "knapsack"])
def test_real_non_llm_task_execution(tmp_path, name):
    path = EXAMPLE.parent / name
    report = execute(path / "task_manifest.json", evaluation_bundle=path / "curator_manifest.json",
                     runs_dir=tmp_path, evaluation_profile="method_output/v1")
    assert report["pass"], report
    assert report["metric"]["authority"].startswith("trusted_")
