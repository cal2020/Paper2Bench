"""Independent cross-boundary regressions from the final swarm audit.

Synthetic contracts exercise software; they are not evidence about a paper.
"""
from pathlib import Path
import shutil
import pytest

from paper2bench import workflow
from paper2bench.bundles import atomic_json, canonical_json, digest, read_json
from paper2bench.executors import ExecutionResult
from paper2bench.generation import GenerationCheckpoint, GenerationCheckpointError
from paper2bench import generation
from paper2bench.journal import JournalError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/tasks"


def scientific_fixture(tmp_path, method="linear-fit"):
    root = tmp_path / "task"
    shutil.copytree(EXAMPLES / method, root)
    manifest = read_json(root / "task_manifest.json")
    manifest["source"] = {"paper_sha256": digest(b"synthetic test source")}
    atomic_json(root / "task_manifest.json", manifest)
    task_hash = digest(canonical_json(manifest))
    code_hash = manifest["files"][manifest["entrypoint"]]
    paper_hash = manifest["source"]["paper_sha256"]
    knapsack = method == "knapsack"
    protocol = {
        "schema_version": 1, "profile": "reproduction/v1", "protocol_id": "audit-fixture",
        "task_manifest_sha256": task_hash, "submission_sha256": code_hash,
        "source": {"paper_sha256": paper_hash, "citations": [
            {"id": "method", "locator": "synthetic contract", "claim": "Test fixture only"}]},
        "datasets": {manifest["inputs"]: manifest["files"][manifest["inputs"]]},
        "plan": {"seed_schedule": [0], "stopping_rule": "fixed_schedule",
                 "variability_source": "controlled fixture", "independence_assumption": False},
        "criteria": [
            {"id": "execution", "kind": "execution", "dimension": "execution", "critical": True, "depends_on": []},
            {"id": "fidelity", "kind": "review", "dimension": "implementation_fidelity", "critical": True,
             "depends_on": ["execution"], "review_id": "review"},
            {"id": "result", "kind": "metric", "dimension": "empirical_agreement", "critical": True,
             "depends_on": ["execution", "fidelity"],
             "metric_id": "feasible_optimal_objective" if knapsack else "mean_squared_error",
             "units": "objective_units" if knapsack else "target_units_squared", "rule": "absolute_tolerance",
             "reference": 12 if knapsack else 0, "tolerance": 1e-12}],
        "reviews": [{"id": "review", "reviewer": "synthetic fixture curator", "status": "pass",
                     "submission_sha256": code_hash, "paper_sha256": paper_hash,
                     "source_citations": ["method"], "evidence_refs": ["fixture-review"]}],
    }
    curator = read_json(root / "curator_manifest.json")
    curator.update(task_manifest_sha256=task_hash, protocol=protocol)
    atomic_json(root / "curator_manifest.json", curator)
    return root / "task_manifest.json", root / "curator_manifest.json"


class Worker:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def preflight(self):
        return {"image_id": "sha256:" + "1" * 64}

    def cancel(self):
        return True

    def execute(self, request):
        self.calls.append(request.seed)
        atomic_json(request.workspace / "predictions.json", self.output)
        return ExecutionResult("succeeded", 0, "", "", 0.01, evidence={"cleanup_confirmed": True})


def test_scientific_parent_can_be_reproduced_from_its_frozen_bundle(tmp_path):
    task, curator = scientific_fixture(tmp_path)
    worker = Worker({"predictions": [9, 11, 21]})
    report = workflow.execute(task, evaluation_bundle=curator, executor=worker,
                              runs_dir=tmp_path / "runs", evaluation_profile="reproduction/v1")
    copied = workflow.reproduce(report["run_id"], executor=worker, runs_dir=tmp_path / "runs")
    assert copied["pass"]
    assert copied["run_id"] != report["run_id"]
    assert worker.calls == [0, 0]


def test_infeasible_knapsack_objective_is_not_a_successful_scientific_metric(tmp_path):
    task, curator = scientific_fixture(tmp_path, method="knapsack")
    # weight 2+3+4=9 exceeds capacity8, but objective3+4+5=12 matches optimum.
    # Numerical equality cannot override the installed scorer's feasibility check.
    worker = Worker({"selected": [0, 1, 2]})
    report = workflow.execute(task, evaluation_bundle=curator, executor=worker,
                              runs_dir=tmp_path / "runs", evaluation_profile="reproduction/v1")
    assert not report["pass"]
    assert report["scientific_reproduction"] != "pass"


def test_historical_generation_does_not_link_to_an_edited_candidate_manifest(tmp_path):
    root = tmp_path / "candidate"
    shutil.copytree(EXAMPLES / "linear-fit", root)
    task = root / "task_manifest.json"
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("original generated instruction")
    checkpoint = GenerationCheckpoint(tmp_path / "checkpoints", "a" * 32, {"fixture": True})
    checkpoint.complete({"instruction_path": str(instruction),
                         "extras": {"task_manifest_path": str(task)}})
    changed = read_json(task)
    changed["research_question"] = "A different candidate task"
    atomic_json(task, changed)
    with pytest.raises(GenerationCheckpointError, match="artifact|candidate|integrity"):
        checkpoint.completed_result()


def test_generation_cannot_commit_a_result_its_own_reader_cannot_load(tmp_path, monkeypatch):
    checkpoint = GenerationCheckpoint(tmp_path / "checkpoints", "b" * 32, {"fixture": True})
    monkeypatch.setattr(generation, "MAX_STAGE_BYTES", 128)
    with pytest.raises(GenerationCheckpointError, match="limit|bound|size"):
        checkpoint.complete({"aggregate": "x" * 256})
    assert checkpoint.state["state"] != "succeeded"


def test_scientific_parent_rejects_a_child_from_a_different_dataset(tmp_path, monkeypatch):
    task, curator = scientific_fixture(tmp_path)
    original = workflow.execute

    class DataWorker(Worker):
        def execute(self, request):
            data = read_json(request.workspace / "input.json")
            self.output = {"predictions": [2*x+1 for x in data["test"]]}
            return super().execute(request)

    worker = DataWorker({})

    def wrong_condition(_task, **kwargs):
        other = tmp_path / "other-condition"
        shutil.copytree(Path(_task).parent, other)
        changed = read_json(other / "input.json")
        changed["test"] = [6, 7, 8]
        atomic_json(other / "input.json", changed)
        manifest = read_json(other / "task_manifest.json")
        manifest["files"]["input.json"] = digest((other / "input.json").read_bytes())
        atomic_json(other / "task_manifest.json", manifest)
        protected = read_json(kwargs["evaluation_bundle"])
        protected["task_manifest_sha256"] = digest(canonical_json(manifest))
        protected["reference"]["labels"] = [13, 15, 17]
        protected_path = tmp_path / "other-curator.json"
        atomic_json(protected_path, protected)
        kwargs["evaluation_bundle"] = protected_path
        return original(other / "task_manifest.json", **kwargs)

    monkeypatch.setattr(workflow, "execute", wrong_condition)
    from paper2bench.scientific_runs import execute_scientific
    with pytest.raises(JournalError, match="identity|condition|frozen|task"):
        execute_scientific(task, evaluation_bundle=curator, executor=worker,
                           runs_dir=tmp_path / "runs", evaluation_profile="reproduction/v1")
