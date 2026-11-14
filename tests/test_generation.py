"""Authoring checkpoint regressions exercised through the actual SDK controller."""
import copy
import json
import sqlite3
from pathlib import Path

import pytest

from paper2bench import api
from paper2bench.bundles import atomic_json
from paper2bench.generation import GenerationCheckpoint, GenerationCheckpointError, task_output_lock
from paper2bench.journal import RunJournal
from paper2bench.runtime_config import get_context


def profile():
    return {"schema_version": 1, "profile": "generation-fixture",
            "models": {"local": {"provider": "ollama", "model": "fixture-model",
                                   "hosting": "local_server", "locality_assurance": "operator_trusted"}},
            "roles": {"author": "local", "reference": "local", "agent": "local", "judge": "local"},
            "policy": {"data_egress": "local_only"},
            "budgets": {pool: {"requests": 30, "input_tokens": 10000, "output_tokens": 10000}
                        for pool in ("authoring", "evaluation", "experiments")}}


class Stages:
    def __init__(self, monkeypatch, failure=None):
        self.calls = {stage: 0 for stage in ("parse", "classify", "extract", "render", "plan")}
        self.run_ids = []
        self.failure = failure
        monkeypatch.setattr("paper2bench.generation.pipeline_identity", lambda: "a" * 64)
        monkeypatch.setattr(api, "parse_paper_to_tree", self.parse)
        monkeypatch.setattr(api, "classify_paper", self.classify)
        monkeypatch.setattr(api, "extract_yaml_from_pdf", self.extract)
        monkeypatch.setattr(api, "render_instruction", self.render)
        monkeypatch.setattr(api, "generate_supplementary_plan", self.plan)

    def count(self, stage, paid=True):
        self.calls[stage] += 1
        context = get_context()
        assert context is not None
        self.run_ids.append(context.budget_run_id)
        if paid:
            reservation = context.ledger.reserve("author", 10, 5)
            context.ledger.settle(reservation, 4, 2)

    def parse(self, *args, **kwargs):
        self.count("parse")
        return {"root": "Research question", "children": []}

    def classify(self, *args, **kwargs):
        self.count("classify")
        return {"type": "empirical_study", "confidence": "high", "rationale": "Fixture"}

    def extract(self, *args, **kwargs):
        self.count("extract")
        return "research_question: Does the method work?\npaper_type: empirical_study\n"

    def render(self, *args, **kwargs):
        self.count("render", paid=False)
        return "Investigate the question and report measurements."

    def plan(self, *args, **kwargs):
        self.calls["plan"] += 1
        context = get_context()
        self.run_ids.append(context.budget_run_id)
        if self.calls["plan"] == 1 and self.failure == "known":
            raise ValueError("Fixture plan input requires correction")
        if self.calls["plan"] == 1 and self.failure == "unknown":
            context.ledger.reserve("author", 10, 5)
            raise RuntimeError("Fixture interrupted provider response")
        reservation = context.ledger.reserve("author", 10, 5)
        context.ledger.settle(reservation, 4, 2)
        return "Record all required experimental parameters."


def start(tmp_path, *, template=None, **kwargs):
    paper = tmp_path / "paper.pdf"
    if not paper.exists():
        paper.write_bytes(b"%PDF-synthetic fixture; parsing is stubbed")
    return api.run(pdf=paper, research_question="Does the method work?", task_id="study",
                   output_dir=tmp_path / "output", runtime_config=profile(),
                   model_runs_dir=tmp_path / "runs", validate_hf_loaders=False,
                   template=template, **kwargs)


def authoring_record(tmp_path):
    journal = RunJournal(tmp_path / "runs")
    with sqlite3.connect(journal.path) as connection:
        ids = [row[0] for row in connection.execute("SELECT id FROM runs ORDER BY created")]
    records = [journal.get(run_id) for run_id in ids]
    return next(record for record in records if record["spec"]["kind"] == "authoring")


def failed_run(tmp_path, monkeypatch, *, template=None):
    stages = Stages(monkeypatch, failure="known")
    with pytest.raises(ValueError, match="input requires correction"):
        start(tmp_path, template=template)
    record = authoring_record(tmp_path)
    assert record["state"] == "failed"
    return stages, record


def test_known_failure_resume_reuses_completed_stages_and_original_budget(tmp_path, monkeypatch):
    stages, before = failed_run(tmp_path, monkeypatch)
    assert stages.calls == {"parse": 1, "classify": 1, "extract": 1, "render": 1, "plan": 1}
    assert before["budget"]["used"]["authoring.requests"] == 3
    resumed = api.resume_generation(before["id"], model_runs_dir=tmp_path / "runs")
    after = RunJournal(tmp_path / "runs").get(before["id"])
    assert after["state"] == "succeeded" and len(after["attempts"]) == 2
    assert stages.calls == {"parse": 1, "classify": 1, "extract": 1, "render": 1, "plan": 2}
    assert set(stages.run_ids) == {before["id"]}
    assert resumed.extras["model_run_id"] == before["id"]
    assert resumed.extras["resumed_stages"] == ["parse", "classify", "extract", "render"]
    assert resumed.extras["cache_semantics"] == "historical_artifact_reuse"
    assert after["budget"]["used"]["authoring.requests"] == 4
    assert not after["budget"]["reserved"]


def test_unknown_plan_outcome_keeps_reservation_and_never_blindly_replays(tmp_path, monkeypatch):
    stages = Stages(monkeypatch, failure="unknown")
    with pytest.raises(RuntimeError, match="interrupted"):
        start(tmp_path)
    record = authoring_record(tmp_path)
    assert record["state"] == "reconciling"
    assert record["budget"]["reserved"]["authoring.requests"] == 1
    with pytest.raises(ValueError, match="reconciliation|unknown"):
        api.resume_generation(record["id"], model_runs_dir=tmp_path / "runs")
    assert stages.calls["plan"] == 1


def test_corrupted_completed_stage_receipt_is_rejected_without_model_replay(tmp_path, monkeypatch):
    stages, record = failed_run(tmp_path, monkeypatch)
    receipt = tmp_path / "runs/generation" / record["id"] / "parse.json"
    atomic_json(receipt, {"value": {"fabricated": "tree"}})
    with pytest.raises(GenerationCheckpointError, match="integrity"):
        api.resume_generation(record["id"], model_runs_dir=tmp_path / "runs")
    assert stages.calls == {"parse": 1, "classify": 1, "extract": 1, "render": 1, "plan": 1}


@pytest.mark.parametrize("changed", ["paper", "template", "pipeline", "runtime"])
def test_failed_resume_rejects_source_prompt_template_and_runtime_changes(tmp_path, monkeypatch, changed):
    template = tmp_path / "template.txt"
    template.write_text("Template v1")
    stages, record = failed_run(tmp_path, monkeypatch, template=template)
    replacement = None
    if changed == "paper":
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-different source")
    elif changed == "template":
        template.write_text("Template v2")
    elif changed == "pipeline":
        monkeypatch.setattr("paper2bench.generation.pipeline_identity", lambda: "b" * 64)
    else:
        replacement = profile()
        replacement["models"]["local"]["model"] = "different-model"
    with pytest.raises(ValueError, match="changed|different|identity"):
        api.resume_generation(record["id"], model_runs_dir=tmp_path / "runs", runtime_config=replacement)
    assert stages.calls["plan"] == 1


def test_completed_resume_returns_history_without_model_or_budget_reexecution(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    first = start(tmp_path)
    run_id = first.extras["model_run_id"]
    before = RunJournal(tmp_path / "runs").get(run_id)
    cached = api.resume_generation(run_id, model_runs_dir=tmp_path / "runs")
    after = RunJournal(tmp_path / "runs").get(run_id)
    assert stages.calls == {"parse": 1, "classify": 1, "extract": 1, "render": 1, "plan": 1}
    assert cached.extras["model_run_id"] == run_id
    assert cached.extras["cache_semantics"] == "historical_artifact_reuse"
    assert cached.instruction == first.instruction
    assert isinstance(cached.instruction_path, Path)
    assert before["budget"] == after["budget"] and len(after["attempts"]) == 1


@pytest.mark.parametrize("changed", ["paper", "template", "pipeline", "runtime"])
def test_completed_resume_still_validates_current_identities(tmp_path, monkeypatch, changed):
    stages = Stages(monkeypatch)
    template = tmp_path / "template.txt"
    template.write_text("Template v1")
    result = start(tmp_path, template=template)
    replacement = None
    if changed == "paper":
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-different source")
    elif changed == "template":
        template.write_text("Template v2")
    elif changed == "pipeline":
        monkeypatch.setattr("paper2bench.generation.pipeline_identity", lambda: "b" * 64)
    else:
        replacement = profile()
        replacement["models"]["local"]["model"] = "different-model"
    with pytest.raises(ValueError, match="changed|different|identity"):
        api.resume_generation(result.extras["model_run_id"], model_runs_dir=tmp_path / "runs", runtime_config=replacement)
    assert stages.calls["plan"] == 1


def test_completed_result_corruption_is_rejected(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    result = start(tmp_path)
    path = tmp_path / "runs/generation" / result.extras["model_run_id"] / "result.json"
    stored = json.loads(path.read_text())
    stored["value"]["instruction"] = "Tampered instruction"
    atomic_json(path, stored)
    with pytest.raises(GenerationCheckpointError, match="integrity"):
        api.resume_generation(result.extras["model_run_id"], model_runs_dir=tmp_path / "runs")
    assert stages.calls["plan"] == 1


def test_task_output_lock_prevents_any_model_dispatch(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    with task_output_lock(tmp_path / "output/tasks/study"):
        with pytest.raises(GenerationCheckpointError, match="owns"):
            start(tmp_path)
    assert all(count == 0 for count in stages.calls.values())
    assert not (tmp_path / "runs/runs.sqlite3").exists()


def test_unknown_direct_stage_cannot_be_replayed(tmp_path):
    plan = {"paper_sha256": "a" * 64}
    checkpoint = GenerationCheckpoint(tmp_path, "a" * 32, plan)
    calls = []

    def interrupted():
        calls.append(True)
        raise KeyboardInterrupt("Unknown external outcome")

    with pytest.raises(KeyboardInterrupt):
        checkpoint.run_stage("extract", interrupted)
    resumed = GenerationCheckpoint(tmp_path, "a" * 32, plan, resume=True)
    with pytest.raises(GenerationCheckpointError, match="unknown"):
        resumed.run_stage("extract", interrupted)
    assert calls == [True]


def test_checkpoint_mismatched_run_id_is_not_accepted_as_another_run(tmp_path):
    plan = {"paper_sha256": "a" * 64}
    checkpoint = GenerationCheckpoint(tmp_path, "a" * 32, plan)
    state = json.loads(checkpoint.path.read_text())
    state["run_id"] = "b" * 32
    atomic_json(checkpoint.path, state)
    with pytest.raises(GenerationCheckpointError, match="identity|run|schema"):
        GenerationCheckpoint(tmp_path, "a" * 32, plan, resume=True)


def test_completed_resume_cannot_return_paths_pointing_to_another_runs_artifacts(tmp_path, monkeypatch):
    Stages(monkeypatch)
    result = start(tmp_path)
    result.instruction_path.write_text("A later run replaced this shared task output")
    try:
        cached = api.resume_generation(result.extras["model_run_id"], model_runs_dir=tmp_path / "runs")
    except ValueError:
        # Explicitly rejecting a changed artifact is safe. A recovery implementation
        # may instead restore/reference the original immutable checkpoint artifact.
        return
    assert cached.instruction_path.read_text() == cached.instruction


def stranded_attempt(tmp_path, run_id):
    """Model process death immediately before the durable terminal transition."""
    journal = RunJournal(tmp_path / "runs")
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE runs SET state='running' WHERE id=?", (run_id,))
        connection.execute("UPDATE attempts SET state='running',result=NULL WHERE run_id=?", (run_id,))


def test_crash_after_complete_checkpoint_finalizes_journal_without_model_replay(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    result = start(tmp_path)
    run_id = result.extras["model_run_id"]
    stranded_attempt(tmp_path, run_id)
    recovered = api.resume_generation(run_id, model_runs_dir=tmp_path / "runs")
    record = RunJournal(tmp_path / "runs").get(run_id)
    assert recovered.instruction == result.instruction
    assert stages.calls == {"parse": 1, "classify": 1, "extract": 1, "render": 1, "plan": 1}
    assert record["state"] == "succeeded"
    assert record["attempts"][-1]["result"]["recovered_generation_checkpoint"] is True
    assert not record["budget"]["reserved"]


def test_crash_after_known_stage_failure_can_resume_settled_work(tmp_path, monkeypatch):
    stages, record = failed_run(tmp_path, monkeypatch)
    stranded_attempt(tmp_path, record["id"])
    api.resume_generation(record["id"], model_runs_dir=tmp_path / "runs")
    assert stages.calls == {"parse": 1, "classify": 1, "extract": 1, "render": 1, "plan": 2}
    assert RunJournal(tmp_path / "runs").get(record["id"])["state"] == "succeeded"


def test_authoring_recovery_cannot_race_a_live_output_owner(tmp_path, monkeypatch):
    stages, record = failed_run(tmp_path, monkeypatch)
    stranded_attempt(tmp_path, record["id"])
    with task_output_lock(tmp_path / "output/tasks/study"):
        with pytest.raises(GenerationCheckpointError, match="owns"):
            api.resume_generation(record["id"], model_runs_dir=tmp_path / "runs")
    assert stages.calls["plan"] == 1
    assert RunJournal(tmp_path / "runs").get(record["id"])["state"] == "running"


def test_historical_verification_file_tampering_is_rejected(tmp_path, monkeypatch):
    Stages(monkeypatch)
    monkeypatch.setattr("paper2bench.verify.verify_instruction_gt", lambda *a, **k: {"pass": False})

    class Executor:
        def preflight(self):
            return {"image_id": "sha256:" + "a" * 64}

    result = start(tmp_path, verify=True, executor=Executor())
    verification = result.instruction_gt_path.parent / "verify_results.json"
    atomic_json(verification, {"pass": True})
    with pytest.raises(GenerationCheckpointError, match="artifact|integrity"):
        api.resume_generation(result.extras["model_run_id"], model_runs_dir=tmp_path / "runs")


def test_receipt_byte_limit_matches_the_bytes_written(tmp_path, monkeypatch):
    from paper2bench import generation
    monkeypatch.setattr(generation, "MAX_STAGE_BYTES", 100)
    plan = {"fixture": "bounded receipt"}
    checkpoint = GenerationCheckpoint(tmp_path, "c" * 32, plan)
    values = [0] * 30  # Compact JSON fits; an indented representation exceeds 100 bytes.
    checkpoint.run_stage("parse", lambda: values)
    checkpoint.complete({"values": values})
    resumed = GenerationCheckpoint(tmp_path, "c" * 32, plan, resume=True)
    resumed.ensure_resumable()
    assert resumed.run_stage("parse", lambda: pytest.fail("No replay")) == values
    assert resumed.completed_result() == {"values": values}


@pytest.mark.parametrize("changed", ["source_data", "snapshot_data", "template_snapshot"])
def test_completed_resume_validates_frozen_authoring_inputs(tmp_path, monkeypatch, changed):
    stages = Stages(monkeypatch)
    monkeypatch.setattr("paper2bench.verify.verify_instruction_gt", lambda *a, **k: {"pass": False})

    class Executor:
        def preflight(self):
            return {"image_id": "sha256:" + "a" * 64}

    data = tmp_path / "output/tasks/study/data"
    data.mkdir(parents=True)
    (data / "input.json").write_text('{"observations": [1, 2]}')
    template = tmp_path / "template.txt"
    template.write_text("Template v1")
    result = start(tmp_path, verify=True, executor=Executor(), template=template)
    target = {"source_data": data / "input.json",
              "snapshot_data": Path(result.extras["verification_data_snapshot"]) / "input.json",
              "template_snapshot": Path(result.extras["template_snapshot"])}[changed]
    target.write_text("Changed input identity")
    with pytest.raises(ValueError, match="changed|identity|artifact"):
        api.resume_generation(result.extras["model_run_id"], model_runs_dir=tmp_path / "runs")
    assert stages.calls["plan"] == 1
