"""Independent checks across authoring input identity and durable finalization."""
from pathlib import Path

import pytest

from paper2bench import api
from paper2bench.journal import RunJournal
from paper2bench.runtime_config import get_context


def profile():
    return {"models": {"local": {"provider": "ollama", "model": "fixture",
                                 "locality_assurance": "operator_trusted"}},
            "roles": {"author": "local", "reference": "local"},
            "policy": {"data_egress": "local_only"}}


def authoring_fixture(tmp_path, monkeypatch, *, parse=None, plan=None):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-authoring fixture; parser is stubbed")
    monkeypatch.setattr("paper2bench.generation.pipeline_identity", lambda: "a" * 64)
    monkeypatch.setattr(api, "parse_paper_to_tree", parse or (lambda *a, **k: {"root": "Question"}))
    monkeypatch.setattr(api, "classify_paper", lambda *a, **k: {"type": "empirical_study"})
    monkeypatch.setattr(api, "extract_yaml_from_pdf", lambda *a, **k: "research_question: Question\npaper_type: empirical_study\n")
    monkeypatch.setattr(api, "generate_supplementary_plan", plan or (lambda *a, **k: "Reference draft"))
    return {"pdf": paper, "research_question": "Question", "task_id": "fixture",
            "output_dir": tmp_path / "output", "runtime_config": profile(),
            "model_runs_dir": tmp_path / "runs", "validate_hf_loaders": False}


def test_template_changed_during_authoring_cannot_change_pinned_generation(tmp_path, monkeypatch):
    template = tmp_path / "template.txt"
    template.write_text("Pinned template: {research_question}")

    def parse(*args, **kwargs):
        # Long authoring calls give an operator time to edit the original file.
        template.write_text("Different template: {research_question}")
        return {"root": "Question"}

    arguments = authoring_fixture(tmp_path, monkeypatch, parse=parse)
    try:
        result = api.run(**arguments, template=template)
    except ValueError:
        # Rejecting a changed input before accepting the result is also safe.
        return
    assert result.instruction == "Pinned template: Question"


def test_cancelled_final_model_response_does_not_return_authoring_success(tmp_path, monkeypatch):
    seen = []

    def plan(*args, **kwargs):
        context = get_context()
        seen.append(context.budget_run_id)
        RunJournal(tmp_path / "runs").request_cancel(context.budget_run_id)
        return "The final model response arrived after cancellation"

    arguments = authoring_fixture(tmp_path, monkeypatch, plan=plan)
    with pytest.raises((ValueError, RuntimeError), match="[Cc]ancel"):
        api.run(**arguments)
    record = RunJournal(tmp_path / "runs").get(seen[0])
    assert record["state"] == "cancelled"
    assert not record["budget"]["reserved"]


def test_verification_data_changed_during_authoring_is_frozen_or_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "output/tasks/fixture/data"
    data_dir.mkdir(parents=True)
    data_file = data_dir / "input.json"
    original = '{"observations": [1, 2, 3]}'
    data_file.write_text(original)
    verified_inputs = []

    def parse(*args, **kwargs):
        data_file.write_text('{"observations": [99, 100, 101]}')
        return {"root": "Question"}

    def verify(*args, **kwargs):
        verified_inputs.append((Path(kwargs["data_dir"]) / "input.json").read_text())
        return {"pass": False, "scientific_reproduction": "not_assessed"}

    class Executor:
        def preflight(self):
            return {"image_id": "sha256:" + "a" * 64}

    arguments = authoring_fixture(tmp_path, monkeypatch, parse=parse)
    monkeypatch.setattr("paper2bench.verify.verify_instruction_gt", verify)
    try:
        api.run(**arguments, verify=True, executor=Executor())
    except ValueError:
        # Failing before verification is safe; assessing changed data and only
        # rejecting the resulting report would already cross the wrong boundary.
        assert verified_inputs == []
        return
    assert verified_inputs == [original]
