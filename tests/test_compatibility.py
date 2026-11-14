import json
from pathlib import Path

import pytest
import yaml

from paper2bench import api, cli
from paper2bench.llm import ChatResponse


def test_sdk_legacy_fields_and_explicit_key_reach_verification(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fixture")
    monkeypatch.setattr(api, "parse_paper_to_tree", lambda *a, **k: {"root": "question"})
    monkeypatch.setattr(api, "classify_paper", lambda *a, **k: {"type": "empirical_study"})
    monkeypatch.setattr(api, "extract_yaml_from_pdf", lambda *a, **k: "research_question: Question\n")
    monkeypatch.setattr(api, "generate_supplementary_plan", lambda *a, **k: "Reference plan")
    from paper2bench import verify
    seen = []
    monkeypatch.setattr(verify, "verify_instruction_gt", lambda *a, **k: seen.append(k) or {"pass": False})

    class Executor:
        def preflight(self):
            return {}

    result = api.run(pdf=pdf, research_question="Question", api_key="explicit-test-key",
                     verify=True, executor=Executor(), output_dir=tmp_path / "out", validate_hf_loaders=False)
    assert result.instruction_path.name == "instruction.txt"
    assert result.instruction_gt_path.name == "instruction_gt.txt"
    assert result.config_path.name == "task_config.yaml"
    assert seen[0]["api_key"] == "explicit-test-key"
    assert Path(result.extras["task_manifest_path"]).exists()
    assert result.extras["readiness"] == "needs_review"


def test_default_host_execution_is_not_available():
    import inspect
    from paper2bench import verify
    source = inspect.getsource(verify)
    assert "subprocess.run" not in source
    assert "os.environ.copy" not in source
    assert "load_dotenv()" not in source


def test_offline_dataset_helpers_emit_unverified_state_without_hub_access(monkeypatch):
    from paper2bench.instruction_generator import _validate_hf_loaders, find_external_dataset_links, resolve_unknown_datasets
    from paper2bench.runtime_config import runtime_scope
    config = {"models": {"local": {"provider": "ollama", "model": "test", "hosting": "local_server",
                                  "locality_assurance": "operator_trusted"}},
              "roles": {"author": "local"}, "policy": {"data_egress": "local_only"}}
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: pytest.fail("Hub access under local_only"))
    with runtime_scope(config):
        for function in (_validate_hf_loaders, resolve_unknown_datasets, find_external_dataset_links):
            output = yaml.safe_load(function("datasets:\n - name: Data\n   source: unknown\n"))
            assert output["resource_validation"] == {"status": "not_run", "reason": "local_only"}


def test_classifier_does_not_silently_select_wrong_archetype(monkeypatch):
    from paper2bench import paper_classifier
    monkeypatch.setattr(paper_classifier, "extract_text_from_pdf", lambda *a, **k: "Paper")
    monkeypatch.setattr(paper_classifier, "chat", lambda *a, **k: ChatResponse('{"type":"nonsense"}'))
    with pytest.raises(ValueError, match="review"):
        paper_classifier.classify_paper("ignored.pdf")


def test_cli_task_validation_and_legacy_render(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("research_question: Does it work?\npaper_type: empirical_study\n")
    output = tmp_path / "instruction.txt"
    cli.main(["render", "--config", str(config), "-o", str(output)])
    assert "Does it work?" in output.read_text()
    task = Path(__file__).resolve().parents[1] / "examples/tasks/knapsack/task_manifest.json"
    cli.main(["validate", "--task", str(task), "--executable"])
    assert '"schema_valid": true' in capsys.readouterr().out


def test_cli_rejects_path_task_id_before_writing(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--task-id", "../escape", "--pdf", "missing.pdf"])
    assert error.value.code == 2


@pytest.mark.parametrize("content", ['{}', '{"leaked":"false"}', 'not JSON'])
def test_malformed_leakage_judge_never_marks_variant_clean(monkeypatch, content):
    from paper2bench import benchmark_generator
    monkeypatch.setattr(benchmark_generator, "chat", lambda *a, **k: ChatResponse(content))
    result = benchmark_generator._check_leakage("task", {}, "paper", "model", None)
    assert result["leaked"] is None and result["status"] == "error"
