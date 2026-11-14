import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper2bench.verify as verify
from paper2bench.evaluation import EvaluationError
from paper2bench.runtime_config import runtime_scope


class FakeExecutor:
    def __init__(self, *, available=True, status="succeeded", exit_code=0):
        self.available = available
        self.status = status
        self.exit_code = exit_code
        self.preflights = 0
        self.requests = []

    def preflight(self):
        self.preflights += 1
        if not self.available:
            raise RuntimeError("Isolation unavailable")
        return {"backend": "fake-isolated"}

    def execute(self, request):
        self.requests.append(request)
        assert not hasattr(request, "env")
        assert not (request.workspace / ".env").exists()
        return SimpleNamespace(
            status=self.status, exit_code=self.exit_code,
            stdout="measured: 2.5", stderr="", duration_seconds=0.1,
            backend="fake", isolation="test-double", evidence={"nonce": "abc"}, artifacts={},
        )

    def cancel(self):
        return True


@pytest.fixture
def instruction(tmp_path):
    path = tmp_path / "instruction.txt"
    path.write_text("Measure the specified calculation using permitted inputs.")
    return path


def model_replies(monkeypatch, *replies):
    responses = iter(replies)
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(verify, "chat", fake_chat)
    return calls


def test_preflight_failure_precedes_all_model_calls(monkeypatch, instruction):
    calls = model_replies(monkeypatch)
    with pytest.raises(RuntimeError, match="Isolation unavailable"):
        verify.verify_instruction_gt(str(instruction), "paper.pdf", executor=FakeExecutor(available=False))
    assert calls == []


def test_invalid_execution_limits_precede_paid_reference_calls(monkeypatch, instruction, tmp_path):
    calls = model_replies(monkeypatch)
    with pytest.raises(ValueError):
        verify.verify_instruction_gt(str(instruction), "paper.pdf", executor=FakeExecutor(),
                                     run_dir=str(tmp_path / "run"),
                                     execution_options={"memory_bytes": 16 * 1024 * 1024})
    assert calls == []


def test_controller_admission_precedes_executor_preflight(monkeypatch, instruction, tmp_path):
    calls = model_replies(monkeypatch)
    executor = FakeExecutor()
    with pytest.raises(ValueError, match="operator limit"):
        verify.run_agent_detailed(str(instruction), executor=executor,
                                  run_dir=str(tmp_path / "run"),
                                  execution_options={"memory_bytes": 8 * 1024**3})
    assert executor.preflights == 0
    assert calls == []


def test_scoped_turn_and_call_deadline_limits_are_honored(monkeypatch, instruction, tmp_path):
    calls = model_replies(monkeypatch, "still investigating")
    config = {
        "schema_version": 1,
        "models": {"local": {"provider": "ollama", "model": "operator-model"}},
        "roles": {"agent": "local"},
        "budgets": {"experiments": {"agent_turns": 1, "wall_seconds": 5,
                                       "memory_bytes": 67108864, "disk_bytes": 16777216,
                                       "cpu_seconds": 5}},
    }
    with runtime_scope(config):
        result = verify.run_agent_detailed(
            str(instruction), executor=FakeExecutor(), max_turns=30,
            execution_options={"timeout_seconds": 1, "memory_bytes": 67108864,
                               "workspace_bytes": 16777216}, run_dir=str(tmp_path / "run"))
    assert result.status == "turn_limit"
    assert len(calls) == 1
    assert 0 < calls[0]["timeout_seconds"] <= 5


def test_cpu_budget_is_conservative_and_cumulative_across_code_blocks(monkeypatch, instruction, tmp_path):
    model_replies(monkeypatch,
        "```python\nprint(1)\n```\n```python\nprint(2)\n```\nFINAL CONCLUSION: Two observations.")
    config = {
        "schema_version": 1,
        "models": {"local": {"provider": "ollama", "model": "operator-model"}},
        "roles": {"agent": "local"},
        "budgets": {"experiments": {"cpu_seconds": 1, "wall_seconds": 5,
                                       "memory_bytes": 67108864, "disk_bytes": 16777216}},
    }
    executor = FakeExecutor()
    with runtime_scope(config):
        result = verify.run_agent_detailed(
            str(instruction), executor=executor,
            execution_options={"timeout_seconds": 1, "cpus": 1, "memory_bytes": 67108864,
                               "workspace_bytes": 16777216}, run_dir=str(tmp_path / "run"))
    assert len(executor.requests) == 2
    assert executor.requests[0].timeout_seconds == 1
    assert executor.requests[1].timeout_seconds == 0.9
    assert [x["cpu_seconds_upper_bound"] for x in result.executions] == [0.1, 0.1]


def test_backend_job_intent_is_durable_before_dispatch_and_callback_is_restored(monkeypatch, instruction, tmp_path):
    model_replies(monkeypatch, "```python\nprint(1)\n```\nFINAL CONCLUSION: One.")
    forwarded = []
    original_callback = lambda event, payload: forwarded.append(event)
    run_dir = tmp_path / "run"

    class LifecycleExecutor(FakeExecutor):
        lifecycle_callback = original_callback

        def execute(self, request):
            self.lifecycle_callback("job_intent", {"container_name": "p2b-test"})
            durable = json.loads((run_dir / "execution_lifecycle.json").read_text())
            assert durable[-1]["event"] == "job_intent"
            assert durable[-1]["payload"]["container_name"] == "p2b-test"
            result = super().execute(request)
            self.lifecycle_callback("job_cleanup", {"container_name": "p2b-test", "confirmed": True})
            return result

    executor = LifecycleExecutor()
    # Assign the function to the instance to avoid Python method binding.
    executor.lifecycle_callback = original_callback
    result = verify.run_agent_detailed(str(instruction), executor=executor, run_dir=str(run_dir))
    assert result.execution_verified
    assert forwarded == ["job_intent", "job_cleanup"]
    assert executor.lifecycle_callback is original_callback
    records = json.loads((run_dir / "execution_lifecycle.json").read_text())
    assert [entry["block_index"] for entry in records] == [1, 1]
    assert [entry["turn_index"] for entry in records] == [1, 1]
    assert all(entry["run_id"] == run_dir.name for entry in records)


def test_conclusion_without_execution_is_not_scientific_verification(monkeypatch, instruction, tmp_path):
    model_replies(monkeypatch,
        "FINAL CONCLUSION: The value is 2.5.",
        json.dumps({"score": 100, "strengths": "Agrees", "weaknesses": "",
                    "scope_mismatch": "", "rationale": "Text agrees"}),
    )
    report = verify.verify_instruction_gt(
        str(instruction), "not-read.pdf", ground_truth="The value is 2.5.",
        eval_mode="llm_judge", executor=FakeExecutor(), run_dir=str(tmp_path / "run"),
    )
    assert report["pass"] is True  # Explicitly legacy conclusion similarity only.
    assert report["evaluation_scope"] == "conclusion_similarity_only"
    assert report["execution_verified"] is False
    assert report["execution_status"] == "no_execution"
    assert report["scientific_verdict"] == "not_assessed"
    assert report["execution_evidence"] == []


def test_code_and_conclusion_in_same_message_execute_first(monkeypatch, instruction, tmp_path):
    model_replies(monkeypatch, "```python\nprint(2.5)\n```\nFINAL CONCLUSION: Observed 2.5.")
    executor = FakeExecutor()
    result = verify.run_agent_detailed(str(instruction), executor=executor,
                                      run_dir=str(tmp_path / "run"))
    assert len(executor.requests) == 1
    assert result.execution_verified is True
    assert result.conclusion == "Observed 2.5."
    assert result.executions[0]["exit_code"] == 0
    record = json.loads((Path(result.run_dir) / "agent_run.json").read_text())
    assert record["executions"][0]["code_sha256"]
    assert not list((Path(result.run_dir) / "workspace").glob("*"))
    assert (Path(result.run_dir) / "log" / "block_0001.py").read_text() == "print(2.5)\n"


def test_failed_code_preserves_exit_and_does_not_verify_execution(monkeypatch, instruction, tmp_path):
    model_replies(monkeypatch, "```python\nraise RuntimeError()\n```\nFINAL CONCLUSION: Could not complete.")
    result = verify.run_agent_detailed(str(instruction), executor=FakeExecutor(status="failed", exit_code=7),
                                      run_dir=str(tmp_path / "run"))
    assert result.executions[0]["exit_code"] == 7
    assert result.execution_verified is False


def test_agent_limits_do_not_make_an_extra_unbudgeted_call(monkeypatch, instruction, tmp_path):
    calls = model_replies(monkeypatch, "thinking", "still thinking")
    result = verify.run_agent_detailed(str(instruction), executor=FakeExecutor(), max_turns=2,
                                      run_dir=str(tmp_path / "run"))
    assert len(calls) == 2
    assert result.status == "turn_limit"
    assert result.conclusion == ""


def test_run_agent_facade_still_returns_string(monkeypatch, instruction, tmp_path):
    model_replies(monkeypatch, "FINAL CONCLUSION: Insufficient evidence.")
    assert verify.run_agent(str(instruction), executor=FakeExecutor(),
                            run_dir=str(tmp_path / "run")) == "Insufficient evidence."


def test_provider_keys_are_not_staged_or_recorded(monkeypatch, instruction, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-secret")
    model_replies(monkeypatch, "```python\nprint(1)\n```\nFINAL CONCLUSION: One.")
    result = verify.run_agent_detailed(str(instruction), executor=FakeExecutor(),
                                      run_dir=str(tmp_path / "run"))
    for path in Path(result.run_dir).rglob("*"):
        if path.is_file():
            assert "sentinel-secret" not in path.read_text()
            assert "other-secret" not in path.read_text()
    assert not list(Path(result.run_dir).rglob(".env"))


@pytest.mark.parametrize("restricted", [".env", "metadata.json", "curator_manifest.json", "nested/.ssh/key"])
def test_restricted_files_never_enter_candidate_workspace(tmp_path, restricted):
    source = tmp_path / "data"
    source.mkdir()
    path = source / restricted
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("restricted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="Restricted"):
        verify._stage_data(str(source), workspace)
    assert not list(workspace.rglob("*"))


def test_explicit_data_allowlist_does_not_copy_curator_neighbors(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    (source / "safe.csv").write_text("x,y\n1,2")
    (source / "metadata.json").write_text('{"answer": 2}')
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert verify._stage_data(str(source), workspace, ["safe.csv"]) == ["safe.csv"]
    assert sorted(p.name for p in workspace.iterdir()) == ["safe.csv"]


def test_symlink_and_traversal_inputs_are_rejected(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    (source / "alias.txt").symlink_to(secret)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="Symlink"):
        verify._stage_data(str(source), workspace)
    with pytest.raises(ValueError, match="inside"):
        verify._stage_data(str(source), workspace, ["../secret.txt"])


def test_string_false_is_an_error_not_truthy_entailment(monkeypatch):
    model_replies(monkeypatch, '{"result": "false", "rationale": "No"}')
    with pytest.raises(EvaluationError):
        verify._check_claim("wrong", "right", "fake", None)


def test_precision_and_recall_use_correct_denominators_and_keep_numbers(monkeypatch):
    seen = []

    def decompose(text, model, key):
        seen.append(text)
        return ["right", "wrong", "wrong2"] if text.startswith("candidate") else ["right", "other"]

    def entailment(claim, reference, model, key):
        return claim == "right" or (claim == "other" and reference.startswith("candidate"))

    monkeypatch.setattr(verify, "_decompose_into_claims", decompose)
    monkeypatch.setattr(verify, "_check_claim", entailment)
    report = verify.evaluate_conclusion("candidate value 23.1 mm", "reference value 23.1 mm")
    assert report["precision"] == 33.3
    assert report["recall"] == 100.0
    assert report["f1"] == 50.0
    assert seen == ["candidate value 23.1 mm", "reference value 23.1 mm"]
    assert report["evaluation_protocol"] == "conclusion_claim_f1/v2"


def test_empty_claim_set_is_ungradable(monkeypatch):
    model_replies(monkeypatch, '{"subclaims": []}')
    report = verify.evaluate_conclusion("candidate", "reference")
    assert report["pass"] is False
    assert report["evaluation_status"] == "error"
    assert report["f1"] is None


@pytest.mark.parametrize("score", [True, False, "100", 101, -1, 100.0])
def test_invalid_judge_scores_are_not_coerced_or_clamped(monkeypatch, score):
    model_replies(monkeypatch, json.dumps({"score": score, "strengths": "", "weaknesses": "",
                                         "scope_mismatch": "", "rationale": ""}))
    report = verify.evaluate_conclusion_llm_judge("candidate", "reference")
    assert report["evaluation_status"] == "error"
    assert report["score"] is None
    assert report["pass"] is False


def test_judge_truncation_and_missing_fields_are_errors(monkeypatch):
    model_replies(monkeypatch, '{"score": 100}')
    report = verify.evaluate_conclusion_llm_judge("candidate", "reference")
    assert report["evaluation_status"] == "error"
    assert report["scientific_verdict"] == "not_assessed"


def test_parseable_but_truncated_judge_output_is_not_accepted(monkeypatch):
    monkeypatch.setattr(verify, "chat", lambda **kwargs: SimpleNamespace(
        content=json.dumps({"score": 100, "strengths": "", "weaknesses": "",
                            "scope_mismatch": "", "rationale": ""}),
        finish_reason="length"))
    report = verify.evaluate_conclusion_llm_judge("candidate", "reference")
    assert report["evaluation_status"] == "error"
    assert report["pass"] is False


def test_model_roles_are_explicit(monkeypatch):
    calls = model_replies(monkeypatch, '{"result": false, "rationale": "Contradicted"}')
    assert verify._check_claim("23 mm", "2 mm", "model", "control-plane-key") is False
    assert calls[0]["role"] == "judge"
