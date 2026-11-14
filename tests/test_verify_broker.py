"""Research-agent/controller target brokerage without worker network access."""
import json

from paper2bench import verify
from paper2bench.bundles import atomic_json, read_json
from paper2bench.executors import ExecutionResult
from paper2bench.llm import ChatResponse
from paper2bench.model_runs import durable_runtime_scope
from paper2bench.runtime_config import get_context


def test_agent_loop_brokers_only_declared_target_and_reuses_known_reply(tmp_path, monkeypatch):
    from paper2bench import model_broker
    config = {"models": {name: {"provider": "ollama", "model": name, "hosting": "local_server",
                                "locality_assurance": "operator_trusted"} for name in ("research_agent", "subject_model")},
              "roles": {"agent": "research_agent"}, "policy": {"data_egress": "local_only"},
              "budgets": {"experiments": {"requests": 10, "input_tokens": 10000, "output_tokens": 1000}},
              "execution": {"model_broker": {"targets": {"subject": "subject_model"},
                                             "max_requests": 1, "max_batch_requests": 1,
                                             "max_tokens_per_request": 32}}}
    calls = []

    def target_chat(*args, **kwargs):
        context = get_context()
        assert context.resolve("agent").model == "subject_model"
        assert kwargs.get("api_key") is None
        reservation = context.ledger.reserve("agent", 100, 32)
        context.ledger.settle(reservation, 10, 1)
        calls.append(kwargs)
        return ChatResponse("4", provider="ollama", model="subject_model", usage_complete=True,
                            finish_reason="stop", metadata={"input_tokens_measured": 10, "output_tokens_measured": 1})

    monkeypatch.setattr(model_broker, "chat", target_chat)
    messages = iter([ChatResponse("```python\n# stage subject request\n```"),
                     ChatResponse("```python\n# read subject response\n```\nFINAL CONCLUSION: Observed 4")])

    def agent_chat(*args, **kwargs):
        assert get_context().resolve("agent").model == "research_agent"
        return next(messages)

    monkeypatch.setattr(verify, "chat", agent_chat)

    class Worker:
        def __init__(self):
            self.calls = 0

        def preflight(self):
            return {}

        def execute(self, request):
            self.calls += 1
            if self.calls == 1:
                atomic_json(request.workspace / "model_requests.json", {"schema_version": 1, "requests": [
                    {"id": "trial1", "target": "subject", "operation": "generate",
                     "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 32, "temperature": 0}]})
            else:
                data = read_json(request.workspace / "model_responses.json")
                assert data["responses"][0]["content"] == "4"
                assert not (request.workspace / ".env").exists()
            return ExecutionResult("succeeded", 0, "done", "", 0.01, evidence={"cleanup_confirmed": True})

    instruction = tmp_path / "instruction.txt"
    instruction.write_text("Query the permitted target model and report observations.")
    with durable_runtime_scope(config, runs_dir=tmp_path / "runs"):
        result = verify.run_agent_detailed(instruction, executor=Worker(), max_turns=2, run_dir=tmp_path / "agent")
    assert result.status == "completed"
    assert result.executions[0]["model_broker"]["model_calls"] == 1
    assert result.executions[1]["model_broker"]["cached_replies"] == 1
    assert len(calls) == 1
    receipts = list((tmp_path / "agent/model_broker/receipts").glob("*.json"))
    assert len(receipts) == 1
    assert read_json(receipts[0])["id"] == "trial1"
