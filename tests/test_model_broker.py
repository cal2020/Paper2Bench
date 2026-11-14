import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paper2bench import model_broker
from paper2bench.bundles import atomic_json, read_json
from paper2bench.journal import RunJournal
from paper2bench.llm import ChatResponse
from paper2bench.model_broker import ModelBroker, ModelBrokerError
from paper2bench.providers import ProviderError
from paper2bench.runtime_config import get_context, runtime_scope


def config(endpoint="http://127.0.0.1:11434/v1", credential_ref=None):
    subject = {"provider": "openai_compatible", "model": "subject-model", "endpoint": endpoint,
               "hosting": "local_server", "locality_assurance": "operator_trusted"}
    if credential_ref:
        subject["credential_ref"] = credential_ref
    return {"models": {"author": dict(subject, model="author-model"), "subject": subject},
            "roles": {"author": "author", "agent": "author"},
            "policy": {"data_egress": "local_only"},
            "budgets": {pool: {"requests": 8, "input_tokens": 10000, "output_tokens": 4096}
                        for pool in ("authoring", "experiments")}}


@contextlib.contextmanager
def active(tmp_path, cfg=None, api_key=None):
    cfg = cfg or config()
    journal = RunJournal(tmp_path / "runs")
    limits = {f"{pool}.{dimension}": value for pool, budget in cfg["budgets"].items() for dimension, value in budget.items()}
    run_id = journal.create({}, limits=limits)
    with runtime_scope(cfg, api_key=api_key, budget_journal=journal, budget_run_id=run_id) as context:
        yield context, journal, run_id


def batch(workspace, identifier="trial_1", **changes):
    workspace.mkdir(exist_ok=True)
    request = {"id": identifier, "target": "subject", "operation": "generate",
               "messages": [{"role": "user", "content": "What is 2 + 2?"}], "max_tokens": 32}
    request.update(changes)
    atomic_json(workspace / "model_requests.json", {"schema_version": 1, "requests": [request]})
    return request


def broker(tmp_path, **kwargs):
    return ModelBroker(target_bindings={"subject": "subject"}, receipt_dir=tmp_path / "private", **kwargs)


def fake_chat(calls, *, content="4", error=None):
    def generate(model, messages, **kwargs):
        context = get_context()
        assert context.config.roles == {"agent": "subject"}
        assert context.api_key is None
        assert kwargs["role"] == "agent"
        reservation = context.ledger.reserve("agent", 100, kwargs["max_tokens"])
        calls.append((model, messages, kwargs))
        if error:
            context.ledger.settle(reservation, None, None)
            raise error
        context.ledger.settle(reservation, 9, 2)
        return ChatResponse(content=content, model=model, provider="openai_compatible", finish_reason="stop",
                            metadata={"input_tokens_measured": 9, "output_tokens_measured": 2}, usage_complete=True)
    return generate


def test_target_mapping_uses_experiment_pool_and_known_reply_replay_is_free(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls))
    workspace = tmp_path / "workspace"
    batch(workspace)
    with active(tmp_path, api_key="author-generic-key") as (_, journal, run_id):
        first = broker(tmp_path).process(workspace)
        second = broker(tmp_path).process(workspace)  # Reopen private receipts, simulating controller restart.
        budget = journal.get(run_id)["budget"]
    assert first["model_calls"] == 1 and second["model_calls"] == 0
    assert second["cached_replies"] == 1
    assert calls[0][0] == "subject-model" and len(calls) == 1
    assert budget["used"]["experiments.requests"] == 1
    assert "authoring.requests" not in budget["used"]
    public = read_json(workspace / "model_responses.json")
    assert public["responses"][0]["cache_semantics"] == "historical_reply_reuse"
    assert "author-generic-key" not in (workspace / "model_responses.json").read_text()


def test_private_intent_precedes_dispatch_and_response_precedes_public_delivery(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    calls = []
    generate = fake_chat(calls)

    def checked(*args, **kwargs):
        receipts = list((tmp_path / "private/receipts").glob("*.json"))
        assert len(receipts) == 1 and read_json(receipts[0])["state"] == "dispatching"
        assert not (workspace / "model_responses.json").exists()
        return generate(*args, **kwargs)

    monkeypatch.setattr(model_broker, "chat", checked)
    with active(tmp_path):
        report = broker(tmp_path).process(workspace)
    assert report["status"] == "complete"
    receipt = read_json(next((tmp_path / "private/receipts").glob("*.json")))
    assert receipt["state"] == "completed" and receipt["response"]["content"] == "4"


def test_public_write_failure_does_not_replay_a_known_target_call(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls))
    original = model_broker.atomic_json

    def fail_public(path, value):
        if path.name == "model_responses.json":
            raise OSError("simulated public delivery failure")
        return original(path, value)

    with active(tmp_path):
        instance = broker(tmp_path)
        monkeypatch.setattr(model_broker, "atomic_json", fail_public)
        with pytest.raises(OSError):
            instance.process(workspace)
        monkeypatch.setattr(model_broker, "atomic_json", original)
        report = instance.process(workspace)
    assert len(calls) == 1 and report["cached_replies"] == 1


def test_unknown_outcome_survives_reopen_without_automatic_resubmit(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls, error=ProviderError("timeout", "safe", outcome_unknown=True)))
    with active(tmp_path) as (_, journal, run_id):
        first = broker(tmp_path).process(workspace)
        second = broker(tmp_path).process(workspace)
        assert journal.get(run_id)["budget"]["reserved"]["experiments.requests"] == 1
    assert first["unknown_outcomes"] == second["unknown_outcomes"] == 1
    assert len(calls) == 1


def test_request_id_cannot_be_rebound_to_a_different_prompt(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls))
    with active(tmp_path):
        instance = broker(tmp_path)
        instance.process(workspace)
        batch(workspace, messages=[{"role": "user", "content": "New experiment"}])
        result = instance.process(workspace)
    assert len(calls) == 1 and result["status"] == "partial"
    assert read_json(workspace / "model_responses.json")["responses"][0]["error_code"] == "request_id_conflict"


@pytest.mark.parametrize("changes", [
    {"target": "unlisted"}, {"operation": "shell"}, {"endpoint": "https://attacker.invalid"},
    {"api_key": "forged"}, {"tools": []}, {"max_tokens": True}, {"max_tokens": 999999},
    {"messages": [{"role": "tool", "content": "tool result"}]},
    {"messages": [{"role": [], "content": "bad"}]}, {"temperature": True},
])
def test_invalid_batch_does_not_dispatch_or_expose_binding_configuration(tmp_path, monkeypatch, changes):
    workspace = tmp_path / "workspace"
    batch(workspace, **changes)
    monkeypatch.setattr(model_broker, "chat", lambda *a, **k: pytest.fail("Invalid batch dispatched"))
    with active(tmp_path):
        result = broker(tmp_path).process(workspace)
    assert result["status"] == "blocked" and result["model_calls"] == 0
    public = (workspace / "model_responses.json").read_text()
    assert "subject-model" not in public and "attacker.invalid" not in public


def test_broker_own_request_limit_is_not_reset_by_reopening(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls))
    with active(tmp_path):
        broker(tmp_path, max_requests=1).process(workspace)
        batch(workspace, identifier="trial_2")
        broker(tmp_path, max_requests=1).process(workspace)
    assert len(calls) == 1
    assert read_json(workspace / "model_responses.json")["responses"][0]["error_code"] == "broker_request_limit"


def test_no_context_or_in_memory_budget_cannot_authorize_calls(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    monkeypatch.setattr(model_broker, "chat", lambda *a, **k: pytest.fail("Non-durable dispatch"))
    instance = ModelBroker(config(), target_bindings={"subject": "subject"}, receipt_dir=tmp_path / "private")
    with pytest.raises(ModelBrokerError, match="durable"):
        instance.process(workspace)
    with runtime_scope(config()), pytest.raises(ModelBrokerError, match="durable"):
        instance.process(workspace)


def test_private_receipts_cannot_be_put_in_candidate_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    batch(workspace)
    with active(tmp_path):
        instance = ModelBroker(target_bindings={"subject": "subject"}, receipt_dir=workspace / "private")
        with pytest.raises(ModelBrokerError, match="overlap"):
            instance.process(workspace)


def test_empty_and_absent_batches_make_no_model_calls(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(model_broker, "chat", lambda *a, **k: pytest.fail("Empty batch dispatched"))
    with active(tmp_path):
        instance = broker(tmp_path)
        assert instance.process(workspace)["status"] == "no_requests"
        atomic_json(workspace / "model_requests.json", {"schema_version": 1, "requests": []})
        result = instance.process(workspace)
    assert result["model_calls"] == 0 and result["status"] == "complete"


def test_response_receipt_integrity_and_operator_target_binding_changes_are_rejected(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls))
    with active(tmp_path):
        instance = broker(tmp_path)
        instance.process(workspace)
        path = next((tmp_path / "private/receipts").glob("*.json"))
        receipt = read_json(path)
        receipt["response"]["content"] = "forged"
        atomic_json(path, receipt)
        with pytest.raises(ModelBrokerError, match="integrity"):
            instance.process(workspace)
        replacement = ModelBroker(target_bindings={"subject": "author"}, receipt_dir=tmp_path / "private")
        with pytest.raises(ModelBrokerError, match="identity_changed"):
            replacement.process(workspace)
    assert len(calls) == 1


@contextlib.contextmanager
def subject_server(*, status=200, content="4", before_reply=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "headers": dict(self.headers), "body": body})
            if before_reply:
                before_reply()
            value = {"model": "subject-model", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_actual_http_uses_only_subject_credential_and_durable_experiments_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECT_MODEL_KEY", "subject-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-forward-host-key")
    workspace = tmp_path / "workspace"
    batch(workspace)
    with subject_server() as (url, calls), active(tmp_path, config(url, "SUBJECT_MODEL_KEY"), api_key="generic-author-key") as (_, journal, run_id):
        report = broker(tmp_path).process(workspace)
        state = journal.get(run_id)
    assert report["status"] == "complete" and len(calls) == 1
    assert calls[0]["body"]["model"] == "subject-model"
    assert calls[0]["headers"]["Authorization"] == "Bearer subject-key"
    assert state["budget"]["used"]["experiments.requests"] == 1
    assert "authoring.requests" not in state["budget"]["used"]
    assert all(value not in (workspace / "model_responses.json").read_text()
               for value in ("subject-key", "generic-author-key", "must-not-forward-host-key"))


def test_actual_unavailable_provider_is_never_replayed_after_unknown_outcome(tmp_path):
    workspace = tmp_path / "workspace"
    batch(workspace)
    with subject_server(status=503) as (url, calls), active(tmp_path, config(url)) as (_, journal, run_id):
        first = broker(tmp_path).process(workspace)
        second = broker(tmp_path).process(workspace)
        assert journal.get(run_id)["budget"]["reserved"]["experiments.requests"] == 1
    assert first["unknown_outcomes"] == second["unknown_outcomes"] == 1 and len(calls) == 1


def test_provider_echoed_target_key_is_redacted_from_candidate_response(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECT_MODEL_KEY", "subject-key")
    workspace = tmp_path / "workspace"
    batch(workspace)
    with subject_server(content="echo subject-key") as (url, _), active(tmp_path, config(url, "SUBJECT_MODEL_KEY")):
        broker(tmp_path).process(workspace)
    public = read_json(workspace / "model_responses.json")["responses"][0]
    assert public["content"] == "echo [redacted]" and public["credential_redaction"]


def test_no_generic_author_key_is_reused_for_native_cloud_target(tmp_path, monkeypatch):
    cfg = config()
    cfg["policy"]["data_egress"] = "allow"
    cfg["models"]["subject"] = {"provider": "openai", "model": "subject-model"}
    workspace = tmp_path / "workspace"
    batch(workspace)
    monkeypatch.setattr(model_broker, "chat", lambda *a, **k: pytest.fail("Author key reused"))
    with active(tmp_path, cfg) as (_, journal, run_id):
        broker(tmp_path).process(workspace)
        assert not journal.get(run_id)["budget"]["used"]
    assert read_json(workspace / "model_responses.json")["responses"][0]["error_code"] == "target_credential_reference_required"


def test_total_output_ceiling_survives_reopen_and_uses_known_usage_refunds(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace, max_tokens=8)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls))  # Two measured output tokens per reply.
    with active(tmp_path):
        broker(tmp_path, max_tokens_per_request=8, max_total_output_tokens=8).process(workspace)
        batch(workspace, identifier="trial_2", max_tokens=8)
        broker(tmp_path, max_tokens_per_request=8, max_total_output_tokens=8).process(workspace)
        assert read_json(workspace / "model_responses.json")["responses"][0]["error_code"] == "broker_output_token_limit"
        batch(workspace, identifier="trial_2", max_tokens=6)
        report = broker(tmp_path, max_tokens_per_request=8, max_total_output_tokens=8).process(workspace)
    assert len(calls) == 2 and report["status"] == "complete"


def test_unknown_reply_keeps_entire_broker_output_reservation(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace, max_tokens=8)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls, error=ProviderError("timeout", "safe", outcome_unknown=True)))
    with active(tmp_path):
        broker(tmp_path, max_tokens_per_request=8, max_total_output_tokens=8).process(workspace)
        batch(workspace, identifier="trial_2", max_tokens=1)
        broker(tmp_path, max_tokens_per_request=8, max_total_output_tokens=8).process(workspace)
    assert len(calls) == 1
    assert read_json(workspace / "model_responses.json")["responses"][0]["error_code"] == "broker_output_token_limit"


def test_public_response_batch_is_bounded_without_truncating_a_subject_answer(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    batch(workspace, max_tokens=8)
    calls = []
    monkeypatch.setattr(model_broker, "chat", fake_chat(calls, content="\n" * 10000))
    with active(tmp_path):
        report = broker(tmp_path, max_batch_requests=1, max_response_bytes=12000).process(workspace)
    public = read_json(workspace / "model_responses.json")
    assert report["status"] == "partial" and len(calls) == 1
    assert public["responses"][0]["error_code"] == "batch_response_limit"
    assert "content" not in public["responses"][0]
    assert (workspace / "model_responses.json").stat().st_size <= 12000


def test_target_instructions_contain_only_public_names_and_file_protocol(tmp_path):
    with active(tmp_path):
        instance = broker(tmp_path)
        instruction = instance.instructions()
    assert instance.names == ("subject",)
    assert "model_requests.json" in instruction and "model_responses.json" in instruction
    assert "subject-model" not in instruction and "http://" not in instruction


def test_shared_batch_deadline_clamps_call_and_prevents_later_dispatch(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    first = batch(workspace)
    second = dict(first, id="trial_2")
    atomic_json(workspace / "model_requests.json", {"schema_version": 1, "requests": [first, second]})
    calls = []
    generate = fake_chat(calls)

    def slow(*args, **kwargs):
        assert 0 < kwargs["timeout_seconds"] <= 0.02
        result = generate(*args, **kwargs)
        time.sleep(0.025)  # The real provider enforces the supplied total deadline.
        return result

    monkeypatch.setattr(model_broker, "chat", slow)
    with active(tmp_path):
        result = broker(tmp_path).process(workspace, timeout_seconds=0.02)
    public = read_json(workspace / "model_responses.json")["responses"]
    assert len(calls) == 1 and result["status"] == "partial"
    assert public[1]["error_code"] == "broker_deadline" and public[1]["request_dispatched"] is False


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_invalid_batch_deadline_cannot_authorize_calls(tmp_path, monkeypatch, timeout):
    workspace = tmp_path / "workspace"
    batch(workspace)
    monkeypatch.setattr(model_broker, "chat", lambda *a, **k: pytest.fail("Invalid deadline dispatched"))
    with active(tmp_path), pytest.raises(ModelBrokerError, match="deadline"):
        broker(tmp_path).process(workspace, timeout_seconds=timeout)
