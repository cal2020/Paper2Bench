import contextlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paper2bench.llm import chat, detect_provider
from paper2bench.providers import ProviderError, get_adapter
from paper2bench.runtime_config import BudgetExceeded, ModelRef, runtime_scope, role_scope


@contextlib.contextmanager
def server(response, status=200, headers=None, *, chunk_delay=0, header_delay=0):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            body = response if isinstance(response, bytes) else json.dumps(response).encode()
            if header_delay:
                prefix = f"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                try:
                    for byte in prefix:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(header_delay)
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                if chunk_delay:
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(chunk_delay)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def profile(endpoint, provider="ollama"):
    return {"models": {"local": {"provider": provider, "model": "local-model", "endpoint": endpoint, "hosting": "local_server", "locality_assurance": "operator_trusted"}},
            "roles": {role: "local" for role in ("author", "reference", "agent", "judge")},
            "policy": {"data_egress": "local_only"},
            "budgets": {pool: {"requests": 5, "input_tokens": 10000, "output_tokens": 10000} for pool in ("authoring", "evaluation", "experiments")}}


OLLAMA = {"model": "local-model", "message": {"role": "assistant", "content": "valid result"}, "done": True, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 4}
OPENAI = {"id": "request-1", "model": "resolved-model", "choices": [{"message": {"content": "result"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 3}}


def test_real_loopback_ollama_routing_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")
    with server(OLLAMA) as (endpoint, calls), runtime_scope(profile(endpoint)) as ctx:
        result = chat("gpt-4o", [{"role": "user", "content": "question"}], max_tokens=32)
        assert result.model == "local-model"
        assert result.provider == "ollama"
        assert result.usage == {"input_tokens": 12, "output_tokens": 4, "complete": True}
        assert calls[0]["path"] == "/api/chat"
        assert calls[0]["body"]["options"]["num_predict"] == 32
        assert "Authorization" not in calls[0]["headers"]
        assert ctx.ledger.snapshot()["usage"]["authoring"]["input_tokens"] == 12


def test_compatible_endpoint_and_judge_pool():
    with server(OPENAI) as (endpoint, calls), runtime_scope(profile(endpoint + "/v1", "openai_compatible")) as ctx:
        with role_scope("judge"):
            result = chat("ignored-legacy-default", [{"role": "user", "content": "judge"}], max_tokens=20)
        assert calls[0]["path"] == "/v1/chat/completions"
        assert calls[0]["body"]["max_tokens"] == 20
        assert result.resolved_model == "resolved-model"
        assert result.metadata["budget_pool"] == "evaluation"
        assert "authoring" not in ctx.ledger.snapshot()["usage"]


def test_ollama_requests_the_context_window_used_by_preflight():
    with server(OLLAMA) as (endpoint, calls):
        config = profile(endpoint)
        name = next(iter(config["models"]))
        config["models"][name]["context_tokens"] = 8192
        with runtime_scope(config):
            chat("ignored", [{"role": "user", "content": "question"}], max_tokens=32)
    assert calls[0]["body"]["options"]["num_ctx"] == 8192


def test_anthropic_payload_conversion_without_provider_sdk():
    response = {"id": "msg-1", "model": "claude-test", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}
    with server(response) as (endpoint, calls):
        ref = ModelRef.from_dict({"provider": "anthropic", "model": "anthropic/claude-test", "endpoint": endpoint + "/v1"})
        result = get_adapter("anthropic").generate(ref, [{"role": "system", "content": "policy"}, {"role": "user", "content": "question"}], max_tokens=10, temperature=0, api_key="test-key")
        assert result.content == "onetwo"
        assert calls[0]["body"]["system"] == "policy"
        assert calls[0]["body"]["model"] == "claude-test"
        assert {k.lower(): v for k, v in calls[0]["headers"].items()}["x-api-key"] == "test-key"


def test_unknown_usage_is_not_measured_zero():
    response = dict(OPENAI)
    response.pop("usage")
    with server(response) as (endpoint, _), runtime_scope(profile(endpoint, "openai_compatible")) as ctx:
        result = chat("ignored", [{"role": "user", "content": "q"}], max_tokens=30)
        assert result.input_tokens == 0  # legacy facade only
        assert result.usage["input_tokens"] is None
        assert not result.usage_complete
        used = ctx.ledger.snapshot()["usage"]["authoring"]
        assert used["input_tokens"] > 0 and used["output_tokens"] == 30
        assert used["usage_unknown_requests"] == 1


@pytest.mark.parametrize("response,code", [
    ({"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}, "truncated"),
    ({"choices": [{"message": {"refusal": "no"}, "finish_reason": "stop"}]}, "refusal"),
    ({"choices": [{"message": {"content": "x", "tool_calls": [{}]}, "finish_reason": "stop"}]}, "unsupported_capability"),
    ({"choices": []}, "malformed_response"),
    (b'{"value":NaN}', "malformed_response"),
    (b'not json', "malformed_response"),
    ({**OPENAI, "usage": {"prompt_tokens": True}}, "malformed_response"),
])
def test_malformed_refused_and_truncated_outputs_are_not_valid(response, code):
    with server(response) as (endpoint, _), runtime_scope(profile(endpoint, "openai_compatible")):
        with pytest.raises(ProviderError) as caught:
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10)
        assert caught.value.code == code


@pytest.mark.parametrize("status,code", [(401, "authentication"), (429, "rate_limit"), (404, "unavailable_model"), (503, "unavailable")])
def test_http_errors_are_normalized_without_secret_body(status, code):
    with server({"error": {"message": "a-provider-echoed-secret"}}, status) as (endpoint, calls), runtime_scope(profile(endpoint, "openai_compatible")) as ctx:
        with pytest.raises(ProviderError) as caught:
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10)
        assert caught.value.code == code
        assert "a-provider-echoed-secret" not in str(caught.value)
        assert len(calls) == 1  # never silently replay a billed or ambiguous request
        assert ctx.ledger.snapshot()["usage"]["authoring"]["usage_unknown_requests"] == 1


def test_redirect_cannot_forward_credentials():
    with server(OPENAI) as (destination, forwarded):
        with server({}, 307, {"Location": destination + "/steal"}) as (endpoint, _):
            ref = ModelRef.from_dict({"provider": "openai_compatible", "endpoint": endpoint, "model": "test"})
            with pytest.raises(ProviderError) as caught:
                get_adapter("openai_compatible").generate(ref, [{"role": "user", "content": "q"}], max_tokens=4, temperature=0, api_key="private-key")
            assert caught.value.code == "policy_denied"
        assert forwarded == []


def test_response_bound_and_context_preflight():
    with server(OPENAI) as (endpoint, calls):
        ref = ModelRef.from_dict({"provider": "openai_compatible", "endpoint": endpoint, "model": "test", "max_response_bytes": 10})
        with pytest.raises(ProviderError) as caught:
            get_adapter("openai_compatible").generate(ref, [{"role": "user", "content": "q"}], max_tokens=4, temperature=0)
        assert caught.value.code == "response_too_large"
        cfg = profile(endpoint, "openai_compatible")
        cfg["models"]["local"]["context_tokens"] = 10
        with runtime_scope(cfg), pytest.raises(ProviderError) as caught:
            chat("ignored", [{"role": "user", "content": "long input"}], max_tokens=4)
        assert caught.value.code == "context_overflow"
        assert len(calls) == 1


def test_budget_blocks_dispatch():
    with server(OLLAMA) as (endpoint, calls):
        cfg = profile(endpoint)
        cfg["budgets"]["authoring"]["requests"] = 0
        with runtime_scope(cfg), pytest.raises(BudgetExceeded):
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=4)
        assert calls == []


def test_legacy_routing_is_preserved():
    assert detect_provider("claude-anything") == "anthropic"
    assert detect_provider("anthropic/claude-anything") == "anthropic"
    assert detect_provider("gpt-4o") == "openai"


def test_capabilities_are_honest():
    caps = get_adapter("ollama").capabilities()
    assert caps["text_generation"] == "supported"
    assert caps["tool_calling"] == "unsupported"


def test_control_characters_in_credentials_are_never_echoed():
    ref = ModelRef.from_dict({"provider": "openai", "model": "test"})
    with pytest.raises(ProviderError) as caught:
        get_adapter("openai").generate(ref, [{"role": "user", "content": "q"}], max_tokens=10, temperature=0, api_key="sensitive-key\r\nHeader: value")
    assert caught.value.code == "authentication"
    assert "sensitive-key" not in str(caught.value)


def test_anthropic_cache_usage_and_reasoning_are_normalized():
    response = {"model": "claude-test", "content": [{"type": "thinking", "thinking": "private reasoning"}, {"type": "text", "text": "answer"}], "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2, "cache_creation_input_tokens": 7, "cache_read_input_tokens": 11}}
    with server(response) as (endpoint, _):
        ref = ModelRef.from_dict({"provider": "anthropic", "model": "claude-test", "endpoint": endpoint})
        result = get_adapter("anthropic").generate(ref, [{"role": "user", "content": "q"}], max_tokens=10, temperature=0, api_key="test-key")
        assert result.input_tokens == 23
        assert result.content == "answer"
        assert "private reasoning" not in repr(result)


def test_anthropic_context_truncation_and_openai_developer_payload():
    response = {"model": "claude-test", "content": [{"type": "text", "text": "cut"}], "stop_reason": "model_context_window_exceeded", "usage": {"input_tokens": 5, "output_tokens": 2}}
    with server(response) as (endpoint, _):
        ref = ModelRef.from_dict({"provider": "anthropic", "model": "claude-test", "endpoint": endpoint})
        with pytest.raises(ProviderError) as caught:
            get_adapter("anthropic").generate(ref, [{"role": "user", "content": "q"}], max_tokens=10, temperature=0, api_key="test-key")
        assert caught.value.code == "truncated"
    with server(OPENAI) as (endpoint, calls):
        ref = ModelRef.from_dict({"provider": "openai", "model": "test", "endpoint": endpoint})
        get_adapter("openai").generate(ref, [{"role": "developer", "content": "policy"}, {"role": "user", "content": "q"}], max_tokens=10, temperature=0, api_key="test-key")
        assert calls[0]["body"]["messages"][0]["role"] == "developer"
        assert calls[0]["body"]["max_completion_tokens"] == 10


def test_caller_deadline_stops_trickling_response_and_retains_budget():
    with server(OLLAMA, chunk_delay=0.01) as (endpoint, _), runtime_scope(profile(endpoint)) as ctx:
        started = time.monotonic()
        with pytest.raises(ProviderError) as caught:
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10, timeout_seconds=0.08)
        assert caught.value.code == "timeout" and caught.value.outcome_unknown
        assert time.monotonic() - started < 1
        assert ctx.ledger.snapshot()["usage"]["authoring"]["usage_unknown_requests"] == 1


def test_absolute_deadline_stops_trickling_headers():
    with server(OLLAMA, header_delay=0.01) as (endpoint, calls), runtime_scope(profile(endpoint)) as ctx:
        started = time.monotonic()
        with pytest.raises(ProviderError) as caught:
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10, timeout_seconds=0.08)
        assert caught.value.code == "timeout" and caught.value.outcome_unknown
        assert time.monotonic() - started < 0.5
        assert len(calls) == 1
        assert ctx.ledger.snapshot()["usage"]["authoring"]["usage_unknown_requests"] == 1


def _join_transport_workers():
    # Test cleanup only: production never waits for uncancellable DNS.
    for thread in threading.enumerate():
        if thread.name == "paper2bench-http-transport":
            thread.join(timeout=1)


def test_absolute_deadline_returns_during_slow_dns_and_prevents_late_post(monkeypatch):
    real_resolver = socket.getaddrinfo
    release, entered = threading.Event(), threading.Event()
    def delayed_resolver(*args, **kwargs):
        entered.set()
        release.wait(timeout=2)
        return real_resolver(*args, **kwargs)
    with server(OLLAMA) as (endpoint, calls), runtime_scope(profile(endpoint)) as ctx:
        monkeypatch.setattr(socket, "getaddrinfo", delayed_resolver)
        try:
            started = time.monotonic()
            with pytest.raises(ProviderError) as caught:
                chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10, timeout_seconds=0.08)
            assert entered.is_set()
            assert caught.value.code == "timeout" and caught.value.outcome_unknown
            assert time.monotonic() - started < 0.5
            assert ctx.ledger.snapshot()["usage"]["authoring"]["usage_unknown_requests"] == 1
        finally:
            release.set()
            _join_transport_workers()
        assert calls == []


def test_stuck_dns_workers_are_globally_capped(monkeypatch):
    from paper2bench.providers import http
    real_resolver = socket.getaddrinfo
    release = threading.Event()
    entered = []
    def delayed_resolver(*args, **kwargs):
        entered.append(1)
        release.wait(timeout=2)
        return real_resolver(*args, **kwargs)
    with server(OLLAMA) as (endpoint, calls):
        monkeypatch.setattr(socket, "getaddrinfo", delayed_resolver)
        ref = ModelRef.from_dict({"provider": "ollama", "model": "test", "endpoint": endpoint, "timeout_seconds": 0.04})
        try:
            for _ in range(http._MAX_TRANSPORT_WORKERS):
                with pytest.raises(ProviderError) as caught:
                    http._post_json(ref, endpoint + "/api/chat", {}, {})
                assert caught.value.code == "timeout"
            started = time.monotonic()
            with pytest.raises(ProviderError) as caught:
                http._post_json(ref, endpoint + "/api/chat", {}, {})
            assert caught.value.code == "transport_capacity" and not caught.value.outcome_unknown
            assert time.monotonic() - started < 0.5
            assert len(entered) == http._MAX_TRANSPORT_WORKERS
        finally:
            release.set()
            _join_transport_workers()
        assert calls == []


@pytest.mark.parametrize("response", [
    b'{"choices":[],"choices":[{"message":{"content":"ambiguous"},"finish_reason":"stop"}]}',
    b'{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"extra":1e999}',
])
def test_duplicate_and_overflow_json_responses_fail(response):
    with server(response) as (endpoint, _), runtime_scope(profile(endpoint, "openai_compatible")):
        with pytest.raises(ProviderError) as caught:
            chat("ignored", [{"role": "user", "content": "q"}], max_tokens=10)
        assert caught.value.code == "malformed_response"


@pytest.mark.parametrize("provider,response", [
    ("openai_compatible", {**OPENAI, "choices": [{"message": {"content": "answer"}, "finish_reason": {"stop": True}}]}),
    ("ollama", {**OLLAMA, "done_reason": []}),
    ("ollama", {**OLLAMA, "done_reason": ""}),
    ("anthropic", {"content": [{"type": "text", "text": "answer"}], "stop_reason": []}),
    ("anthropic", {"content": [{"type": "text", "text": "answer"}], "stop_reason": "end_turn", "usage": {"cache_read_input_tokens": True}}),
])
def test_malformed_completion_metadata_uses_normalized_errors(provider, response):
    with server(response) as (endpoint, _):
        ref = ModelRef.from_dict({"provider": provider, "model": "test", "endpoint": endpoint})
        with pytest.raises(ProviderError) as caught:
            get_adapter(provider).generate(ref, [{"role": "user", "content": "q"}], max_tokens=10, temperature=0,
                                           api_key="test-key" if provider == "anthropic" else None)
        assert caught.value.code == "malformed_response"


def test_localhost_resolution_cannot_leak_prompt_or_credentials(monkeypatch):
    from paper2bench.providers import http
    real_resolver = socket.getaddrinfo
    attempted = []
    real_socket = socket.socket

    class RecordingSocket(real_socket):
        def connect(self, address):
            attempted.append(address)
            return super().connect(address)

    def remote_localhost(host, port, *args, **kwargs):
        if host == "localhost":
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("198.51.100.7", port))]
        return real_resolver(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", remote_localhost)
    monkeypatch.setattr(socket, "socket", RecordingSocket)
    ref = ModelRef.from_dict({"provider": "ollama", "model": "test", "endpoint": "http://localhost:11434"})
    with pytest.raises(ProviderError) as caught:
        http._post_json(ref, ref.endpoint + "/api/chat", {"prompt": "private prompt"}, {"Authorization": "Bearer private-key"})
    assert caught.value.code == "policy_denied"
    assert attempted == []


def test_localhost_connect_uses_validated_addresses_once(monkeypatch):
    real_resolver = socket.getaddrinfo
    resolutions = []
    with server(OLLAMA) as (endpoint, calls):
        def loopback_localhost(host, port, *args, **kwargs):
            if host == "localhost":
                resolutions.append(host)
                assert len(resolutions) == 1, "validated hostname must not be re-resolved during connection"
                return real_resolver("127.0.0.1", port, *args, **kwargs)
            return real_resolver(host, port, *args, **kwargs)
        monkeypatch.setattr(socket, "getaddrinfo", loopback_localhost)
        endpoint = endpoint.replace("127.0.0.1", "localhost")
        with runtime_scope(profile(endpoint)):
            result = chat("ignored", [{"role": "user", "content": "question"}], max_tokens=10)
        assert result.content == "valid result"
        assert len(calls) == 1 and resolutions == ["localhost"]
