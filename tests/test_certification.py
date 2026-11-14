import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paper2bench import certification
from paper2bench.bundles import read_json
from paper2bench.journal import RunJournal
from paper2bench.runtime_config import runtime_scope


DIGEST = "a" * 64


def profile(endpoint="http://127.0.0.1:11434", *, provider="ollama", roles=None):
    return {"models": {"local": {"provider": provider, "model": "tiny:latest", "endpoint": endpoint,
                                   "hosting": "local_server", "locality_assurance": "operator_trusted"}},
            "roles": roles or {"author": "local"}, "policy": {"data_egress": "local_only"},
            "budgets": {pool: {"requests": 5, "input_tokens": 10000, "output_tokens": 1000}
                        for pool in ("authoring", "evaluation", "experiments")}}


@contextlib.contextmanager
def endpoint(*, content='{"answer":4}', response_model="tiny:latest", status=200,
             digests=None, inventory_status=200, redirect=None):
    calls = []
    digests = list(digests or [DIGEST, DIGEST])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            if redirect:
                self.send_header("Location", redirect)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            calls.append(("GET", self.path, None, dict(self.headers)))
            digest = digests.pop(0) if len(digests) > 1 else digests[0]
            self.send(inventory_status, {"models": [{"name": "tiny:latest", "model": "tiny:latest",
                       "digest": digest, "size": 123, "details": {"quantization_level": "Q4_K_M", "format": "gguf"}}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, body, dict(self.headers)))
            self.send(status, {"model": response_model, "message": {"role": "assistant", "content": content},
                       "done": True, "done_reason": "stop", "prompt_eval_count": 8, "eval_count": 7})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_real_protocol_fixture_preserves_identity_and_reports_only_tested_capability(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-local-server")
    with endpoint() as (url, calls):
        output = tmp_path / "receipt.json"
        report = certification.certify_runtime(profile(url), output_path=output)
    assert report["status"] == "passed"
    assert report["model_calls"] == report["probe_calls_attempted"] == 1
    assert [method for method, *_ in calls] == ["GET", "POST", "GET"]
    probe = report["probes"][0]
    assert probe["model_identity_status"] == "digest_stable_across_probe"
    assert probe["local_identity_before"]["digest"] == "sha256:" + DIGEST
    assert probe["fixture_result"] == {"answer": 4}
    assert probe["capabilities"]["text_generation"] == "tested_pass"
    assert probe["capabilities"]["structured_outputs"] == "unsupported"
    assert probe["usage"]["complete"] is True
    assert all("Authorization" not in headers for *_, headers in calls)
    assert calls[1][2]["options"]["num_predict"] == 32
    assert read_json(output) == report
    assert output.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("content", ['{"answer":5}', '{"answer":"4"}', '{"answer":true}',
                                      '{"answer":4,"extra":"data"}', 'answer: 4',
                                      '{"answer":4,"answer":4}'])
def test_invalid_fixture_is_failed_without_retry_or_untrusted_text_retention(content):
    with endpoint(content=content) as (url, calls):
        report = certification.certify_runtime(profile(url))
    assert report["status"] == "failed"
    assert len([call for call in calls if call[0] == "POST"]) == 1
    assert "fixture_result" not in report["probes"][0]
    assert "content" not in report["probes"][0]


def test_model_digest_drift_cannot_pass():
    with endpoint(digests=[DIGEST, "b" * 64]) as (url, _):
        report = certification.certify_runtime(profile(url))
    assert report["status"] == "failed"
    assert report["probes"][0]["error_code"] == "model_identity_drift"


def test_wrong_resolved_local_model_cannot_pass():
    with endpoint(response_model="other:latest") as (url, _):
        report = certification.certify_runtime(profile(url))
    assert report["status"] == "failed"
    assert report["probes"][0]["error_code"] == "model_identity_mismatch"


def test_inventory_redirect_is_blocked_and_never_probes_generation():
    with endpoint(inventory_status=307, redirect="http://127.0.0.1:9/steal") as (url, calls):
        report = certification.certify_runtime(profile(url))
    assert report["status"] == "blocked"
    assert report["model_calls"] == report["probe_calls_attempted"] == 0
    assert all(call[0] == "GET" for call in calls)


def test_cloud_probe_requires_explicit_authorization_and_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = {"models": {"cloud": {"provider": "openai", "model": "model"}},
           "roles": {"author": "cloud"},
           "budgets": {"authoring": {"requests": 1, "input_tokens": 10000, "output_tokens": 128}}}
    monkeypatch.setattr(certification, "chat", lambda *a, **k: pytest.fail("Cloud request dispatched"))
    denied = certification.certify_runtime(cfg)
    missing = certification.certify_runtime(cfg, allow_cloud=True)
    assert denied["status"] == missing["status"] == "blocked"
    assert denied["model_calls"] == missing["model_calls"] == 0
    assert denied["probes"][0]["error_code"] == "policy_denied"
    assert missing["probes"][0]["error_code"] == "invalid_runtime_config"


def test_active_durable_budget_is_reused_instead_of_reset(tmp_path):
    with endpoint() as (url, calls):
        cfg = profile(url)
        journal = RunJournal(tmp_path / "runs")
        limits = {f"{pool}.{key}": value for pool, budget in cfg["budgets"].items() for key, value in budget.items()}
        run = journal.create({}, limits=limits)
        with runtime_scope(cfg, budget_journal=journal, budget_run_id=run):
            report = certification.certify_runtime(cfg)
    assert report["status"] == "passed"
    assert report["budget"]["scope"] == "durable_run"
    assert journal.get(run)["budget"]["used"]["authoring.requests"] == 1


def test_other_active_runtime_cannot_be_replaced(monkeypatch):
    cfg = profile()
    other = profile("http://127.0.0.1:9999")
    monkeypatch.setattr(certification, "chat", lambda *a, **k: pytest.fail("Request dispatched"))
    with runtime_scope(cfg), pytest.raises(ValueError, match="active runtime"):
        certification.certify_runtime(other)


def test_exhausted_budget_blocks_generation_dispatch():
    with endpoint() as (url, calls):
        cfg = profile(url)
        cfg["budgets"]["authoring"]["requests"] = 0
        report = certification.certify_runtime(cfg)
    assert report["status"] == "blocked"
    assert report["model_calls"] == 0
    assert report["probes"][0]["request_dispatched"] is False
    assert all(call[0] == "GET" for call in calls)


@pytest.mark.parametrize("tokens", [0, 129, True, 1.5])
def test_invalid_probe_budget_is_rejected_without_calls(tokens, monkeypatch):
    monkeypatch.setattr(certification, "chat", lambda *a, **k: pytest.fail("Request dispatched"))
    with pytest.raises(ValueError):
        certification.certify_runtime(profile(), probe_max_tokens=tokens)


def test_existing_receipt_cannot_be_overwritten_after_new_calls(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("{}")
    monkeypatch.setattr(certification, "chat", lambda *a, **k: pytest.fail("Request dispatched"))
    with pytest.raises(ValueError, match="already exists"):
        certification.certify_runtime(profile(), output_path=path)


def test_wrong_role_is_rejected_without_calls():
    with pytest.raises(ValueError, match="roles"):
        certification.certify_runtime(profile(), roles=["judge"])
