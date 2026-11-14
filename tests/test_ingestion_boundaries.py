"""Source framing and exact-byte checkpoint boundary regressions."""
import io
import json
import socket

import pytest

from paper2bench import network, paper_context
from paper2bench.bundles import canonical_json
from paper2bench.evidence import EvidenceError, index_text
from paper2bench.llm import ChatResponse


def framing_response(monkeypatch, headers, body=b"pdf"):
    class Response:
        status = 200

        def __init__(self):
            self.stream = io.BytesIO(body)

        def getheaders(self):
            return headers

        def read1(self, count):
            return self.stream.read(count)

        def close(self):
            pass

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(network, "_resolve", lambda *a: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))])
    monkeypatch.setattr(network, "_PinnedConnection", Connection)


@pytest.mark.parametrize("length", ["2", "4"])
def test_public_source_rejects_declared_length_mismatch(monkeypatch, length):
    framing_response(monkeypatch, [("Content-Length", length)])
    with pytest.raises(network.NetworkFetchError, match="declared length"):
        network.safe_fetch("https://example.com/paper.pdf")


@pytest.mark.parametrize("headers", [
    [("Content-Length", "3"), ("content-length", "3")],
    [("Content-Length", "3"), ("Transfer-Encoding", "chunked")],
    [("Transfer-Encoding", "gzip, chunked")],
    [("Transfer-Encoding", "chunked"), ("transfer-encoding", "chunked")],
    [("Content-Length", "+3")],
    [("Content-Length", "٣")],
])
def test_public_source_rejects_ambiguous_response_framing(monkeypatch, headers):
    framing_response(monkeypatch, headers)
    with pytest.raises(network.NetworkFetchError, match="framing|encoding|Content-Length"):
        network.safe_fetch("https://example.com/paper.pdf")


def test_public_source_accepts_exact_length_with_optional_whitespace(monkeypatch):
    framing_response(monkeypatch, [("Content-Length", " 3\t")])
    assert network.safe_fetch("https://example.com/paper.pdf").body == b"pdf"


def test_evidence_checkpoint_publishes_only_reader_bounded_bytes(tmp_path, monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(1)
        return ChatResponse('{"requirements":[],"unresolved_items":[]}', provider="fake", model="fixture")

    monkeypatch.setattr("paper2bench.llm.chat", generate)
    index = index_text("Methods\nUse a fixed experimental schedule.")
    original = tmp_path / "first.json"
    paper_context.extract_requirements(index, checkpoint_path=original)
    limit = len(canonical_json(json.loads(original.read_bytes())))
    monkeypatch.setattr(paper_context, "MAX_CHECKPOINT_BYTES", limit)
    bounded = tmp_path / "bounded.json"
    first = paper_context.extract_requirements(index, checkpoint_path=bounded)
    assert bounded.stat().st_size <= limit
    second = paper_context.extract_requirements(index, checkpoint_path=bounded)
    assert first["model_calls"] == 1 and second["model_calls"] == 0
    assert len(calls) == 2


def test_evidence_intent_limit_fails_before_any_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_context, "MAX_CHECKPOINT_BYTES", 128)
    monkeypatch.setattr("paper2bench.llm.chat", lambda **kwargs: pytest.fail("No dispatch without durable intent"))
    checkpoint = tmp_path / "too-small.json"
    with pytest.raises(EvidenceError, match="checkpoint exceeds size"):
        paper_context.extract_requirements(index_text("Methods\nFixed schedule."), checkpoint_path=checkpoint)
    assert not checkpoint.exists()
