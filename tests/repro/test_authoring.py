from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from paper2bench.repro.authoring import (
    AuthoringError,
    EvidenceValidationError,
    SourceError,
    draft_experiment,
    ingest_source,
    validate_evidence,
)
from paper2bench.repro.models import canonical_bytes

IMAGE = "sha256:" + "a" * 64
SOURCE = "The nearest centroid method attains accuracy 1.0 on this task."
CODE = (
    "import json, sys\ndata = json.load(sys.stdin)\nprint(json.dumps({'ids': [], 'labels': []}))\n"
)


class MemoryArtifacts:
    def __init__(self):
        self.data = {}

    def put(self, value):
        key = hashlib.sha256(value).hexdigest()
        self.data[key] = value
        return key

    def get(self, key):
        value = self.data[key]
        assert hashlib.sha256(value).hexdigest() == key
        return value

    def get_json(self, key):
        return json.loads(self.get(key))


class FakeProfile(BaseModel):
    name: str = "local"
    base_url: str = "http://127.0.0.1:11434/v1"
    location: str = "local"
    api_key_env: str = "EXAMPLE_API_KEY"
    timeout: float = 30.0
    # Even an unexpected profile implementation cannot export this field.
    api_key: str = "never-export-this-secret"


class FakeClient:
    def __init__(self, response, *, error=None, finish_reason="stop"):
        self.response = response
        self.calls = []
        self.error = error
        self.profile = FakeProfile()
        self.finish_reason = finish_reason

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            text=self.response,
            model="local-model-revision",
            request_id="request-123",
            fingerprint="fingerprint-1",
            usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            finish_reason=self.finish_reason,
        )


def response(**changes):
    value = {
        "title": "Source-grounded nearest centroid",
        "evidence": [{"start": 0, "end": len(SOURCE), "quote": SOURCE}],
        "assumptions": ["Euclidean distance is inferred"],
        "unresolved": [],
        "claim": {"target": 1.0, "tolerance": 0.1},
        "implementation": CODE,
    }
    value.update(changes)
    return json.dumps(value)


def inputs():
    artifacts = MemoryArtifacts()
    source = artifacts.put(SOURCE.encode())
    dataset = artifacts.put(
        canonical_bytes(
            {
                "train": [
                    {"id": "train-left", "x": [0.0], "y": "left"},
                    {"id": "train-right", "x": [2.0], "y": "right"},
                ],
                "test": [{"id": "hidden-case-456", "x": [999.0], "y": "right"}],
            }
        )
    )
    return artifacts, source, dataset


def draft(client, *, image=IMAGE, max_output_tokens=8192):
    artifacts, source, dataset = inputs()
    result = draft_experiment(
        source,
        dataset,
        artifacts,
        client,
        "requested-model",
        True,
        image,
        max_output_tokens,
    )
    return artifacts, result


def test_ingest_preserves_original_and_normalizes_text(tmp_path):
    original = "\ufeffCafe\u0301\r\nResult: accuracy 1.0\r".encode()
    path = tmp_path / "paper.md"
    path.write_bytes(original)
    artifacts = MemoryArtifacts()
    receipt = ingest_source(path, artifacts)
    assert artifacts.get(receipt["original_sha256"]) == original
    assert artifacts.get(receipt["source_sha256"]).decode() == "Café\nResult: accuracy 1.0\n"
    assert receipt["source_sha256"] != receipt["original_sha256"]
    assert receipt["media_type"] == "text/markdown"


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("paper.csv", b"text"),
        ("paper.txt", b""),
        ("paper.txt", b"\xff"),
        ("paper.md", b" \n\t"),
        ("paper.txt", b"a\0b"),
        ("paper.pdf", b"not a pdf"),
    ],
)
def test_ingest_rejects_unsupported_or_invalid_source(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(SourceError):
        ingest_source(path, MemoryArtifacts())


def test_ingest_refuses_large_file_without_storing_it(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    artifacts = MemoryArtifacts()
    with pytest.raises(SourceError, match="16 MiB"):
        ingest_source(path, artifacts)
    assert artifacts.data == {}


@pytest.mark.docker
def test_ingest_pdf_marks_pages_and_rejects_missing_text(tmp_path):
    import os

    parser_image = os.environ.get("PAPER2BENCH_PARSER_IMAGE")
    if not parser_image:
        pytest.skip("set PAPER2BENCH_PARSER_IMAGE for isolated PDF ingestion")
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 100 Td (Research accuracy 1.0) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "paper.pdf"
    writer.write(path)
    artifacts = MemoryArtifacts()
    receipt = ingest_source(path, artifacts, parser_image=parser_image)
    text = artifacts.get(receipt["source_sha256"]).decode()
    assert "--- Page 1 ---" in text and "--- Page 2 ---" in text
    assert "Research accuracy 1.0" in text
    assert receipt["page_count"] == 2
    assert len(receipt["warnings"]) == 1

    blank = PdfWriter()
    blank.add_blank_page(width=200, height=200)
    blank.write(path)
    with pytest.raises(SourceError, match="Isolated PDF extraction"):
        ingest_source(path, artifacts, parser_image=parser_image)


def test_draft_records_real_provenance_and_keeps_test_data_private():
    client = FakeClient(response())
    artifacts, spec = draft(client)
    assert len(client.calls) == 1
    assert artifacts.get(spec.workload.implementation_sha256).decode() == CODE
    assert spec.workload.kind == "python" and spec.workload.image == IMAGE
    assert spec.policy.local_only is True
    assert spec.policy.max_provider_calls == 0
    sent = client.calls[0]
    assert sent["model"] == "requested-model" and sent["local_only"] is True
    payload = json.loads(sent["messages"][1]["content"])
    assert set(payload) == {"source_text", "training_examples", "feature_count", "allowed_labels"}
    assert "hidden-case-456" not in sent["messages"][1]["content"]
    assert "999" not in sent["messages"][1]["content"]
    receipt = artifacts.get_json(spec.authoring_receipt_sha256)
    assert receipt["requested_model"] == "requested-model"
    assert receipt["actual_model"] == "local-model-revision"
    assert receipt["request_id"] == "request-123"
    assert receipt["usage"]["total_tokens"] == 300
    assert receipt["provider_calls"] == 1
    assert receipt["implementation_fidelity"] == "unverified"
    assert "never-export-this-secret" not in json.dumps(receipt)
    assert receipt["implementation_sha256"] == spec.workload.implementation_sha256
    assert artifacts.get_json(receipt["prompt_sha256"]) == sent["messages"]


def test_unresolved_is_editable_but_blocks_freeze():
    _, spec = draft(FakeClient(response(unresolved=["Missing preprocessing setting"])))
    assert spec.unresolved == ("Missing preprocessing setting",)
    validate_evidence(spec, SOURCE, require_resolved=False)
    with pytest.raises(EvidenceValidationError, match="Unresolved"):
        validate_evidence(spec, SOURCE)


def test_wrong_source_quote_is_rejected_before_code_storage():
    client = FakeClient(response(evidence=[{"start": 0, "end": 4, "quote": "Fake"}]))
    with pytest.raises(EvidenceValidationError):
        draft(client)


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{}\n```",
        "[]",
        "{}",
        '{"title":"a","title":"b"}',
        response(implementation="def broken("),
        response(unsupported="field"),
        response(claim={"target": float("nan")}),
    ],
)
def test_invalid_authoring_outputs_are_not_repaired_or_retried(text):
    client = FakeClient(text)
    with pytest.raises(AuthoringError):
        draft(client)
    assert len(client.calls) == 1


def test_provider_unknown_outcome_propagates_without_retry():
    error = TimeoutError("unknown provider outcome")
    client = FakeClient("", error=error)
    with pytest.raises(TimeoutError) as caught:
        draft(client)
    assert caught.value is error
    assert len(client.calls) == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_truncated_or_filtered_response_is_not_published(finish_reason):
    with pytest.raises(AuthoringError, match="truncated or filtered"):
        draft(FakeClient(response(), finish_reason=finish_reason))


@pytest.mark.parametrize("image", ["python:3.12", "latest", "sha256:bad"])
def test_unpinned_image_rejected_before_billable_call(image):
    client = FakeClient(response())
    with pytest.raises(AuthoringError):
        draft(client, image=image)
    assert client.calls == []


@pytest.mark.parametrize("limit", [0, 32769, True, 1.5])
def test_bad_token_cap_is_rejected_before_call(limit):
    client = FakeClient(response())
    with pytest.raises(AuthoringError):
        draft(client, max_output_tokens=limit)
    assert client.calls == []


def test_oversized_code_and_response_are_rejected():
    with pytest.raises(AuthoringError):
        draft(FakeClient(response(implementation="#" + "é" * (128 * 1024))))
    with pytest.raises(AuthoringError):
        draft(FakeClient(" " * (512 * 1024 + 1)))
