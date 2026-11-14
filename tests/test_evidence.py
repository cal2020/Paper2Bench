import copy
import json
import re

import pytest

from paper2bench.bundles import atomic_json, canonical_json
from paper2bench.evidence import (EvidenceError, index_paper, index_text, load_index, sha256,
                                 span_text, validate_index, validate_requirements)
from paper2bench.llm import ChatResponse
from paper2bench.paper_context import extract_requirements, select_context


def paper():
    return index_text("--- Page 1 ---\n1 Introduction\nA study of café 🧪.\n\n"
                      "2 Methods\nFit y = 2x + 1. Use a learning rate of 0.1.\n"
                      "--- Page 2 ---\n3 Results\nAccuracy = 0.95 on the held-out sample.\n"
                      "Figure 1 shows results.\n", max_span_chars=64)


def rehash(index):
    index["index_sha256"] = sha256(canonical_json({k: v for k, v in index.items() if k != "index_sha256"}))
    return index


def requirement(index, *, status="stated", quote="Use a learning rate of 0.1."):
    span = next(span for span in index["spans"] if quote in span_text(index, span))
    return {"id": "learning-rate", "kind": "hyperparameter", "text": "Use learning rate 0.1",
            "status": status, "critical": True, "reason": "",
            "evidence": [{"span_id": span["id"], "quote": quote}]}


def fake_extractor(calls):
    def generate(**kwargs):
        calls.append(kwargs)
        user = kwargs["messages"][1]["content"]
        match = re.search(r"\[(span-[0-9a-f]+) page=\d+ section=\w+ excerpt_start=(\d+) excerpt_end=(\d+)\]\n", user)
        assert match
        visible = user[match.end():].split("\n[span-", 1)[0]
        quote = visible[:min(20, int(match.group(3)) - int(match.group(2)))]
        data = {"requirements": [{"kind": "method", "text": "Draft requirement for review", "status": "stated",
                                  "critical": True, "reason": "", "evidence": [{"span_id": match.group(1),
                                    "quote": quote, "start": int(match.group(2)),
                                    "end": int(match.group(2)) + len(quote)}]}], "unresolved_items": []}
        return ChatResponse(json.dumps(data), provider="fake", model="fixture", resolved_model="fixture-r1")
    return generate


def test_source_index_is_stable_and_unicode_offsets_are_explicit():
    first, second = paper(), paper()
    assert first == second
    assert validate_index(first) is first
    assert first["source"]["page_count"] == 2
    assert {span["section"] for span in first["spans"]} >= {"introduction", "methods", "results"}
    for page in first["pages"]:
        spans = [span for span in first["spans"] if span["page"] == page["page"]]
        assert "".join(span_text(first, span) for span in spans) == page["text"]
        for span in spans:
            assert span["utf8_start"] == len(page["text"][:span["start"]].encode("utf-8"))
            assert span["utf16_end"] == len(page["text"][:span["end"]].encode("utf-16-le")) // 2
    assert any(span["features"]["equation_like"] for span in first["spans"])
    assert any(span["features"]["figure_reference"] for span in first["spans"])


def test_overlong_paragraph_is_partitioned_without_dropping_any_text():
    text = "Methods\n" + "abc 🧪 " * 2000 + "\nResults\n" + "result " * 1000
    index = index_text(text, max_span_chars=64)
    assert "".join(span_text(index, span) for span in index["spans"]) == text
    assert max(span["end"] - span["start"] for span in index["spans"]) <= 64
    validate_index(index)


@pytest.mark.parametrize("text", ["--- Page 2 ---\nx\n--- Page 1 ---\ny", "--- Page 1 ---\nx\n--- Page 1 ---\ny",
                                 "unmarked content\n--- Page 1 ---\nx"])
def test_ambiguous_page_markers_are_rejected(text):
    with pytest.raises(EvidenceError):
        index_text(text)


def test_index_corruption_missing_partition_or_encoded_offset_is_rejected():
    original = paper()
    changed = copy.deepcopy(original)
    changed["pages"][0]["text"] += "fabricated"
    with pytest.raises(EvidenceError, match="digest"):
        validate_index(changed)
    changed = copy.deepcopy(original)
    changed["spans"].pop()
    with pytest.raises(EvidenceError, match="omit"):
        validate_index(rehash(changed))
    changed = copy.deepcopy(original)
    changed["spans"][0]["utf16_end"] += 1
    with pytest.raises(EvidenceError, match="offsets"):
        validate_index(rehash(changed))


def test_exact_requirement_quotes_anchor_to_original_source():
    index = paper()
    result = validate_requirements([requirement(index)], index)[0]
    citation = result["evidence"][0]
    assert citation["source_sha256"] == index["source"]["sha256"]
    assert citation["offset_unit"] == "unicode_codepoints"
    assert result["evidence_validation"] == "quote_presence_not_entailment"
    assert result["critical"] is True


@pytest.mark.parametrize("change", ["unknown_span", "fabricated_quote", "wrong_source", "string_critical", "missing_evidence"])
def test_fabricated_or_malformed_requirement_evidence_is_rejected(change):
    index = paper()
    item = requirement(index)
    if change == "unknown_span":
        item["evidence"][0]["span_id"] = "span-unknown"
    elif change == "fabricated_quote":
        item["evidence"][0]["quote"] = "learning rate 0.9"
    elif change == "wrong_source":
        item["evidence"][0]["source_sha256"] = "0" * 64
    elif change == "string_critical":
        item["critical"] = "false"
    else:
        item["evidence"] = []
    with pytest.raises(EvidenceError):
        validate_requirements([item], index)


def test_ambiguous_quote_requires_exact_unicode_offsets():
    index = index_text("Methods\nrepeat repeat\n")
    span = index["spans"][0]
    item = {"id": "repeat", "kind": "method", "text": "Repeated word", "status": "stated",
            "evidence": [{"span_id": span["id"], "quote": "repeat"}]}
    with pytest.raises(EvidenceError, match="ambiguous"):
        validate_requirements([item], index)
    item["evidence"][0].update(start=8, end=14)
    assert validate_requirements([item], index)[0]["evidence"][0]["start"] == 8


def test_uncertain_and_conflicting_requirements_do_not_invent_source_facts():
    index = paper()
    assumed = {"id": "seed", "kind": "hyperparameter", "text": "Use seed 42", "status": "assumed",
               "reason": "Paper does not specify a seed", "evidence": []}
    result = validate_requirements([assumed], index)[0]
    assert result["status"] == "assumed" and result["evidence"] == []
    assumed["reason"] = ""
    with pytest.raises(EvidenceError, match="reason"):
        validate_requirements([assumed], index)
    conflict = requirement(index, status="conflicting")
    conflict["reason"] = "Two different values"
    with pytest.raises(EvidenceError, match="two distinct"):
        validate_requirements([conflict], index)


def test_bounded_retrieval_prioritizes_methods_and_results_without_hiding_omissions():
    index = index_text("Introduction\n" + "background " * 1000 + "\nMethods\n" + "method 🧪 " * 1000 +
                       "\nResults\n" + "accuracy " * 1000, max_span_chars=256)
    selected = select_context(index, "accuracy method", budget_bytes=1500)
    assert len(selected["text"].encode("utf-8")) <= 1500
    assert {item["section"] for item in selected["spans"]} >= {"methods", "results"}
    assert selected["coverage"]["character_fraction"] < 1
    assert selected["coverage"]["incomplete_required_sections"] == ["methods", "results"]
    assert selected["omitted_span_ids"]
    assert "EXCERPTS ONLY" in selected["text"] and selected["readiness"] == "needs_review"


def test_missing_methods_or_results_is_visible_even_when_all_text_fits():
    selected = select_context(index_text("Abstract\nOnly an abstract exists."), budget_bytes=10000)
    assert selected["coverage"]["complete_extracted_text"] is True
    assert selected["coverage"]["missing_required_sections"] == ["methods", "results"]
    assert selected["readiness"] == "needs_review"


def test_tiny_context_and_unicode_remain_bounded():
    index = index_text("Methods\n" + "🧪" * 1000 + "\nResults\n" + "é" * 1000)
    selected = select_context(index, budget_bytes=256)
    assert len(selected["text"].encode("utf-8")) <= 256
    assert selected["coverage"]["character_fraction"] < 1


def test_checkpoint_reuses_historical_artifacts_without_new_calls(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("paper2bench.llm.chat", fake_extractor(calls))
    checkpoint = tmp_path / "checkpoint.json"
    first = extract_requirements(paper(), checkpoint_path=checkpoint)
    second = extract_requirements(paper(), checkpoint_path=checkpoint)
    assert first["model_calls"] == 1 and second["model_calls"] == 0 and len(calls) == 1
    assert second["requirements"] == first["requirements"]
    assert second["chunks"][0]["origin"] == "checkpoint_historical_output"
    assert second["readiness"] == "needs_review" and second["scientific_verdict"] == "not_assessed"


def test_checkpoint_source_model_and_chunk_policy_changes_are_rejected(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("paper2bench.llm.chat", fake_extractor(calls))
    checkpoint = tmp_path / "checkpoint.json"
    extract_requirements(paper(), checkpoint_path=checkpoint)
    with pytest.raises(EvidenceError, match="identity"):
        extract_requirements(index_text("Methods\nDifferent paper"), checkpoint_path=checkpoint)
    with pytest.raises(EvidenceError, match="identity"):
        extract_requirements(paper(), model="other-model", checkpoint_path=checkpoint)
    with pytest.raises(EvidenceError, match="identity"):
        extract_requirements(paper(), chunk_bytes=5000, checkpoint_path=checkpoint)
    assert len(calls) == 1


def test_fabricated_citation_never_becomes_a_valid_requirement(monkeypatch):
    index = paper()
    raw = {"requirements": [{"kind": "method", "text": "Invented", "status": "stated",
                             "critical": True, "evidence": [{"span_id": index["spans"][0]["id"],
                                                               "quote": "does not exist"}]}], "unresolved_items": []}
    monkeypatch.setattr("paper2bench.llm.chat", lambda **k: ChatResponse(json.dumps(raw)))
    report = extract_requirements(index)
    assert report["status"] == "inconclusive" and report["requirements"] == []
    assert report["coverage"]["selected_characters"] == 0


def test_chunk_limit_is_explicit_and_long_span_is_not_silently_truncated(monkeypatch):
    calls = []
    monkeypatch.setattr("paper2bench.llm.chat", fake_extractor(calls))
    index = index_text("Methods\n" + "Method text. " * 1000 + "\nResults\nKnown result.")
    report = extract_requirements(index, chunk_bytes=800, max_chunks=2)
    assert len(calls) == 2 and report["total_chunks"] > 2
    assert report["omitted_chunk_ids"] and report["status"] == "inconclusive"
    assert report["coverage"]["character_fraction"] < 1
    assert all(len(c["messages"][1]["content"].encode("utf-8")) < 1100 for c in calls)


def test_extraction_rejects_citation_outside_the_supplied_excerpt(monkeypatch):
    index = index_text("Methods\n" + "padding " * 600 + "SECRET LATE PHRASE", max_span_chars=6000)
    span = index["spans"][0]
    raw = {"requirements": [{"kind": "method", "text": "Not seen", "status": "stated",
                             "critical": False, "evidence": [{"span_id": span["id"], "quote": "SECRET LATE PHRASE"}]}],
           "unresolved_items": []}
    monkeypatch.setattr("paper2bench.llm.chat", lambda **k: ChatResponse(json.dumps(raw)))
    report = extract_requirements(index, chunk_bytes=600, max_chunks=1)
    assert report["requirements"] == [] and report["chunks"][0]["status"] == "error"


def test_dispatch_intent_is_saved_before_model_call_and_unknown_is_not_retried(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.json"
    calls = []

    def crash(**kwargs):
        calls.append(kwargs)
        recorded = json.loads(checkpoint.read_text())
        assert next(iter(recorded["chunks"].values()))["reason"] == "dispatch_outcome_unknown"
        raise RuntimeError("simulated caller interruption")

    monkeypatch.setattr("paper2bench.llm.chat", crash)
    with pytest.raises(RuntimeError):
        extract_requirements(paper(), checkpoint_path=checkpoint)
    report = extract_requirements(paper(), checkpoint_path=checkpoint)
    assert len(calls) == 1 and report["model_calls"] == 0 and report["status"] == "inconclusive"


def test_provider_failure_stops_automatic_dispatch_of_remaining_chunks(monkeypatch):
    from paper2bench.providers import ProviderError
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise ProviderError("timeout", "fixture", outcome_unknown=True)

    monkeypatch.setattr("paper2bench.llm.chat", fail)
    index = index_text("Methods\n" + "long text " * 1000)
    report = extract_requirements(index, chunk_bytes=700)
    assert len(calls) == 1 and report["omitted_chunk_ids"]


def test_checkpoint_corruption_is_rejected_before_any_new_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("paper2bench.llm.chat", fake_extractor(calls))
    checkpoint = tmp_path / "checkpoint.json"
    extract_requirements(paper(), checkpoint_path=checkpoint)
    data = json.loads(checkpoint.read_text())
    next(iter(data["chunks"].values()))["requirements"] = "not a list"
    atomic_json(checkpoint, data)
    with pytest.raises(EvidenceError, match="chunk schema"):
        extract_requirements(paper(), checkpoint_path=checkpoint)
    assert len(calls) == 1


def test_checkpoint_has_exclusive_owner_before_model_dispatch(tmp_path, monkeypatch):
    from paper2bench.paper_context import _checkpoint_lock
    calls = []
    monkeypatch.setattr("paper2bench.llm.chat", fake_extractor(calls))
    checkpoint = tmp_path / "checkpoint.json"
    with _checkpoint_lock(checkpoint):
        with pytest.raises(EvidenceError, match="already owned"):
            extract_requirements(paper(), checkpoint_path=checkpoint)
    assert calls == []
    assert extract_requirements(paper(), checkpoint_path=checkpoint)["model_calls"] == 1


def test_invalid_model_unicode_is_a_chunk_error_not_source_evidence(monkeypatch):
    raw = {"requirements": [], "unresolved_items": ["\ud800"]}
    monkeypatch.setattr("paper2bench.llm.chat", lambda **k: ChatResponse(json.dumps(raw)))
    report = extract_requirements(paper())
    assert report["status"] == "inconclusive" and not report["requirements"]


def test_profile_context_limit_is_respected_before_dispatch(monkeypatch):
    from paper2bench.runtime_config import runtime_scope
    calls = []
    monkeypatch.setattr("paper2bench.llm.chat", fake_extractor(calls))
    config = {"models": {"local": {"provider": "ollama", "model": "fixture", "hosting": "local_server",
                                   "locality_assurance": "operator_trusted", "context_tokens": 3000}},
              "roles": {"author": "local"}, "policy": {"data_egress": "local_only"}}
    with runtime_scope(config):
        report = extract_requirements(index_text("Methods\n" + "text " * 1000), max_tokens=128, max_chunks=2)
    assert report["effective_chunk_bytes"] < 3000
    assert all(sum(len(m["content"].encode("utf-8")) + 64 for m in call["messages"]) + 128 <= 3000 for call in calls)


def test_index_round_trip_and_real_pdf_source_hash(tmp_path):
    index = paper()
    path = tmp_path / "index.json"
    atomic_json(path, index)
    assert load_index(path) == index
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "paper.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 40), "Methods\nFit y = 2x + 1.\nResults\nThe fit is exact.")
        document.save(pdf)
    indexed = index_paper(pdf)
    assert indexed["source"]["sha256"] == sha256(pdf.read_bytes())
    assert indexed["source"]["kind"] == "pdf"
    assert validate_index(indexed)
