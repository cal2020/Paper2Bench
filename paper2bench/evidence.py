"""Versioned source indexes and exact-quote evidence validation.

Offsets are Unicode code points unless explicitly labelled UTF-8 bytes or
UTF-16 code units. Matching a quote proves its presence in extracted text;
it does not prove that the quoted passage entails a generated requirement.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from .bundles import canonical_json, read_json

MAX_TEXT_CHARS = 8 * 1024 * 1024
MAX_INDEX_BYTES = 48 * 1024 * 1024
MAX_SPANS = 20000
SECTION_KINDS = frozenset({"unknown", "abstract", "introduction", "background", "methods",
                           "experiments", "results", "discussion", "conclusion", "limitations",
                           "references", "appendix"})
REQUIREMENT_KINDS = frozenset({"research_question", "method", "dataset", "metric", "hyperparameter",
                              "resource", "result", "constraint", "other"})
REQUIREMENT_STATES = frozenset({"stated", "inferred", "assumed", "unresolved", "conflicting"})
_PAGE = re.compile(r"(?m)^--- Page ([1-9][0-9]{0,3}) ---[ \t]*\r?\n")
_HEADING = re.compile(
    r"^\s*(?:(?:[0-9]+(?:\.[0-9]+)*\.?|[IVX]+)\s+)?"
    r"(abstract|introduction|background|related work|materials and methods|methods|methodology|method|"
    r"experimental setup|experiments|evaluation|results|discussion|conclusions|conclusion|limitations|"
    r"references|bibliography|appendix|appendices)(?:\s*[:.\-]\s*|\s+)?([^\n]*)$", re.I)
_SECTION_MAP = {"related work": "background", "materials and methods": "methods", "method": "methods",
                "methodology": "methods", "experimental setup": "experiments", "evaluation": "experiments",
                "conclusions": "conclusion", "bibliography": "references", "appendices": "appendix"}


class EvidenceError(ValueError):
    """Malformed, inconsistent or out-of-bounds source evidence."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _hash(value):
    return sha256(canonical_json(value))


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unicode_text(value):
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
        return True
    except UnicodeError:
        return False


def _index_payload(index):
    return {key: value for key, value in index.items() if key != "index_sha256"}


def _heading(line):
    stripped = line.strip()
    if not stripped or len(stripped) > 120 or len(stripped.split()) > 14:
        return None
    match = _HEADING.fullmatch(stripped)
    if not match:
        return None
    # Prose such as "Methods may vary substantially." is not a heading.
    suffix = match.group(2).strip()
    if suffix and (suffix.endswith(('.', '?', '!')) or len(suffix.split()) > 8):
        return None
    kind = match.group(1).lower()
    return _SECTION_MAP.get(kind, kind), stripped


def _chunks(text, start, end, maximum):
    cursor = start
    while cursor < end:
        stop = min(end, cursor + maximum)
        if stop < end:
            # Retain all characters: boundaries are slices, never stripped text.
            floor = cursor + maximum // 2
            boundary = text.rfind("\n\n", floor, stop)
            if boundary >= floor:
                stop = boundary + 2
            else:
                boundary = text.rfind("\n", floor, stop)
                if boundary < floor:
                    boundary = text.rfind(" ", floor, stop)
                if boundary >= floor:
                    stop = boundary + 1
        yield cursor, stop
        cursor = stop


def index_text(text, *, source_sha256=None, max_span_chars=6000, page_count=None,
               source_kind="text", parser="provided_page_marked_text"):
    """Index complete extracted text without discarding overlong paragraphs.

    Page markers use the format emitted by the shared PDF parser. Section and
    equation/table detection are explicitly heuristic, not layout recovery.
    """
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
        raise EvidenceError("Source text must be bounded Unicode text")
    if type(max_span_chars) is not int or not 64 <= max_span_chars <= 32000:
        raise EvidenceError("max_span_chars must be an integer in [64, 32000]")
    if not _unicode_text(source_kind) or len(source_kind) > 80 or not _unicode_text(parser) or len(parser) > 128:
        raise EvidenceError("Source kind and parser identity must be bounded Unicode text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeError as error:
        raise EvidenceError("Source text contains invalid Unicode surrogates") from error
    source_sha256 = source_sha256 or sha256(encoded)
    if not _digest(source_sha256):
        raise EvidenceError("Source identity must be a SHA-256 hex digest")
    markers = list(_PAGE.finditer(text))
    pages = []
    if markers:
        if text[:markers[0].start()].strip():
            raise EvidenceError("Unmarked text precedes the first page marker")
        previous = 0
        for offset, marker in enumerate(markers):
            number = int(marker.group(1))
            if number <= previous or number > 2000:
                raise EvidenceError("Page markers must be unique, increasing and bounded")
            previous = number
            start = marker.end()
            end = markers[offset + 1].start() if offset + 1 < len(markers) else len(text)
            content = text[start:end]
            pages.append({"page": number, "text": content, "document_start": start,
                          "document_end": end, "sha256": sha256(content.encode("utf-8"))})
    else:
        pages = [{"page": 1, "text": text, "document_start": 0, "document_end": len(text),
                  "sha256": sha256(encoded)}]
    observed_pages = max((page["page"] for page in pages), default=0)
    if page_count is None:
        page_count = observed_pages
    if type(page_count) is not int or not observed_pages <= page_count <= 2000:
        raise EvidenceError("Physical page count conflicts with extracted page markers")
    spans, section, title = [], "unknown", "Unclassified extracted text"
    for page in pages:
        content = page["text"]
        boundaries = [(0, section, title)]
        position = 0
        for line in content.splitlines(keepends=True):
            identified = _heading(line)
            if identified:
                section, title = identified
                if position == 0:
                    boundaries[0] = (0, section, title)
                else:
                    boundaries.append((position, section, title))
            position += len(line)
        byte_offset = utf16_offset = 0
        for offset, (start, current_section, current_title) in enumerate(boundaries):
            end = boundaries[offset + 1][0] if offset + 1 < len(boundaries) else len(content)
            for left, right in _chunks(content, start, end, max_span_chars):
                quote = content[left:right]
                quote_bytes = quote.encode("utf-8")
                quote_utf16 = len(quote.encode("utf-16-le")) // 2
                identity = {"source": source_sha256, "page": page["page"], "start": left,
                            "end": right, "sha256": sha256(quote_bytes)}
                spans.append({"id": "span-" + _hash(identity)[:24], "page": page["page"],
                              "start": left, "end": right,
                              "document_start": page["document_start"] + left,
                              "document_end": page["document_start"] + right,
                              "utf8_start": byte_offset, "utf8_end": byte_offset + len(quote_bytes),
                              "utf16_start": utf16_offset, "utf16_end": utf16_offset + quote_utf16,
                              "sha256": identity["sha256"], "section": current_section,
                              "section_title": current_title,
                              "features": {"equation_like": bool(re.search(r"(?m)^.{0,160}[=∑∫≈≤≥].{0,160}$", quote)),
                                           "table_like": bool(re.search(r"(?m)^.{0,100}(?:\|.+\||\S {3,}\S).{0,100}$", quote)),
                                           "figure_reference": bool(re.search(r"\b(?:figure|fig\.)\s*\d", quote, re.I))}})
                byte_offset += len(quote_bytes)
                utf16_offset += quote_utf16
                if len(spans) > MAX_SPANS:
                    raise EvidenceError("Evidence span count exceeds supported bounds")
    index = {"schema_version": 1,
             "source": {"kind": source_kind, "sha256": source_sha256, "text_sha256": sha256(encoded),
                        "text_bytes": len(encoded), "text_characters": len(text), "page_count": page_count},
             "extraction": {"parser": parser, "offset_unit": "unicode_codepoints",
                            "page_number_origin": "parser_text_markers", "section_detection": "heuristic",
                            "ocr_performed": False,
                            "limitations": ["Text extraction does not preserve all figures, tables or equations",
                                            "Quote presence does not establish entailment or method fidelity",
                                            "Page markers and section labels are not a layout proof"]},
             "pages": pages, "spans": spans}
    index["index_sha256"] = _hash(index)
    return index


def index_paper(pdf_path, *, limits=None, max_span_chars=6000):
    """Hash and parse the same bounded PDF bytes using the shared parser worker."""
    from .pdf_utils import PDFLimits, PDFExtractionError, _parse_in_worker, parser_limits_profile
    limits = limits or PDFLimits()
    if limits.require_hard_memory_limit and not parser_limits_profile()["hard_memory"]:
        raise PDFExtractionError("Hard parser memory limits require the supported Linux worker profile")
    path = Path(pdf_path)
    if path.is_symlink():
        raise EvidenceError("PDF source must not be a symlink")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limits.max_file_bytes:
                raise EvidenceError("PDF must be a regular file within its byte limit")
            data = stream.read(limits.max_file_bytes + 1)
    except OSError as error:
        raise EvidenceError("PDF source cannot be read safely") from error
    if len(data) > limits.max_file_bytes or not data.startswith(b"%PDF-"):
        raise EvidenceError("PDF source is invalid or exceeds its byte limit")
    parsed = _parse_in_worker(data, limits)
    return index_text(parsed["text"], source_sha256=sha256(data), max_span_chars=max_span_chars,
                      page_count=parsed["pages"], source_kind="pdf", parser="bounded_pdf_subprocess")


def load_index(path):
    return validate_index(read_json(path, limit=MAX_INDEX_BYTES))


def validate_index(index):
    if not isinstance(index, dict) or type(index.get("schema_version")) is not int or index["schema_version"] != 1:
        raise EvidenceError("Unsupported evidence index schema")
    try:
        actual_digest = _hash(_index_payload(index))
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise EvidenceError("Evidence index is not canonical finite Unicode data") from error
    if not _digest(index.get("index_sha256")) or actual_digest != index["index_sha256"]:
        raise EvidenceError("Evidence index digest mismatch")
    source, pages, spans = index.get("source"), index.get("pages"), index.get("spans")
    extraction = index.get("extraction")
    if (not isinstance(source, dict) or not _digest(source.get("sha256"))
            or not _digest(source.get("text_sha256"))
            or type(source.get("text_bytes")) is not int or source["text_bytes"] < 0
            or type(source.get("text_characters")) is not int or not 0 <= source["text_characters"] <= MAX_TEXT_CHARS
            or not isinstance(source.get("kind"), str) or len(source["kind"]) > 80
            or not isinstance(pages, list) or not 1 <= len(pages) <= 2000
            or not isinstance(spans, list) or len(spans) > MAX_SPANS):
        raise EvidenceError("Invalid source/page/span schema")
    if (not isinstance(extraction, dict) or extraction.get("offset_unit") != "unicode_codepoints"
            or not isinstance(extraction.get("parser"), str) or len(extraction["parser"]) > 128
            or extraction.get("section_detection") != "heuristic" or type(extraction.get("ocr_performed")) is not bool
            or not isinstance(extraction.get("limitations"), list) or len(extraction["limitations"]) > 32
            or any(not isinstance(item, str) or len(item) > 2000 for item in extraction["limitations"])):
        raise EvidenceError("Invalid extraction metadata")
    page_map, previous, total_chars, document_end = {}, 0, 0, 0
    for page in pages:
        if not isinstance(page, dict) or type(page.get("page")) is not int or not previous < page["page"] <= 2000:
            raise EvidenceError("Invalid page identity")
        previous = page["page"]
        content = page.get("text")
        if not isinstance(content, str) or sha256(content.encode("utf-8")) != page.get("sha256"):
            raise EvidenceError("Page content hash mismatch")
        total_chars += len(content)
        if total_chars > MAX_TEXT_CHARS:
            raise EvidenceError("Index source text exceeds bounds")
        if (type(page.get("document_start")) is not int or type(page.get("document_end")) is not int
                or page["document_start"] < document_end or page["document_end"] - page["document_start"] != len(content)):
            raise EvidenceError("Invalid page document offsets")
        document_end = page["document_end"]
        page_map[page["page"]] = page
    if type(source.get("page_count")) is not int or not previous <= source["page_count"] <= 2000:
        raise EvidenceError("Invalid physical page count")
    if document_end != source["text_characters"] or source["text_bytes"] < sum(len(page["text"].encode("utf-8")) for page in pages):
        raise EvidenceError("Source length metadata conflicts with page content")
    seen, cursors, byte_offsets, utf16_offsets = set(), {}, {}, {}
    previous_span_position = (-1, -1)
    for span in spans:
        if not isinstance(span, dict) or type(span.get("page")) is not int or span["page"] not in page_map:
            raise EvidenceError("Span refers to an unknown page")
        page = page_map[span["page"]]
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(page["text"]) or end - start > 32000:
            raise EvidenceError("Invalid span offsets")
        if (span["page"], start) <= previous_span_position:
            raise EvidenceError("Evidence spans must be in document order")
        previous_span_position = (span["page"], start)
        if start != cursors.get(span["page"], 0):
            raise EvidenceError("Evidence spans must form ordered complete page partitions")
        quote = page["text"][start:end]
        identity = {"source": source["sha256"], "page": span["page"], "start": start,
                    "end": end, "sha256": sha256(quote.encode("utf-8"))}
        ident = "span-" + _hash(identity)[:24]
        if span.get("id") != ident or ident in seen or span.get("sha256") != identity["sha256"]:
            raise EvidenceError("Span identity/hash mismatch")
        if (span.get("section") not in SECTION_KINDS or not isinstance(span.get("section_title"), str)
                or len(span["section_title"]) > 256):
            raise EvidenceError("Invalid section metadata")
        features = span.get("features")
        if (not isinstance(features, dict) or set(features) != {"equation_like", "table_like", "figure_reference"}
                or any(type(value) is not bool for value in features.values())):
            raise EvidenceError("Invalid span feature metadata")
        byte_start, utf16_start = byte_offsets.get(span["page"], 0), utf16_offsets.get(span["page"], 0)
        byte_end = byte_start + len(quote.encode("utf-8"))
        utf16_end = utf16_start + len(quote.encode("utf-16-le")) // 2
        expected = {"document_start": page["document_start"] + start, "document_end": page["document_start"] + end,
                    "utf8_start": byte_start, "utf8_end": byte_end, "utf16_start": utf16_start, "utf16_end": utf16_end}
        if any(type(span.get(key)) is not int or span[key] != value for key, value in expected.items()):
            raise EvidenceError("Span encoded offsets are inconsistent")
        seen.add(ident)
        cursors[span["page"]], byte_offsets[span["page"]], utf16_offsets[span["page"]] = end, byte_end, utf16_end
    if any(cursors.get(number, 0) != len(page["text"]) for number, page in page_map.items()):
        raise EvidenceError("Evidence spans omit source text")
    return index


def span_text(index, span):
    """Resolve a span after validate_index has established its partition."""
    if isinstance(span, str):
        span = next((item for item in index["spans"] if item["id"] == span), None)
    if not isinstance(span, dict):
        raise EvidenceError("Unknown evidence span")
    page = next((item for item in index["pages"] if item["page"] == span["page"]), None)
    if page is None:
        raise EvidenceError("Unknown evidence page")
    return page["text"][span["start"]:span["end"]]


def validate_requirements(requirements, index, *, max_requirements=512):
    """Validate exact source citations; never infer entailment from a quote match."""
    validate_index(index)
    if not isinstance(requirements, list) or len(requirements) > max_requirements:
        raise EvidenceError("Requirements must be a bounded list")
    spans = {span["id"]: span for span in index["spans"]}
    validated, seen = [], set()
    for item in requirements:
        if not isinstance(item, dict):
            raise EvidenceError("Requirements must be objects")
        allowed = {"id", "kind", "text", "status", "critical", "evidence", "reason"}
        if set(item) - allowed:
            raise EvidenceError("Unknown requirement field")
        ident = item.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", ident) or ident in seen:
            raise EvidenceError("Requirement IDs must be unique stable identifiers")
        if item.get("kind") not in REQUIREMENT_KINDS or item.get("status") not in REQUIREMENT_STATES:
            raise EvidenceError("Unknown requirement kind/status")
        if not _unicode_text(item.get("text")) or not item["text"].strip() or len(item["text"]) > 8000:
            raise EvidenceError("Requirement text must be nonempty and bounded")
        if type(item.get("critical", False)) is not bool:
            raise EvidenceError("critical must be a boolean")
        reason = item.get("reason", "")
        if not _unicode_text(reason) or len(reason) > 4000:
            raise EvidenceError("Requirement reason must be bounded text")
        citations = item.get("evidence", [])
        if not isinstance(citations, list) or len(citations) > 8:
            raise EvidenceError("Evidence citations must be a bounded list")
        if item["status"] in {"stated", "inferred", "conflicting"} and not citations:
            raise EvidenceError("Source-based requirements require evidence")
        if item["status"] in {"assumed", "unresolved", "conflicting"} and not reason.strip():
            raise EvidenceError("Uncertain/conflicting requirements require an explicit reason")
        normalized, cited_ranges = [], set()
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) - {"span_id", "quote", "start", "end", "source_sha256"}:
                raise EvidenceError("Invalid citation schema")
            span = spans.get(citation.get("span_id"))
            quote = citation.get("quote")
            if span is None or not _unicode_text(quote) or not quote.strip() or len(quote) > 4000:
                raise EvidenceError("Citation requires a known span and bounded nonempty quote")
            text = span_text(index, span)
            if "start" in citation or "end" in citation:
                start, end = citation.get("start"), citation.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != quote:
                    raise EvidenceError("Citation offsets do not match the source quote")
            else:
                start = text.find(quote)
                if start < 0 or text.find(quote, start + 1) >= 0:
                    raise EvidenceError("Citation quote is absent or ambiguous; exact offsets are required")
                end = start + len(quote)
            if citation.get("source_sha256", index["source"]["sha256"]) != index["source"]["sha256"]:
                raise EvidenceError("Citation refers to a different source")
            key = (span["id"], start, end)
            if key in cited_ranges:
                raise EvidenceError("Duplicate citation")
            cited_ranges.add(key)
            normalized.append({"span_id": span["id"], "page": span["page"], "quote": quote,
                               "start": start, "end": end, "offset_unit": "unicode_codepoints",
                               "document_start": span["document_start"] + start,
                               "document_end": span["document_start"] + end,
                               "source_sha256": index["source"]["sha256"]})
        if item["status"] == "conflicting" and len(cited_ranges) < 2:
            raise EvidenceError("A conflict needs two distinct evidence ranges")
        validated.append({"id": ident, "kind": item["kind"], "text": item["text"], "status": item["status"],
                          "critical": item.get("critical", False), "reason": reason, "evidence": normalized,
                          "evidence_validation": "quote_presence_not_entailment"})
        seen.add(ident)
    return validated
