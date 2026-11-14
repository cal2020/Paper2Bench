"""Parser diagnostics must never share the bounded JSON protocol stream."""
from dataclasses import asdict
import json
import subprocess
import sys

import pytest

from paper2bench import pdf_utils
from paper2bench.pdf_utils import PDFExtractionError, PDFLimits


def noisy_importer(*, legacy=False, dependency_missing=False, flood=False):
    """Inject a parser double before the real isolated worker bootstrap runs."""
    return f'''
import builtins, os, types
original_import = builtins.__import__
class Page:
    def get_text(self):
        print("Python page diagnostic")
        os.write(1, b"Native page diagnostic\\n")
        return "Fixture text"
class Document:
    needs_pass = False
    page_count = 1
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __iter__(self): return iter([Page()])
def fake_import(name, *args, **kwargs):
    if name == "pymupdf":
        if {dependency_missing!r}:
            raise ModuleNotFoundError("missing parser dependency", name="parser_dependency")
        if {legacy!r}:
            raise ModuleNotFoundError("old PyMuPDF module layout", name="pymupdf")
    if name in {{"pymupdf", "fitz"}}:
        if name == "fitz" and not {legacy!r}:
            raise AssertionError("Canonical import must not fall back")
        print("Python import diagnostic: " + name)
        os.write(1, b"Native import diagnostic\\n")
        if {flood!r}: os.write(1, b"x" * 32768)
        return types.SimpleNamespace(open=lambda **kwargs: Document())
    return original_import(name, *args, **kwargs)
builtins.__import__ = fake_import
'''


@pytest.mark.parametrize("legacy", [False, True])
def test_noisy_parser_import_and_extraction_keep_exact_json_stdout(legacy):
    worker = noisy_importer(legacy=legacy) + pdf_utils._WORKER
    process = subprocess.run(
        [sys.executable, "-I", "-u", "-c", worker, json.dumps(asdict(PDFLimits()))],
        input=b"%PDF-fake-parser-fixture", capture_output=True, timeout=5,
    )
    assert process.returncode == 0
    assert json.loads(process.stdout) == {"text": "--- Page 1 ---\nFixture text", "pages": 1}
    assert process.stdout.count(b"\n") == 1
    name = b"fitz" if legacy else b"pymupdf"
    assert b"Python import diagnostic: " + name in process.stderr
    assert b"Native import diagnostic" in process.stderr
    assert b"Python page diagnostic" in process.stderr
    assert b"Native page diagnostic" in process.stderr


def test_canonical_import_dependency_failure_does_not_use_legacy_fallback(monkeypatch):
    monkeypatch.setattr(pdf_utils, "_WORKER", noisy_importer(dependency_missing=True) + pdf_utils._WORKER)
    with pytest.raises(PDFExtractionError, match="ModuleNotFoundError"):
        pdf_utils._parse_in_worker(b"%PDF-fixture", PDFLimits())


def test_redirected_parser_diagnostics_keep_existing_stderr_bound(monkeypatch):
    monkeypatch.setattr(pdf_utils, "_WORKER", noisy_importer(flood=True) + pdf_utils._WORKER)
    with pytest.raises(PDFExtractionError, match="time or output limit"):
        pdf_utils._parse_in_worker(b"%PDF-fixture", PDFLimits())


def test_controller_does_not_accept_arbitrary_protocol_prefix(monkeypatch):
    monkeypatch.setattr(pdf_utils, "_WORKER", 'print("unexpected prefix"); print(\'{"text":"fixture","pages":1}\')')
    with pytest.raises(PDFExtractionError, match="valid bounded response"):
        pdf_utils._parse_in_worker(b"%PDF-fixture", PDFLimits())
