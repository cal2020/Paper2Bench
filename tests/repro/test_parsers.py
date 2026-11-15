from __future__ import annotations

import base64
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper2bench.repro import parsers
from paper2bench.repro.parsers import (
    ParserFailure,
    ParserLimitExceeded,
    ParserUnavailable,
    parse_pdf,
    prepare_parser_image,
)
from paper2bench.repro.runtime import ResourceExhausted, RuntimeUnavailable

IMAGE = "sha256:" + "a" * 64


def text_pdf(text="Hello Paper2Bench", *, blank=False, encrypted=False, pages=1):
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=300, height=300)
        if not blank:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 20 200 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("fixture-password")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def fake_runtime(monkeypatch):
    state = {
        "output": {
            "ok": True,
            "text": "\n\n--- Page 1 ---\n\nfixture text",
            "page_count": 1,
            "warnings": [],
        },
        "exit_state": {"Running": False, "ExitCode": 0, "OOMKilled": False},
        "create_args": None,
        "payload": None,
        "cleanups": [],
    }

    def control(self, args, **kwargs):
        if args[0] == "create":
            state["create_args"] = args
            return b"c" * 64
        if args[0] == "inspect":
            return json.dumps(state["exit_state"]).encode()
        raise AssertionError(args)

    def attached(self, name, payload, **kwargs):
        state["payload"] = json.loads(payload)
        state["attach_options"] = kwargs
        if "attached_error" in state:
            raise state["attached_error"]
        output = state["output"]
        return output if isinstance(output, bytes) else json.dumps(output).encode(), b"", 0

    def cleanup(self, name, **kwargs):
        state["cleanups"].append((name, kwargs))
        return state.get("cleanup_confirmed", True)

    monkeypatch.setattr(parsers.DockerRunner, "_control", control)
    monkeypatch.setattr(parsers.DockerRunner, "_attached", attached)
    monkeypatch.setattr(parsers.DockerRunner, "_cleanup", cleanup)
    return state


def test_parser_container_has_no_host_mounts_network_or_host_secrets(fake_runtime, monkeypatch):
    monkeypatch.setenv("PAPER2BENCH_SECRET", "never-in-container")
    result = parse_pdf(b"%PDF-fixture", IMAGE)
    args = fake_runtime["create_args"]
    assert args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert args[args.index("--user") + 1] == "0:0"
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert [args[index + 1] for index, flag in enumerate(args) if flag == "--cap-add"] == [
        "SETUID",
        "SETGID",
        "KILL",
    ]
    assert "--mount" not in args and "--volume" not in args
    assert "never-in-container" not in " ".join(args)
    assert "HTTP_PROXY=" in args and "HTTPS_PROXY=" in args
    payload = fake_runtime["payload"]
    assert base64.b64decode(payload["input"]["pdf_base64"]) == b"%PDF-fixture"
    assert "PdfReader" in payload["code"]
    assert result["runtime"]["cleanup_confirmed"] is True
    assert result["runtime"]["image"] == IMAGE
    assert len(fake_runtime["cleanups"]) == 1


@pytest.mark.parametrize(
    "options,error",
    [
        ({"raw": b""}, ParserFailure),
        ({"raw": "not-bytes"}, ParserFailure),
        ({"raw": b"x" * (16 * 1024 * 1024 + 1)}, ParserLimitExceeded),
        ({"image": "python:3.12-slim"}, ParserUnavailable),
        ({"timeout_seconds": 0}, ParserFailure),
        ({"timeout_seconds": float("nan")}, ParserFailure),
        ({"timeout_seconds": True}, ParserFailure),
        ({"memory_mb": 32}, ParserFailure),
        ({"memory_mb": True}, ParserFailure),
        ({"max_chars": 400001}, ParserFailure),
        ({"max_chars": False}, ParserFailure),
        ({"max_pages": 201}, ParserFailure),
    ],
)
def test_invalid_input_fails_before_docker(monkeypatch, options, error):
    def never_construct(*_, **__):
        pytest.fail("preflight must precede Docker access")

    monkeypatch.setattr(parsers, "DockerRunner", never_construct)
    kwargs = {"raw": b"%PDF-fixture", "image": IMAGE, **options}
    with pytest.raises(error):
        parse_pdf(**kwargs)


@pytest.mark.parametrize(
    "code,error",
    [
        ("encrypted", ParserFailure),
        ("no_text", ParserFailure),
        ("invalid_pdf", ParserFailure),
        ("parser_unavailable", ParserUnavailable),
        ("source_limit", ParserLimitExceeded),
        ("page_limit", ParserLimitExceeded),
        ("text_limit", ParserLimitExceeded),
        ("memory_limit", ParserLimitExceeded),
    ],
)
def test_parser_statuses_are_sanitized_and_cleanup_runs(fake_runtime, code, error):
    fake_runtime["output"] = {"ok": False, "code": code, "attacker_text": "secret-must-not-echo"}
    with pytest.raises(error) as caught:
        parse_pdf(b"%PDF-fixture", IMAGE)
    assert "secret-must-not-echo" not in str(caught.value)
    assert caught.value.cleanup_confirmed is True
    assert caught.value.container_name.startswith("paper2bench-parser-")
    assert len(fake_runtime["cleanups"]) == 1


@pytest.mark.parametrize(
    "output",
    [
        b'{"ok":true,"ok":false}',
        b"not JSON",
        {"ok": True, "text": "text", "page_count": True, "warnings": []},
        {"ok": True, "text": "text\x00bad", "page_count": 1, "warnings": []},
        {"ok": True, "text": "text", "page_count": 1, "warnings": ["x" * 201]},
        {"ok": True, "text": "text", "page_count": 1, "warnings": [], "unexpected": 1},
    ],
)
def test_malformed_parser_result_rejected(fake_runtime, output):
    fake_runtime["output"] = output
    with pytest.raises(ParserFailure):
        parse_pdf(b"%PDF-fixture", IMAGE)


def test_runtime_timeout_maps_to_parser_limit_and_retains_cleanup(fake_runtime):
    fake_runtime["attached_error"] = ResourceExhausted("raw secret debug")
    fake_runtime["cleanup_confirmed"] = False
    with pytest.raises(ParserLimitExceeded) as caught:
        parse_pdf(b"%PDF-fixture", IMAGE)
    assert "raw secret debug" not in str(caught.value)
    assert caught.value.cleanup_confirmed is False


def test_oom_state_maps_to_memory_limit(fake_runtime):
    fake_runtime["exit_state"]["OOMKilled"] = True
    with pytest.raises(ParserLimitExceeded):
        parse_pdf(b"%PDF-fixture", IMAGE)


def test_missing_runtime_fails_closed_and_reconciles_create(monkeypatch):
    cleanup = []

    def unavailable(*_, **__):
        raise RuntimeUnavailable("private socket detail")

    monkeypatch.setattr(parsers.DockerRunner, "_control", unavailable)
    monkeypatch.setattr(
        parsers.DockerRunner, "_cleanup", lambda _, name, **kw: cleanup.append(name) or True
    )
    with pytest.raises(ParserUnavailable) as caught:
        parse_pdf(b"%PDF-fixture", IMAGE)
    assert "private socket detail" not in str(caught.value)
    assert len(cleanup) == 1


def test_vendor_staging_excludes_cache_and_rejects_symlinks(tmp_path, monkeypatch):
    source = tmp_path / "installed" / "pypdf"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# installed package")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "leak.pyc").write_bytes(b"unneeded")
    monkeypatch.setattr(
        parsers.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(source / "__init__.py")),
    )
    _, source_hash = parsers._copy_vendor(tmp_path / "copy")
    assert len(source_hash) == 64
    assert (tmp_path / "copy").stat().st_mode & 0o777 == 0o755
    assert not (tmp_path / "copy" / "__pycache__").exists()
    (source / "external.py").symlink_to(tmp_path / "private")
    with pytest.raises(ParserUnavailable, match="symbolic"):
        parsers._copy_vendor(tmp_path / "second-copy")


def test_prepare_image_uses_only_pinned_base_and_locked_vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(parsers.DockerRunner, "resolve_image", lambda self, ref: IMAGE)

    def vendor(destination):
        destination.mkdir(parents=True)
        (destination / "__init__.py").write_text("# trusted fixture")
        return "6.0.0", "b" * 64

    monkeypatch.setattr(parsers, "_copy_vendor", vendor)
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        assert command[:2] == ["docker", "--host=unix:///var/run/docker.sock"]
        assert command[command.index("--network") + 1] == "none"
        assert "--pull=false" in command
        assert kwargs["env"]["DOCKER_BUILDKIT"] == "0"
        context = Path(command[-1])
        dockerfile = (context / "Dockerfile").read_text()
        assert dockerfile.startswith("FROM " + IMAGE)
        assert "RUN " not in dockerfile and "pip " not in dockerfile
        assert sorted(
            path.relative_to(context).as_posix() for path in context.rglob("*") if path.is_file()
        ) == ["Dockerfile", "vendor/pypdf/__init__.py"]
        Path(command[command.index("--iidfile") + 1]).write_text(IMAGE)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(parsers.subprocess, "run", run)
    assert prepare_parser_image() == IMAGE
    assert len(seen) == 1


@pytest.fixture(scope="module")
def actual_parser_image():
    supplied = os.environ.get("PAPER2BENCH_PARSER_IMAGE")
    if supplied:
        return supplied
    base = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not base:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to stage a real parser image")
    return prepare_parser_image(base)


@pytest.mark.docker
def test_real_pdf_extracts_in_isolated_container(actual_parser_image):
    result = parse_pdf(text_pdf(), actual_parser_image)
    assert result["text"].startswith("\n\n--- Page 1 ---\n\n")
    assert "Hello Paper2Bench" in result["text"]
    assert result["page_count"] == 1
    assert result["runtime"]["cleanup_confirmed"] is True
    assert result["runtime"]["network_used"] is False


@pytest.mark.docker
@pytest.mark.parametrize(
    "options,match", [({"blank": True}, "OCR"), ({"encrypted": True}, "Encrypted")]
)
def test_real_blank_and_encrypted_pdf_fail_closed(actual_parser_image, options, match):
    with pytest.raises(ParserFailure, match=match):
        parse_pdf(text_pdf(**options), actual_parser_image)


@pytest.mark.docker
def test_real_pdf_text_and_page_bounds(actual_parser_image):
    with pytest.raises(ParserLimitExceeded):
        parse_pdf(text_pdf("too much text"), actual_parser_image, max_chars=5)
    with pytest.raises(ParserLimitExceeded):
        parse_pdf(text_pdf(pages=2), actual_parser_image, max_pages=1)
