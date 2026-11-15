"""Bounded PDF extraction in a pinned, disposable no-egress container.

The control process only validates input lengths and parses bounded JSON. It
never imports or executes a PDF parser against document bytes on the host.
Preparing an image copies the installed, version-locked pypdf package, not the
project, credentials, or arbitrary user files.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from .models import Limits
from .runtime import DockerRunner, ResourceExhausted, RuntimeUnavailable

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_CHARS = 400_000
MAX_PDF_PAGES = 200
MAX_VENDOR_BYTES = 16 * 1024 * 1024
_PINNED_IMAGE = re.compile(r"(?:[A-Za-z0-9._:/-]+@)?sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")


class ParserFailure(ValueError):
    """Document extraction failed; message contains no document or stderr text."""


class ParserUnavailable(ParserFailure):
    """A pinned parser image or local daemon is unavailable."""


class ParserLimitExceeded(ParserFailure):
    """A configured document, output, memory, or execution limit was reached."""


# This code executes only inside the isolated container. Sys.path is extended
# explicitly because DockerRunner invokes Python -I, which ignores PYTHONPATH.
_PARSER_CODE = r"""
import base64, io, json, sys, unicodedata
sys.path.insert(0, '/opt/vendor')

def fail(code):
    print(json.dumps({'ok': False, 'code': code}))
    raise SystemExit(0)

try:
    from pypdf import PdfReader
except Exception:
    fail('parser_unavailable')

try:
    payload = json.load(sys.stdin)
    raw = base64.b64decode(payload['pdf_base64'], validate=True)
    if len(raw) > 16777216:
        fail('source_limit')
    reader = PdfReader(io.BytesIO(raw), strict=True)
    if reader.is_encrypted:
        fail('encrypted')
    page_count = len(reader.pages)
    if page_count < 1:
        fail('no_pages')
    if page_count > payload['max_pages']:
        fail('page_limit')
    warnings, parts, total, nonempty = [], [], 0, 0
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ''
        if page_text.strip():
            nonempty += 1
        else:
            warnings.append(f'Page {index} has no extractable text; OCR is unsupported')
        part = f'\n\n--- Page {index} ---\n\n{page_text}'
        total += len(part)
        if total > payload['max_chars']:
            fail('text_limit')
        parts.append(part)
    if not nonempty:
        fail('no_text')
    text = unicodedata.normalize('NFC', ''.join(parts).removeprefix('\ufeff').replace('\r\n', '\n').replace('\r', '\n'))
    if len(text) > payload['max_chars']:
        fail('text_limit')
    if '\x00' in text:
        fail('invalid_text')
    print(json.dumps({'ok': True, 'text': text, 'page_count': page_count, 'warnings': warnings}, ensure_ascii=False))
except SystemExit:
    raise
except MemoryError:
    fail('memory_limit')
except Exception:
    fail('invalid_pdf')
"""


def _copy_vendor(destination: Path) -> tuple[str, str]:
    """Copy only regular installed package files, with an auditable source hash."""
    spec = importlib.util.find_spec("pypdf")
    if spec is None or not spec.origin:
        raise ParserUnavailable("The locked pypdf package is not installed")
    source = Path(spec.origin).parent
    if source.is_symlink() or not source.is_dir():
        raise ParserUnavailable("Installed parser package has an invalid location")
    destination.mkdir(parents=True)
    total = 0
    hasher = hashlib.sha256()
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ParserUnavailable("Installed parser package contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ParserUnavailable("Installed parser package contains a nonregular file")
        data = path.read_bytes()
        total += len(data)
        if total > MAX_VENDOR_BYTES:
            raise ParserLimitExceeded("Installed parser package exceeds the staging limit")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)
        hasher.update(relative.as_posix().encode("utf-8") + b"\0")
        hasher.update(hashlib.sha256(data).digest())
        copied += 1
    if copied == 0 or not (destination / "__init__.py").is_file():
        raise ParserUnavailable("Installed parser package contains no Python package")
    # The workspace may use umask 077. Package directories copied into a root-
    # owned image still need traversal by the unprivileged parser user.
    destination.chmod(0o755)
    for path in destination.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    return importlib.metadata.version("pypdf"), hasher.hexdigest()


def prepare_parser_image(base_image: str = "python:3.12-slim") -> str:
    """Stage a parser image without executing pip, a document, or networked builds.

    Legacy local-image construction is used deliberately: its FROM accepts a
    locally staged immutable image ID without registry name resolution. Builds
    contain only COPY and LABEL instructions. A modern daemon lacking this
    capability fails closed, rather than falling back to fetching a base image.
    """
    runner = DockerRunner()
    try:
        base_id = runner.resolve_image(base_image)
        with tempfile.TemporaryDirectory(prefix="paper2bench-parser-build-") as directory:
            context = Path(directory)
            version, vendor_hash = _copy_vendor(context / "vendor" / "pypdf")
            if re.fullmatch(r"[A-Za-z0-9._+-]+", version) is None:
                raise ParserUnavailable("Installed parser version has an invalid identifier")
            (context / "Dockerfile").write_text(
                f"FROM {base_id}\n"
                "COPY vendor/pypdf/ /opt/vendor/pypdf/\n"
                "LABEL paper2bench.parser=true\n"
                f"LABEL paper2bench.parser.pypdf={version}\n"
                f"LABEL paper2bench.parser.vendor_sha256={vendor_hash}\n"
                f"LABEL paper2bench.parser.base_image={base_id}\n",
                encoding="utf-8",
            )
            iid_file = context / "image-id"
            environment = runner._environment()
            environment["DOCKER_BUILDKIT"] = "0"
            try:
                result = subprocess.run(
                    runner.command
                    + [
                        "build",
                        "--network",
                        "none",
                        "--pull=false",
                        "--quiet",
                        "--iidfile",
                        str(iid_file),
                        str(context),
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    env=environment,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ParserUnavailable("Local parser image preparation did not complete") from exc
            if result.returncode or not iid_file.is_file():
                raise ParserUnavailable("Local parser image preparation failed")
            image = iid_file.read_text(encoding="ascii").strip()
            if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
                raise ParserUnavailable("Docker returned an invalid parser image identity")
            return runner.resolve_image(image)
    except RuntimeUnavailable as exc:
        raise ParserUnavailable("Local Docker runtime or staged base image is unavailable") from exc


def _validate_limits(raw, image, timeout_seconds, memory_mb, max_chars, max_pages) -> None:
    if not isinstance(raw, bytes):
        raise ParserFailure("PDF input must be bytes")
    if not raw:
        raise ParserFailure("PDF source is empty")
    if len(raw) > MAX_SOURCE_BYTES:
        raise ParserLimitExceeded("PDF source exceeds the 16 MiB limit")
    if not isinstance(image, str) or not _PINNED_IMAGE.fullmatch(image):
        raise ParserUnavailable("PDF extraction requires an immutable parser image ID or digest")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 120
    ):
        raise ParserFailure("Parser timeout must be positive and at most 120 seconds")
    if type(memory_mb) is not int or not 64 <= memory_mb <= 4096:
        raise ParserFailure("Parser memory must be between 64 and 4096 MiB")
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_SOURCE_CHARS:
        raise ParserFailure("Parser character limit must be between 1 and 400000")
    if type(max_pages) is not int or not 1 <= max_pages <= MAX_PDF_PAGES:
        raise ParserFailure("Parser page limit must be between 1 and 200")


def _decode_result(output: bytes, max_chars: int, max_pages: int) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        result = json.loads(output.decode("utf-8"), object_pairs_hook=unique)
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("invalid result")
        if result["ok"] is False:
            code = result.get("code")
            if code in {"source_limit", "page_limit", "text_limit", "memory_limit"}:
                raise ParserLimitExceeded("PDF extraction reached a declared resource limit")
            if code == "parser_unavailable":
                raise ParserUnavailable("Pinned image does not contain the staged PDF parser")
            if code == "encrypted":
                raise ParserFailure("Encrypted PDFs are unsupported; provide extracted text")
            if code == "no_text":
                raise ParserFailure("PDF has no extractable text; provide OCR text explicitly")
            raise ParserFailure("PDF could not be parsed")
        if set(result) != {"ok", "text", "page_count", "warnings"}:
            raise ValueError("invalid fields")
        text = result["text"]
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError("invalid text")
        if len(text) > max_chars:
            raise ParserLimitExceeded("PDF text exceeds the declared character limit")
        if type(result["page_count"]) is not int or not 1 <= result["page_count"] <= max_pages:
            raise ValueError("invalid page count")
        warnings = result["warnings"]
        if (
            not isinstance(warnings, list)
            or len(warnings) > max_pages
            or any(not isinstance(warning, str) or len(warning) > 200 for warning in warnings)
        ):
            raise ValueError("invalid warnings")
        return {"text": text, "page_count": result["page_count"], "warnings": warnings}
    except ParserFailure:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ParserFailure("Parser returned an invalid bounded text result") from exc


def parse_pdf(
    raw: bytes,
    image: str,
    *,
    timeout_seconds: float = 20,
    memory_mb: int = 512,
    max_chars: int = MAX_SOURCE_CHARS,
    max_pages: int = MAX_PDF_PAGES,
) -> dict:
    """Extract text in a pinned Docker image; never execute a host PDF fallback."""
    import base64

    _validate_limits(raw, image, timeout_seconds, memory_mb, max_chars, max_pages)
    output_limit = max_chars * 4 + 65536
    limits = Limits(
        timeout_seconds=float(timeout_seconds), memory_mb=memory_mb, output_bytes=output_limit
    )
    runtime_spec = SimpleNamespace(seed=0, limits=limits)
    runner = DockerRunner()
    name = f"paper2bench-parser-{uuid.uuid4().hex}"
    started = time.monotonic()
    deadline = started + timeout_seconds
    payload = json.dumps(
        {
            "code": _PARSER_CODE,
            "input": {
                "pdf_base64": base64.b64encode(raw).decode("ascii"),
                "max_chars": max_chars,
                "max_pages": max_pages,
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    uncertain_create = False
    primary_error = None
    metadata = {
        "image": image,
        "network_used": False,
        "memory_limit_mb": memory_mb,
        "isolation_scope": "single_user_container",
        "cleanup_confirmed": False,
    }
    try:
        try:
            created = runner._control(
                runner._create_args(name, image, runtime_spec), timeout=min(10, timeout_seconds)
            )
        except RuntimeUnavailable as exc:
            uncertain_create = isinstance(exc.__cause__, subprocess.TimeoutExpired)
            if time.monotonic() >= deadline:
                raise ParserLimitExceeded("PDF parser setup exceeded its time limit") from exc
            raise ParserUnavailable("Local parser runtime or staged image is unavailable") from exc
        if not _CONTAINER_ID.fullmatch(created.decode("ascii").strip()):
            raise ParserUnavailable("Docker returned an invalid parser container identity")
        output, _stderr, client_exit = runner._attached(
            name, payload, deadline=deadline, output_limit=output_limit, cancelled=lambda: False
        )
        try:
            state = json.loads(runner._control(["inspect", "--format", "{{json .State}}", name]))
        except (ValueError, TypeError) as exc:
            raise ParserUnavailable("Cannot verify the parser container exit state") from exc
        if state.get("OOMKilled"):
            raise ParserLimitExceeded("PDF parser exceeded its memory limit")
        if client_exit or state.get("ExitCode") != 0 or state.get("Running"):
            raise ParserFailure("PDF parser container exited unsuccessfully")
        if time.monotonic() >= deadline:
            raise ParserLimitExceeded("PDF parser exceeded its time limit")
        result = _decode_result(output, max_chars, max_pages)
        result["runtime"] = metadata
        metadata["wall_seconds"] = time.monotonic() - started
        return result
    except ResourceExhausted as exc:
        primary_error = ParserLimitExceeded("PDF parser exceeded its time or output limit")
        raise primary_error from exc
    except RuntimeUnavailable as exc:
        primary_error = ParserUnavailable("Local PDF parser runtime is unavailable")
        raise primary_error from exc
    except ParserFailure as exc:
        primary_error = exc
        raise
    finally:
        confirmed = runner._cleanup(name, uncertain_create=uncertain_create)
        metadata["cleanup_confirmed"] = confirmed
        if primary_error is not None:
            primary_error.cleanup_confirmed = confirmed
            primary_error.container_name = name
