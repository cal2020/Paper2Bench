"""Bounded PDF parsing in a resource-limited subprocess.

This contains accidental hangs and content bombs. A subprocess is not a strong
VM security boundary for exploits in the native parser. macOS does not provide
the supported hard address-space limit: require_hard_memory_limit rejects that
profile rather than claiming a RAM guarantee.
"""
from __future__ import annotations

import json
import math
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


class PDFExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class PDFLimits:
    max_file_bytes: int = 32 * 1024 * 1024
    max_pages: int = 500
    max_text_chars: int = 2 * 1024 * 1024
    timeout_seconds: float = 20
    memory_bytes: int = 512 * 1024 * 1024
    require_hard_memory_limit: bool = False

    def __post_init__(self):
        for name in ("max_file_bytes", "max_pages", "max_text_chars", "memory_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise PDFExtractionError(f"{name} must be a positive integer")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 120:
            raise PDFExtractionError("PDF timeout must be positive and at most 120 seconds")
        if self.max_file_bytes > 64 * 1024 * 1024 or self.max_text_chars > 8 * 1024 * 1024 or self.max_pages > 2000:
            raise PDFExtractionError("PDF limits exceed the supported parser profile")
        if self.memory_bytes < 128 * 1024 * 1024:
            raise PDFExtractionError("PDF parser requires at least 128 MiB of address/heap budget")
        if type(self.require_hard_memory_limit) is not bool:
            raise PDFExtractionError("require_hard_memory_limit must be a boolean")


def parser_limits_profile() -> dict:
    return {"boundary": "subprocess_not_vm", "input_bytes": True, "pages": True,
            "text_chars": True, "stdout_bytes": True, "wall_time": True,
            "cpu_time": os.name == "posix", "hard_memory": sys.platform.startswith("linux")}


_WORKER = r'''
import json, math, sys
sys.dont_write_bytecode = True
limits = json.loads(sys.argv[1])
try:
    import resource
    seconds = max(1, math.ceil(limits["timeout_seconds"]))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
    elif limits["require_hard_memory_limit"]:
        raise ValueError("unsupported_memory_limit")
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    import fitz
    data = sys.stdin.buffer.read(limits["max_file_bytes"] + 1)
    if len(data) > limits["max_file_bytes"]:
        raise ValueError("input_limit")
    with fitz.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("encrypted_pdf")
        if document.page_count > limits["max_pages"]:
            raise ValueError("page_limit")
        pages, size = [], 0
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
            if text.strip():
                marked = f"--- Page {page_number} ---\n{text}"
                size += len(marked) + (2 if pages else 0)
                if size > limits["max_text_chars"]:
                    raise ValueError("text_limit")
                pages.append(marked)
        result = {"text": "\n\n".join(pages), "pages": document.page_count}
    print(json.dumps(result, ensure_ascii=False))
except BaseException as exc:
    message = str(exc) if isinstance(exc, ValueError) and str(exc) in {"input_limit", "encrypted_pdf", "page_limit", "text_limit", "unsupported_memory_limit"} else type(exc).__name__
    print(json.dumps({"error": message}))
    sys.exit(2)
'''


def _terminate_parser_group(process) -> bool:
    """Best-effort group kill; never confuse child-only fallback with proof."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        # A managed macOS sandbox can deny group signals during exit races.
        # Killing the direct child still bounds its output/time, but does not
        # establish that all descendants stopped.
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return False


def _parse_in_worker(data: bytes, limits: PDFLimits) -> dict:
    process = subprocess.Popen(
        [sys.executable, "-I", "-u", "-c", _WORKER, json.dumps(asdict(limits))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"}, start_new_session=True,
    )
    outputs = [bytearray(), bytearray()]
    caps = [limits.max_text_chars * 6 + 4096, 16384]
    exceeded = threading.Event()

    def read(pipe, index):
        try:
            while True:
                chunk = pipe.read(8192)
                if not chunk:
                    break
                room = max(0, caps[index] - len(outputs[index]))
                outputs[index].extend(chunk[:room])
                if len(chunk) > room:
                    exceeded.set()
        finally:
            pipe.close()

    def write():
        try:
            process.stdin.write(data)
            process.stdin.flush()
        except OSError:
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    readers = [threading.Thread(target=read, args=(process.stdout, 0), daemon=True),
               threading.Thread(target=read, args=(process.stderr, 1), daemon=True)]
    writer = threading.Thread(target=write, daemon=True)
    for thread in readers + [writer]:
        thread.start()
    deadline = time.monotonic() + limits.timeout_seconds
    timed_out = False
    cleanup_confirmed = True
    while process.poll() is None:
        timed_out = time.monotonic() >= deadline
        if exceeded.is_set() or timed_out:
            cleanup_confirmed = _terminate_parser_group(process)
            break
        time.sleep(0.01)
    process.wait(timeout=3)
    cleanup_confirmed = _terminate_parser_group(process) and cleanup_confirmed
    for thread in readers + [writer]:
        thread.join(timeout=2)
    if timed_out or exceeded.is_set():
        suffix = "; descendant cleanup could not be confirmed" if not cleanup_confirmed else ""
        raise PDFExtractionError("PDF parser exceeded its time or output limit" + suffix)
    if not cleanup_confirmed:
        raise PDFExtractionError("PDF parser descendant cleanup could not be confirmed")
    try:
        result = json.loads(bytes(outputs[0]))
    except (ValueError, UnicodeError) as exc:
        raise PDFExtractionError("PDF parser terminated without a valid bounded response") from exc
    if process.returncode != 0 or not isinstance(result, dict) or "error" in result:
        detail = result.get("error", "parser_failure") if isinstance(result, dict) else "parser_failure"
        raise PDFExtractionError("PDF parser rejected the input: " + str(detail)[:80])
    if set(result) != {"text", "pages"} or not isinstance(result["text"], str) or type(result["pages"]) is not int:
        raise PDFExtractionError("PDF parser returned an invalid result schema")
    if len(result["text"]) > limits.max_text_chars or not 0 <= result["pages"] <= limits.max_pages:
        raise PDFExtractionError("PDF parser returned an out-of-bounds result")
    return result


def extract_text_from_pdf(pdf_path: str, *, limits: Optional[PDFLimits] = None) -> str:
    """Return page-marked text, rejecting symlinks and over-budget documents."""
    limits = limits or PDFLimits()
    if limits.require_hard_memory_limit and not parser_limits_profile()["hard_memory"]:
        raise PDFExtractionError("Hard PDF parser memory limits require the supported Linux worker profile")
    path = Path(pdf_path)
    if path.is_symlink():
        raise PDFExtractionError("PDF input must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limits.max_file_bytes:
                raise PDFExtractionError("PDF must be a regular file within the configured byte limit")
            data = handle.read(limits.max_file_bytes + 1)
    except OSError as exc:
        raise PDFExtractionError("PDF input could not be opened safely") from exc
    if len(data) > limits.max_file_bytes or not data.startswith(b"%PDF-"):
        raise PDFExtractionError("PDF header is invalid or input exceeds its byte limit")
    return _parse_in_worker(data, limits)["text"]
