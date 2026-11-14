"""Bounded public-source HTTP fetching with policy, redirect and DNS checks."""
from __future__ import annotations

import http.client
import ipaddress
import math
import socket
import ssl
import threading
import time
import queue
from dataclasses import dataclass
from urllib.parse import quote, urljoin, urlsplit

from .runtime_config import PolicyDenied, get_context


class NetworkFetchError(ValueError):
    """A public URL could not be fetched safely within its limits."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    headers: dict
    body: bytes


def require_external_access(operation: str = "external source access") -> None:
    context = get_context()
    if context is not None:
        policy = context.config.policy
        if policy.get("data_egress") == "local_only" or policy.get("verified_offline_required"):
            raise PolicyDenied(f"{operation} is prohibited by the runtime's local-only policy")


_DNS_SLOTS = threading.BoundedSemaphore(4)


def _resolve(hostname, port, timeout):
    deadline = time.monotonic() + timeout
    if not _DNS_SLOTS.acquire(timeout=timeout):
        raise NetworkFetchError("DNS resolver capacity exceeded")
    result = queue.Queue(maxsize=1)

    def lookup():
        try:
            result.put((True, socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)))
        except OSError as exc:
            result.put((False, exc))
        finally:
            _DNS_SLOTS.release()

    threading.Thread(target=lookup, daemon=True).start()
    try:
        successful, value = result.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty as exc:
        raise NetworkFetchError("Public source DNS lookup timed out") from exc
    if not successful:
        raise NetworkFetchError("Public source DNS lookup failed") from value
    return value


def _destination(url: str, timeout_seconds: float = 10):
    if not isinstance(url, str) or not url or len(url) > 8192 or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise NetworkFetchError("URL must be bounded text without control characters")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise NetworkFetchError("Malformed public-source URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.fragment:
        raise NetworkFetchError("Only public HTTP(S) URLs without fragments are supported")
    if parsed.username or parsed.password:
        raise NetworkFetchError("Credentials are prohibited in public-source URLs")
    try:
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        hostname = parsed.hostname.encode("idna").decode("ascii")
    except (ValueError, UnicodeError) as exc:
        raise NetworkFetchError("Invalid URL hostname or port") from exc
    if "%" in hostname:
        raise NetworkFetchError("Scoped/encoded hosts are prohibited")
    if not 1 <= port <= 65535 or len(hostname) > 253:
        raise NetworkFetchError("Invalid public-source port or hostname")
    addresses = _resolve(hostname, port, timeout_seconds)
    if not addresses or len(addresses) > 32:
        raise NetworkFetchError("Invalid public-source DNS result")
    seen, approved = set(), []
    for family, socktype, proto, _, address in addresses:
        if family not in (socket.AF_INET, socket.AF_INET6):
            raise NetworkFetchError("Unsupported destination address family")
        try:
            ip = ipaddress.ip_address(address[0])
        except ValueError as exc:
            raise NetworkFetchError("Invalid resolved address") from exc
        # is_global alone can be true for multicast and deprecated IPv6
        # site-local addresses in supported Python versions.
        if (not ip.is_global or ip.is_multicast or ip.is_reserved or getattr(ip, "is_site_local", False)
                or str(ip) == "168.63.129.16"  # Azure platform virtual address.
                or (getattr(ip, "ipv4_mapped", None) and not ip.ipv4_mapped.is_global)):
            raise NetworkFetchError("Private, local, reserved and metadata destinations are prohibited")
        identity = (family, address)
        if identity not in seen:
            seen.add(identity)
            approved.append((family, socktype, proto, address))
    return parsed, hostname, port, approved


class _PinnedConnection(http.client.HTTPConnection):
    def __init__(self, hostname, port, address, *, tls, timeout):
        super().__init__(hostname, port, timeout=timeout)
        self._approved_address, self._tls = address, tls

    def connect(self):
        family, socktype, proto, address = self._approved_address
        sock = socket.socket(family, socktype, proto)
        self.sock = sock
        try:
            sock.settimeout(self.timeout)
            sock.connect(address)  # No second hostname resolution.
            if self._tls:
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
            self.sock = sock
        except BaseException:
            sock.close()
            raise


def safe_fetch(url: str, *, max_bytes: int = 16 * 1024 * 1024,
               timeout_seconds: float = 30, max_redirects: int = 5,
               method: str = "GET") -> FetchResult:
    require_external_access("public-source fetch")
    if type(max_bytes) is not int or not 0 <= max_bytes <= 64 * 1024 * 1024:
        raise NetworkFetchError("Fetch byte limit must be between zero and 64 MiB")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise NetworkFetchError("Fetch timeout must be positive and at most 120 seconds")
    if type(max_redirects) is not int or not 0 <= max_redirects <= 10 or method not in {"GET", "HEAD"}:
        raise NetworkFetchError("Unsupported fetch method or redirect limit")
    deadline, lock = time.monotonic() + timeout_seconds, threading.Lock()
    active = [None]

    def close_active():
        with lock:
            connection = active[0]
        if connection is not None:
            try:
                if connection.sock:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    timer = threading.Timer(timeout_seconds, close_active)
    timer.daemon = True
    timer.start()
    current = url
    try:
        for redirect in range(max_redirects + 1):
            if time.monotonic() >= deadline:
                raise NetworkFetchError("Public-source fetch exceeded its time limit")
            parsed, hostname, port, approved = _destination(current, deadline - time.monotonic())
            response = None
            last_error = None
            for address in approved:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise NetworkFetchError("Public-source fetch exceeded its time limit")
                connection = _PinnedConnection(hostname, port, address, tls=parsed.scheme == "https", timeout=remaining)
                with lock:
                    active[0] = connection
                path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
                if parsed.query:
                    path += "?" + quote(parsed.query, safe="%=&/:?@!$'()*+,;~-._")
                try:
                    connection.request(method, path, headers={"User-Agent": "Paper2Bench/0.4 source-fetcher", "Accept-Encoding": "identity", "Connection": "close"})
                    response = connection.getresponse()
                    break
                except (OSError, http.client.HTTPException) as exc:
                    last_error = exc
                    connection.close()
            if response is None:
                raise NetworkFetchError("Public-source connection failed") from last_error
            try:
                raw_headers = response.getheaders()
                headers = {k.lower(): v for k, v in raw_headers}
                # HTTPResponse.read1 may return EOF with bytes still missing.
                # Reject ambiguous framing before retaining any source bytes.
                framing = {}
                for key, value in raw_headers:
                    key = key.lower()
                    if key in {"content-length", "transfer-encoding"}:
                        if key in framing:
                            raise NetworkFetchError("Duplicate source response framing header")
                        framing[key] = value
                if "content-length" in framing and "transfer-encoding" in framing:
                    raise NetworkFetchError("Ambiguous source response framing")
                if ("transfer-encoding" in framing
                        and framing["transfer-encoding"].strip().lower() != "chunked"):
                    raise NetworkFetchError("Unsupported source transfer encoding")
                if response.status in {301, 302, 303, 307, 308}:
                    target = headers.get("location")
                    if not target or redirect >= max_redirects:
                        raise NetworkFetchError("Invalid redirect or redirect limit exceeded")
                    target = urljoin(current, target)
                    if parsed.scheme == "https" and urlsplit(target).scheme != "https":
                        raise NetworkFetchError("HTTPS redirects cannot downgrade transport")
                    current = target
                    continue
                if headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                    raise NetworkFetchError("Compressed source responses are not accepted")
                content_length = headers.get("content-length")
                size = None
                if content_length is not None:
                    content_length = content_length.strip(" \t")
                    if (not isinstance(content_length, str) or not content_length.isascii()
                            or not content_length.isdecimal() or len(content_length) > 20):
                        raise NetworkFetchError("Invalid source Content-Length")
                    size = int(content_length)
                    if size < 0 or (method != "HEAD" and size > max_bytes):
                        raise NetworkFetchError("Source exceeds its byte limit")
                content = bytearray()
                while method != "HEAD":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise NetworkFetchError("Public-source fetch exceeded its time limit")
                    if connection.sock:
                        connection.sock.settimeout(remaining)
                    chunk = response.read1(min(65536, max_bytes + 1 - len(content)))
                    if not chunk:
                        break
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise NetworkFetchError("Source exceeds its byte limit")
                if (method != "HEAD" and response.status not in {204, 304}
                        and size is not None and len(content) != size):
                    raise NetworkFetchError("Source response length does not match its declared length")
                return FetchResult(current, response.status, headers, bytes(content))
            finally:
                response.close()
                connection.close()
        raise NetworkFetchError("Source redirect limit exceeded")
    except (OSError, http.client.HTTPException) as exc:
        raise NetworkFetchError("Public-source fetch failed or timed out") from exc
    finally:
        timer.cancel()
        close_active()


def url_reachable(url: str, timeout: float = 6) -> bool:
    require_external_access("external URL validation")
    try:
        return 200 <= safe_fetch(url, method="HEAD", max_bytes=0, timeout_seconds=timeout).status < 300
    except NetworkFetchError:
        return False
