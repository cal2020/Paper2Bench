import io
import json
import socket
import time
from types import SimpleNamespace

import pytest

from paper2bench import downloader, network, pdf_utils
from paper2bench.network import FetchResult, NetworkFetchError
from paper2bench.pdf_utils import PDFExtractionError, PDFLimits, extract_text_from_pdf
from paper2bench.runtime_config import PolicyDenied


def resolve(ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 443))]


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "0.0.0.0", "::1", "fc00::1", "::ffff:127.0.0.1", "224.0.0.1", "ff02::1", "fec0::1", "168.63.129.16"])
def test_private_dns_destinations_rejected(monkeypatch, ip):
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *a, **kw: resolve(ip))
    with pytest.raises(NetworkFetchError):
        network._destination("https://public.example/paper.pdf")


def test_mixed_private_and_public_dns_fails_closed(monkeypatch):
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *a, **kw: resolve("1.1.1.1") + resolve("127.0.0.1"))
    with pytest.raises(NetworkFetchError):
        network._destination("https://public.example")


@pytest.mark.parametrize("url", ["file:///secret", "http://user:key@example.com", "https://example.com/#x", "http://[bad", "http://example.com:0", "http://example.com/\r\nx:y"])
def test_invalid_urls_rejected_before_connect(url):
    with pytest.raises(NetworkFetchError):
        network._destination(url)


def test_local_policy_blocks_before_dns(monkeypatch):
    monkeypatch.setattr(network, "get_context", lambda: SimpleNamespace(config=SimpleNamespace(policy={"data_egress": "local_only"})))
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *a, **kw: pytest.fail("DNS must not run"))
    for operation in (lambda: network.safe_fetch("https://example.com"), lambda: network.url_reachable("https://example.com"), lambda: downloader.search_arxiv("paper")):
        with pytest.raises(PolicyDenied):
            operation()


def test_dns_lookup_has_a_deadline(monkeypatch):
    def slow(*a, **kw):
        time.sleep(0.2)
        return resolve("1.1.1.1")
    monkeypatch.setattr(network.socket, "getaddrinfo", slow)
    start = time.monotonic()
    with pytest.raises(NetworkFetchError, match="timed out"):
        network._destination("https://example.com", timeout_seconds=0.02)
    assert time.monotonic() - start < 0.15


class Response:
    def __init__(self, status=200, headers=None, body=b"ok"):
        self.status, self.headers, self.stream = status, headers or {}, io.BytesIO(body)
    def getheaders(self):
        return list(self.headers.items())
    def read1(self, count):
        return self.stream.read(count)
    def close(self):
        pass


def fake_connection(monkeypatch, responses, addresses=None):
    connections = []
    monkeypatch.setattr(network.socket, "getaddrinfo", addresses or (lambda *a, **kw: resolve("1.1.1.1")))
    class Connection:
        sock = None
        def __init__(self, hostname, port, address, **kwargs):
            connections.append((hostname, address, kwargs))
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return responses.pop(0)
        def close(self):
            pass
    monkeypatch.setattr(network, "_PinnedConnection", Connection)
    return connections


def test_public_fetch_is_bounded_and_has_pinned_addresses(monkeypatch):
    connections = fake_connection(monkeypatch, [Response(body=b"public")])
    result = network.safe_fetch("https://example.com/paper", max_bytes=10)
    assert result.body == b"public"
    assert connections[0][0] == "example.com"
    assert connections[0][1][3][0] == "1.1.1.1"
    fake_connection(monkeypatch, [Response(body=b"x" * 11)])
    with pytest.raises(NetworkFetchError, match="byte limit"):
        network.safe_fetch("https://example.com", max_bytes=10)


def test_each_redirect_is_checked_before_connection(monkeypatch):
    def addresses(host, *a, **kw):
        return resolve("127.0.0.1" if host == "private.example" else "1.1.1.1")
    connections = fake_connection(monkeypatch, [Response(302, {"Location": "https://private.example/metadata"})], addresses)
    with pytest.raises(NetworkFetchError, match="Private"):
        network.safe_fetch("https://example.com/paper")
    assert len(connections) == 1


def test_https_downgrade_and_compression_are_rejected(monkeypatch):
    fake_connection(monkeypatch, [Response(302, {"Location": "http://example.com/"})])
    with pytest.raises(NetworkFetchError, match="downgrade"):
        network.safe_fetch("https://example.com")
    fake_connection(monkeypatch, [Response(200, {"Content-Encoding": "gzip"})])
    with pytest.raises(NetworkFetchError, match="Compressed"):
        network.safe_fetch("https://example.com")


def test_connect_uses_validated_ip_and_tls_hostname(monkeypatch):
    connected, sni = [], []
    class Sock:
        def settimeout(self, value): pass
        def connect(self, address): connected.append(address)
        def close(self): pass
    sock = Sock()
    monkeypatch.setattr(network.socket, "socket", lambda *a: sock)
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *a, **kw: pytest.fail("must not re-resolve"))
    def wrap(value, server_hostname):
        sni.append(server_hostname)
        return value
    monkeypatch.setattr(network.ssl, "create_default_context", lambda: SimpleNamespace(wrap_socket=wrap))
    connection = network._PinnedConnection("paper.example", 443, (socket.AF_INET, socket.SOCK_STREAM, 6, ("1.1.1.1", 443)), tls=True, timeout=1)
    connection.connect()
    assert connected == [("1.1.1.1", 443)] and sni == ["paper.example"]


def test_download_rejects_bad_content_without_overwriting(monkeypatch, tmp_path):
    output = tmp_path / "paper.pdf"
    output.write_bytes(b"original")
    monkeypatch.setattr(downloader, "safe_fetch", lambda *a, **kw: FetchResult("x", 200, {}, b"not a PDF"))
    with pytest.raises(NetworkFetchError):
        downloader.download_pdf("https://example.com", str(output))
    assert output.read_bytes() == b"original"


def test_download_is_atomic_and_rejects_symlink(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "safe_fetch", lambda *a, **kw: FetchResult("x", 200, {}, b"%PDF-1.7\nexample"))
    output = tmp_path / "paper.pdf"
    downloader.download_pdf("https://example.com", str(output))
    assert output.read_bytes().startswith(b"%PDF-")
    link = tmp_path / "link.pdf"
    link.symlink_to(output)
    with pytest.raises(ValueError):
        downloader.download_pdf("https://example.com", str(link))


def test_task_path_is_checked_before_search(monkeypatch):
    monkeypatch.setattr(downloader, "search_arxiv", lambda *a, **kw: pytest.fail("must not search"))
    with pytest.raises(ValueError):
        downloader.find_and_download("paper", "../../escape")


def test_search_retains_legacy_result_shape(monkeypatch):
    atom = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1234.5678v1</id><title>Test Paper</title><author><name>A Researcher</name></author><published>2025-01-02T00:00:00Z</published><summary>A result</summary><link title="pdf" href="https://arxiv.org/pdf/1234.5678v1" /></entry></feed>'''
    monkeypatch.setattr(downloader, "safe_fetch", lambda *a, **kw: FetchResult("x", 200, {}, atom))
    result = downloader.search_arxiv("test", max_results=1)
    assert result[0] == {"title": "Test Paper", "authors": ["A Researcher"], "published": "2025-01-02", "arxiv_id": "1234.5678v1", "abstract": "A result", "pdf_url": "https://arxiv.org/pdf/1234.5678v1"}


def test_search_rejects_entity_declarations(monkeypatch):
    monkeypatch.setattr(downloader, "safe_fetch", lambda *a, **kw: FetchResult("x", 200, {}, b'<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>'))
    with pytest.raises(NetworkFetchError):
        downloader.search_arxiv("test")


def make_pdf(path, pages=1):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), "Paper evidence page " + str(index + 1))
    document.save(path)
    document.close()


def test_pdf_worker_preserves_page_markers(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path, pages=2)
    text = extract_text_from_pdf(path)
    assert "--- Page 1 ---" in text and "--- Page 2 ---" in text and "Paper evidence" in text


def test_pdf_page_and_text_limits_are_enforced(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path, pages=2)
    with pytest.raises(PDFExtractionError, match="page_limit"):
        extract_text_from_pdf(path, limits=PDFLimits(max_pages=1))
    with pytest.raises(PDFExtractionError, match="text_limit"):
        extract_text_from_pdf(path, limits=PDFLimits(max_text_chars=10))


def test_pdf_rejects_links_and_oversized_input_before_parser(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-" + b"x" * 30)
    monkeypatch.setattr(pdf_utils, "_parse_in_worker", lambda *a: pytest.fail("parser must not start"))
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(path, limits=PDFLimits(max_file_bytes=10))
    link = tmp_path / "link.pdf"
    link.symlink_to(path)
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(link)


def test_parser_worker_has_no_provider_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(pdf_utils, "_WORKER", 'import json,os; print(json.dumps({"text":os.getenv("OPENAI_API_KEY", "absent"),"pages":0}))')
    assert pdf_utils._parse_in_worker(b"", PDFLimits())["text"] == "absent"


def test_parser_deadline_and_output_are_bounded(monkeypatch):
    monkeypatch.setattr(pdf_utils, "_WORKER", 'import time; time.sleep(30)')
    with pytest.raises(PDFExtractionError, match="time or output"):
        pdf_utils._parse_in_worker(b"", PDFLimits(timeout_seconds=0.05))
    monkeypatch.setattr(pdf_utils, "_WORKER", 'print("x"*1000000)')
    with pytest.raises(PDFExtractionError, match="time or output"):
        pdf_utils._parse_in_worker(b"", PDFLimits(max_text_chars=10))


def test_required_memory_guarantee_fails_if_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_utils, "parser_limits_profile", lambda: {"hard_memory": False})
    with pytest.raises(PDFExtractionError, match="Linux worker"):
        extract_text_from_pdf(tmp_path / "not-opened.pdf", limits=PDFLimits(require_hard_memory_limit=True))


def test_denied_group_signal_falls_back_without_false_cleanup_claim(monkeypatch):
    calls = []
    child = SimpleNamespace(pid=123, poll=lambda: None, kill=lambda: calls.append("child_killed"))
    def denied(*args):
        raise PermissionError("managed sandbox")
    monkeypatch.setattr(pdf_utils.os, "killpg", denied)
    assert pdf_utils._terminate_parser_group(child) is False
    assert calls == ["child_killed"]
