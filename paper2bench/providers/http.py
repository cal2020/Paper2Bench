"""Bounded, non-streaming text adapters. Redirects and implicit proxies are disabled."""
from __future__ import annotations
import json
import http.client
import ipaddress
import math
import queue
import socket
import threading
import time
from urllib import error, request
from .base import ChatResponse, ProviderError
from ..runtime_config import ModelRef, is_loopback_endpoint


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_MAX_TRANSPORT_WORKERS = 4
_TRANSPORT_SLOTS = threading.BoundedSemaphore(_MAX_TRANSPORT_WORKERS)


class _DeadlineState:
    """Cancellation shared with a bounded daemon worker.

    DNS cannot be interrupted portably. Its worker retains an admission slot
    until it ends; an expired connection can never proceed to send HTTP data.
    Socket shutdown interrupts header/body reads without acquiring buffered-I/O
    locks, which may themselves be held by a blocking worker.
    """
    def __init__(self, deadline, *, loopback_only=False):
        self.deadline = deadline
        self.loopback_only = loopback_only
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.connections = []
        self.sockets = []

    def check(self):
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise TimeoutError("absolute provider deadline")

    def register_socket(self, sock):
        if sock is not None:
            with self.lock:
                self.sockets.append(sock)

    def connect_loopback(self, address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        """Resolve once and connect only to those validated loopback addresses.

        A hostname spelled ``localhost`` is not itself a network boundary: a
        changed resolver/hosts entry must not send credentials or prompt data to
        a remote address. Do not validate then re-resolve via create_connection.
        """
        self.check()
        addresses = socket.getaddrinfo(address[0], address[1], 0, socket.SOCK_STREAM)
        self.check()
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_loopback for item in addresses):
            raise ProviderError("policy_denied", "loopback model endpoint resolved outside the loopback network")
        last_error = None
        for family, kind, protocol, _, target in addresses:
            self.check()
            sock = socket.socket(family, kind, protocol)
            self.register_socket(sock)
            try:
                sock.settimeout(max(0.001, self.deadline - time.monotonic()))
                if source_address:
                    sock.bind(source_address)
                sock.connect(target)
                self.check()
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        raise last_error or OSError("loopback model endpoint has no usable address")

    def connection_factory(self, cls):
        def factory(host, **kwargs):
            self.check()
            connection = cls(host, **kwargs)
            if self.loopback_only:
                connection._create_connection = self.connect_loopback
            with self.lock:
                self.connections.append(connection)
            original_connect, original_send = connection.connect, connection.send

            def connect():
                self.check()
                original_connect()
                self.register_socket(connection.sock)
                # A late DNS resolution must not send even request headers.
                try:
                    self.check()
                except TimeoutError:
                    self.abort()
                    raise

            def send(data):
                self.check()
                if connection.sock is not None:
                    connection.sock.settimeout(max(0.001, self.deadline - time.monotonic()))
                original_send(data)

            connection.connect, connection.send = connect, send
            return connection
        return factory

    def abort(self):
        self.cancelled.set()
        with self.lock:
            sockets = list(self.sockets) + [getattr(c, "sock", None) for c in self.connections]
        seen = set()
        for sock in sockets:
            if sock is None or id(sock) in seen:
                continue
            seen.add(id(sock))
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass


class _DeadlineHTTPHandler(request.HTTPHandler):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def http_open(self, req):
        return self.do_open(self.state.connection_factory(http.client.HTTPConnection), req)


class _DeadlineHTTPSHandler(request.HTTPSHandler):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def https_open(self, req):
        return self.do_open(self.state.connection_factory(http.client.HTTPSConnection), req, context=self._context)


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _post_json(ref: ModelRef, url: str, payload: dict, headers: dict) -> tuple[dict, str | None]:
    deadline = time.monotonic() + ref.timeout_seconds
    if not _TRANSPORT_SLOTS.acquire(timeout=max(0, deadline - time.monotonic())):
        raise ProviderError("transport_capacity", "provider transport capacity is occupied by unfinished requests", outcome_unknown=False)
    state = _DeadlineState(deadline, loopback_only=is_loopback_endpoint(ref.endpoint))
    completed = queue.Queue(maxsize=1)

    def run():
        try:
            result = (True, _perform_post(ref, url, payload, headers, state))
        except BaseException as exc:
            result = (False, exc)
        finally:
            state.abort()
        completed.put_nowait(result)
        _TRANSPORT_SLOTS.release()

    worker = threading.Thread(target=run, name="paper2bench-http-transport", daemon=True)
    try:
        worker.start()
    except BaseException:
        _TRANSPORT_SLOTS.release()
        raise ProviderError("transport_capacity", "provider transport worker could not start") from None
    try:
        ok, result = completed.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty:
        state.abort()
        raise ProviderError("timeout", "absolute provider deadline expired; completion and billing may be unknown", outcome_unknown=True) from None
    if time.monotonic() >= deadline:
        state.abort()
        raise ProviderError("timeout", "absolute provider deadline expired; completion and billing may be unknown", outcome_unknown=True)
    if ok:
        return result
    if isinstance(result, ProviderError):
        raise result
    raise ProviderError("transport", "provider transport failed; completion and billing may be unknown", outcome_unknown=True) from None


def _perform_post(ref: ModelRef, url: str, payload: dict, headers: dict, state: _DeadlineState) -> tuple[dict, str | None]:
    state.check()
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    req = request.Request(url, data=encoded, headers={"Content-Type": "application/json", "Accept": "application/json", **headers}, method="POST")
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect(), _DeadlineHTTPHandler(state), _DeadlineHTTPSHandler(state))
    deadline = state.deadline
    try:
        state.check()
        with opener.open(req, timeout=max(0.001, deadline - time.monotonic())) as response:
            state.register_socket(getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None))
            size = response.headers.get("Content-Length")
            if size is not None:
                try:
                    if int(size) < 0:
                        raise ValueError("negative response length")
                    if int(size) > ref.max_response_bytes:
                        raise ProviderError("response_too_large", "provider response exceeded configured size limit")
                except ValueError:
                    raise ProviderError("malformed_response", "invalid provider response length") from None
            # read1 returns after one underlying read; a trickling server cannot
            # reset the whole response deadline indefinitely.
            chunks, received = [], 0
            while True:
                remaining = deadline - time.monotonic()
                state.check()
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(remaining)
                chunk = response.read1(min(65536, ref.max_response_bytes + 1 - received))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if received > ref.max_response_bytes:
                    raise ProviderError("response_too_large", "provider response exceeded configured size limit")
            body = b"".join(chunks)
            if len(body) > ref.max_response_bytes:
                raise ProviderError("response_too_large", "provider response exceeded configured size limit")
            if size is not None and len(body) != int(size):
                raise ProviderError("malformed_response", "provider response ended before its declared length")
            identity = response.headers.get("x-request-id") or response.headers.get("request-id")
    except error.HTTPError as exc:
        status = exc.code
        exc.close()
        if status in {301, 302, 303, 307, 308}:
            raise ProviderError("policy_denied", "provider redirects are not permitted") from None
        if status in {401, 403}:
            raise ProviderError("authentication", f"provider authentication failed (HTTP {status})") from None
        if status == 429:
            raise ProviderError("rate_limit", "provider rate limit reached", retryable=True) from None
        if status == 404:
            raise ProviderError("unavailable_model", "provider endpoint or model is unavailable") from None
        if status >= 500:
            raise ProviderError("unavailable", f"provider returned HTTP {status}; request outcome may be unknown", outcome_unknown=True) from None
        raise ProviderError("invalid_request", f"provider rejected request (HTTP {status})") from None
    except (TimeoutError, socket.timeout):
        raise ProviderError("timeout", "provider request timed out; completion and billing may be unknown", outcome_unknown=True) from None
    except error.URLError as exc:
        timeout = isinstance(exc.reason, (TimeoutError, socket.timeout))
        raise ProviderError("timeout" if timeout else "transport", "provider connection failed; completion and billing may be unknown", outcome_unknown=True) from None
    except OSError:
        raise ProviderError("transport", "provider connection failed; completion outcome is unknown", outcome_unknown=True) from None
    try:
        state.check()
        data = json.loads(body.decode("utf-8"), parse_constant=_reject_constant,
                          parse_float=_finite_float, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError):
        raise ProviderError("malformed_response", "provider did not return a valid JSON object") from None
    if not isinstance(data, dict):
        raise ProviderError("malformed_response", "provider did not return a JSON object")
    return data, identity


def _usage(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ProviderError("malformed_response", "provider reported invalid token usage")
    return value


class TextHTTPAdapter:
    def __init__(self, provider: str):
        self.provider = provider

    def capabilities(self) -> dict[str, str]:
        return {"text_generation": "supported", "streaming": "unsupported", "tool_calling": "unsupported",
                "image_input": "unsupported", "structured_outputs": "unsupported", "cancellation": "unsupported",
                "model_revision": "unverified", "token_usage": "unverified"}

    def validate_request(self, ref: ModelRef, messages: list[dict], *, max_tokens: int, temperature: float, api_key: str | None = None) -> None:
        if ref.provider != self.provider:
            raise ProviderError("invalid_request", "model binding does not belong to this adapter")
        if api_key is not None and (not isinstance(api_key, str) or len(api_key) > 16384 or any(ord(c) < 32 for c in api_key)):
            raise ProviderError("authentication", "provider credential has an invalid shape")
        if any(self.capabilities().get(name) != "supported" for name in ref.required_capabilities):
            raise ProviderError("unsupported_capability", "required capability is not implemented by this text adapter")
        if not isinstance(messages, list) or not messages:
            raise ProviderError("invalid_request", "messages must be a nonempty list")
        roles = {"system", "user", "assistant"}
        if self.provider in {"openai", "openai_compatible"}:
            roles.add("developer")
        for m in messages:
            if not isinstance(m, dict) or set(m) - {"role", "content"} or m.get("role") not in roles or not isinstance(m.get("content"), str):
                raise ProviderError("unsupported_capability", "only system/user/assistant text messages are supported")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ProviderError("invalid_request", "max_tokens must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ProviderError("invalid_request", "temperature must be finite and between zero and two")
        if self.provider in {"openai", "anthropic"} and not api_key:
            raise ProviderError("authentication", "provider credential is not configured")
        if self.provider == "anthropic" and not any(m["role"] != "system" for m in messages):
            raise ProviderError("invalid_request", "Anthropic requires a non-system message")

    def generate(self, ref: ModelRef, messages: list[dict], *, max_tokens: int, temperature: float, api_key: str | None = None) -> ChatResponse:
        self.validate_request(ref, messages, max_tokens=max_tokens, temperature=temperature, api_key=api_key)
        temperature = ref.parameters.get("temperature", temperature)
        headers = {}
        if self.provider == "ollama":
            url = ref.endpoint + "/api/chat"
            options = {"num_predict": max_tokens}
            if ref.context_tokens is not None:
                # Request the same context allocation that controller preflight
                # used; a model's advertised maximum is not its server default.
                options["num_ctx"] = ref.context_tokens
            if temperature is not None:
                options["temperature"] = temperature
            if "seed" in ref.parameters:
                options["seed"] = ref.parameters["seed"]
            payload = {"model": ref.model, "messages": messages, "stream": False, "options": options}
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
        elif self.provider == "anthropic":
            url = ref.endpoint + "/messages"
            systems = [m["content"] for m in messages if m["role"] == "system"]
            payload = {"model": ref.model.removeprefix("anthropic/"), "messages": [m for m in messages if m["role"] != "system"], "max_tokens": max_tokens}
            if systems:
                payload["system"] = "\n\n".join(systems)
            if not payload["messages"]:
                raise ProviderError("invalid_request", "Anthropic requires a non-system message")
            if temperature is not None:
                payload["temperature"] = temperature
            headers.update({"x-api-key": api_key, "anthropic-version": ref.parameters.get("api_version", "2023-06-01")})
        else:
            url = ref.endpoint + "/chat/completions"
            token_key = ref.parameters.get("max_tokens_parameter", "max_completion_tokens" if self.provider == "openai" else "max_tokens")
            payload = {"model": ref.model, "messages": messages, "stream": False, token_key: max_tokens}
            if temperature is not None:
                payload["temperature"] = temperature
            if "seed" in ref.parameters:
                payload["seed"] = ref.parameters["seed"]
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
        start = time.monotonic()
        data, request_id = _post_json(ref, url, payload, headers)
        if data.get("error"):
            raise ProviderError("provider_error", "provider returned an error response")
        if self.provider == "ollama":
            if data.get("done") is not True:
                raise ProviderError("truncated", "Ollama response was not marked complete")
            message = data.get("message")
            if not isinstance(message, dict) or message.get("tool_calls"):
                raise ProviderError("unsupported_capability", "Ollama output requires unsupported tool handling")
            content = message.get("content")
            finish = data.get("done_reason", "stop")
            input_tokens, output_tokens = _usage(data.get("prompt_eval_count")), _usage(data.get("eval_count"))
        elif self.provider == "anthropic":
            blocks = data.get("content")
            if not isinstance(blocks, list) or any(not isinstance(b, dict) or b.get("type") not in {"text", "thinking", "redacted_thinking"} or (b.get("type") == "text" and not isinstance(b.get("text"), str)) for b in blocks):
                raise ProviderError("unsupported_capability", "Anthropic returned unsupported output blocks")
            # Reasoning blocks are not exposed or retained as assistant text.
            content, finish = "".join(b["text"] for b in blocks if b["type"] == "text"), data.get("stop_reason")
            usage = data.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ProviderError("malformed_response", "invalid provider usage object")
            usage = usage or {}
            input_tokens, output_tokens = _usage(usage.get("input_tokens")), _usage(usage.get("output_tokens"))
            cache_creation = _usage(usage.get("cache_creation_input_tokens"))
            cache_read = _usage(usage.get("cache_read_input_tokens"))
            if input_tokens is not None:
                # Anthropic's input_tokens excludes cache creation/read tokens.
                input_tokens += (cache_creation or 0) + (cache_read or 0)
        else:
            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ProviderError("malformed_response", "provider did not return one completion")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ProviderError("malformed_response", "provider completion has no message")
            if message.get("refusal"):
                raise ProviderError("refusal", "model refused the request")
            if message.get("tool_calls") or message.get("function_call"):
                raise ProviderError("unsupported_capability", "completion requires unsupported tool handling")
            content, finish = message.get("content"), choice.get("finish_reason")
            usage = data.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ProviderError("malformed_response", "invalid provider usage object")
            usage = usage or {}
            input_tokens, output_tokens = _usage(usage.get("prompt_tokens")), _usage(usage.get("completion_tokens"))
        if not isinstance(finish, str):
            raise ProviderError("malformed_response", "provider completion reason is not text")
        if finish in {"length", "max_tokens", "model_context_window_exceeded"}:
            raise ProviderError("truncated", "provider output reached the token limit")
        if finish in {"content_filter", "refusal"}:
            raise ProviderError("refusal", "model refused or filtered the request")
        if finish in {"tool_use", "pause_turn", "compaction", "tool_calls", "function_call"}:
            raise ProviderError("unsupported_capability", "provider response requires unsupported tool, continuation, or compaction handling")
        if finish not in {"stop", "end_turn", "stop_sequence"}:
            raise ProviderError("malformed_response", "provider returned an unsupported or missing completion reason")
        if not isinstance(content, str):
            raise ProviderError("malformed_response", "provider output is not text")
        return ChatResponse(content=content, input_tokens=input_tokens or 0, output_tokens=output_tokens or 0,
                            total_tokens=(input_tokens or 0) + (output_tokens or 0), provider=self.provider,
                            model=ref.model, usage_complete=input_tokens is not None and output_tokens is not None,
                            finish_reason=finish, request_id=request_id or data.get("id"),
                            resolved_model=data.get("model") if isinstance(data.get("model"), str) else None,
                            duration_seconds=time.monotonic() - start,
                            metadata={"input_tokens_measured": input_tokens, "output_tokens_measured": output_tokens,
                                      "requested_revision": ref.revision, "resolved_revision": None, "hosting": ref.hosting,
                                      "locality_assurance": ref.locality_assurance, "quantization": ref.quantization,
                                      "adapter_protocol": "text_http/v1"})
