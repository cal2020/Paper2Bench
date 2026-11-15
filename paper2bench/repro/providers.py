"""Explicit, bounded OpenAI-compatible model access.

Only text chat is implemented. An endpoint being API-compatible does not imply
support for tools, images, JSON schemas, or a reproducible model revision.
Credentials are resolved from the environment and never become profile fields.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderError(RuntimeError):
    """Base exception with a message safe for persisted run records."""


class ProviderConfigurationError(ProviderError):
    """The profile, credentials, or request are invalid before dispatch."""


class UnsupportedCapability(ProviderConfigurationError):
    """The declared policy or operation is unsupported by this adapter."""


class ProviderRequestError(ProviderError):
    """A rejected request, unavailable connection, or invalid response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OutcomeUnknown(ProviderError):
    """Dispatch may have succeeded; do not retry without reconciliation."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class _Immutable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ProviderProfile(_Immutable):
    name: str = Field(min_length=1, max_length=100)
    base_url: str = Field(min_length=1, max_length=2048)
    location: Literal["local", "cloud"]
    api_key_env: str | None = None
    timeout: float = Field(default=60, gt=0, le=600)
    model: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("name", "model")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("must be nonblank without surrounding whitespace")
        return value

    @field_validator("api_key_env")
    @classmethod
    def environment_name(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must name an environment variable")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            # Access port to reject invalid or out-of-range values at configuration time.
            _ = parsed.port
        except ValueError:
            raise ValueError("invalid provider URL") from None
        if (
            value != value.strip()
            or any(ord(char) < 33 or ord(char) == 127 for char in value)
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "\\" in value
        ):
            raise ValueError(
                "provider URL must be HTTP(S), without credentials, query, or fragment"
            )
        # Base URLs are API roots, conventionally ending in /v1. Preserve their path.
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))

    def canonical_config(self) -> dict[str, Any]:
        """Serializable configuration; includes an environment NAME, never its value."""
        return self.model_dump(mode="json")


class ProviderUsage(_Immutable):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class Generation(_Immutable):
    text: str
    model: str
    request_id: str | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    fingerprint: str | None = None
    finish_reason: str | None = None


def _identifier(value: Any) -> str | None:
    if (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and all(32 <= ord(char) < 127 for char in value)
    ):
        return value
    return None


def _token_count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 10**12 else None


class ProviderClient:
    """Text-only adapter with a whole-request deadline and no automatic retries.

    ``generate`` is the synchronous CLI entry point. Async applications must await
    ``agenerate`` instead; calls never create an abandoned background thread.
    """

    MAX_RESPONSE_BYTES = 2 * 1024 * 1024
    MAX_REQUEST_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        profile: ProviderProfile,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if transport is not None and not isinstance(transport, httpx.AsyncBaseTransport):
            raise ProviderConfigurationError("Injected transports must support async requests.")
        self.profile = profile
        self._transport = transport

    def validate(self, *, local_only: bool) -> None:
        if local_only and self.profile.location != "local":
            raise UnsupportedCapability("Local-only policy forbids this cloud provider.")
        url = urlsplit(self.profile.base_url)
        if self.profile.location == "local":
            try:
                address = ipaddress.ip_address(url.hostname or "")
            except ValueError:
                raise ProviderConfigurationError(
                    "Local providers require a numeric loopback address; DNS names are unsupported."
                ) from None
            if not address.is_loopback:
                raise ProviderConfigurationError("Local provider address must be loopback.")
        elif url.scheme != "https":
            raise ProviderConfigurationError("Cloud provider URLs require HTTPS.")
        if self.profile.api_key_env:
            key = os.environ.get(self.profile.api_key_env)
            if not key or not all(33 <= ord(char) < 127 for char in key):
                raise ProviderConfigurationError(
                    "Configured provider credential is missing or invalid."
                )

    def generate(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        local_only: bool,
        total_timeout: float | None = None,
    ) -> Generation:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ProviderConfigurationError(
                "Synchronous generation cannot run inside an event loop; await agenerate instead."
            )
        return asyncio.run(
            self.agenerate(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                local_only=local_only,
                total_timeout=total_timeout,
            )
        )

    async def agenerate(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        local_only: bool,
        total_timeout: float | None = None,
    ) -> Generation:
        self.validate(local_only=local_only)
        deadline = self.profile.timeout if total_timeout is None else total_timeout
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or deadline <= 0
        ):
            raise ProviderConfigurationError("total_timeout must be a finite positive duration.")
        selected_model = model or self.profile.model
        if (
            not isinstance(selected_model, str)
            or not selected_model.strip()
            or len(selected_model) > 256
        ):
            raise ProviderConfigurationError("A nonempty model identifier is required.")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 1_000_000:
            raise ProviderConfigurationError(
                "max_output_tokens must be an integer from 1 to 1000000."
            )
        if not isinstance(messages, list) or not messages:
            raise ProviderConfigurationError("A nonempty list of text messages is required.")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message.get("role"), str)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise UnsupportedCapability(
                    "This adapter supports only system/user/assistant messages with text content."
                )
        body = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": False,
            "n": 1,
        }
        try:
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ProviderConfigurationError("Request is not valid UTF-8 JSON.") from None
        if len(encoded) > self.MAX_REQUEST_BYTES:
            raise ProviderConfigurationError("Provider request exceeds the configured size limit.")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        if self.profile.api_key_env:
            key = os.environ.get(self.profile.api_key_env)
            if not key or not all(33 <= ord(char) < 127 for char in key):
                raise ProviderConfigurationError(
                    "Configured provider credential is missing or invalid."
                )
            headers["Authorization"] = "Bearer " + key
        request_id = None
        try:
            # trust_env=False for local endpoints prevents proxying a local-only request.
            # Cloud requests retain the managed environment's proxy and CA configuration.
            async with (
                asyncio.timeout(deadline),
                httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self.profile.timeout,
                    follow_redirects=False,
                    trust_env=self.profile.location != "local",
                ) as client,
                client.stream(
                    "POST",
                    self.profile.base_url + "/chat/completions",
                    content=encoded,
                    headers=headers,
                ) as response,
            ):
                request_id = _identifier(response.headers.get("x-request-id"))
                if 300 <= response.status_code < 400:
                    raise ProviderRequestError(
                        "Provider redirect refused; configure the exact endpoint.",
                        status_code=response.status_code,
                    )
                if response.status_code >= 500:
                    raise OutcomeUnknown(
                        "Provider returned a server error; completion and charges are unknown.",
                        request_id=request_id,
                    )
                if response.status_code != 200:
                    raise ProviderRequestError(
                        f"Provider rejected the request with HTTP {response.status_code}.",
                        status_code=response.status_code,
                    )
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ProviderRequestError(
                        "Compressed provider responses are unsupported; identity encoding is required."
                    )
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(chunks) + len(chunk) > self.MAX_RESPONSE_BYTES:
                        raise ProviderRequestError("Provider response exceeds the size limit.")
                    chunks.extend(chunk)
        except (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ConnectError):
            raise ProviderRequestError("Provider connection unavailable before dispatch.") from None
        except TimeoutError:
            raise OutcomeUnknown(
                "Provider total deadline exceeded after possible dispatch; completion and charges are unknown.",
                request_id=request_id,
            ) from None
        except httpx.InvalidURL:
            raise ProviderConfigurationError("Provider or proxy URL is invalid.") from None
        except httpx.TransportError:
            raise OutcomeUnknown(
                "Provider transport failed after possible dispatch; completion and charges are unknown.",
                request_id=request_id,
            ) from None
        return self._parse_response(bytes(chunks), selected_model, request_id)

    @staticmethod
    def _parse_response(raw: bytes, model: str, request_id: str | None) -> Generation:
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ProviderRequestError("Provider returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise ProviderRequestError("Provider response must be a JSON object.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProviderRequestError("Provider response must contain exactly one completion.")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ProviderRequestError("Provider returned an invalid completion.")
        message = choice["message"]
        if message.get("tool_calls") or message.get("function_call"):
            raise UnsupportedCapability(
                "Provider returned tool calls; text-only output is required."
            )
        if message.get("role", "assistant") != "assistant" or not isinstance(
            message.get("content"), str
        ):
            raise ProviderRequestError("Provider completion must contain assistant text.")
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return Generation(
            text=message["content"],
            model=_identifier(payload.get("model")) or model,
            request_id=request_id or _identifier(payload.get("id")),
            usage=ProviderUsage(
                input_tokens=_token_count(usage.get("prompt_tokens")),
                output_tokens=_token_count(usage.get("completion_tokens")),
                total_tokens=_token_count(usage.get("total_tokens")),
            ),
            fingerprint=_identifier(payload.get("system_fingerprint")),
            finish_reason=_identifier(choice.get("finish_reason")),
        )
