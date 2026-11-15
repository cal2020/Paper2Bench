"""Provider policy and failure tests; no live or billable model calls."""

import asyncio
import json
import time

import httpx
import pytest
from pydantic import ValidationError

from paper2bench.repro.providers import (
    OutcomeUnknown,
    ProviderClient,
    ProviderConfigurationError,
    ProviderProfile,
    ProviderRequestError,
    UnsupportedCapability,
)


def profile(**overrides):
    values = {"name": "test", "base_url": "http://127.0.0.1:9999/v1", "location": "local"}
    values.update(overrides)
    return ProviderProfile(**values)


def completion(**overrides):
    payload = {
        "id": "completion-1",
        "model": "actual-revision",
        "choices": [
            {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        "system_fingerprint": "fp-1",
    }
    payload.update(overrides)
    return payload


def generate(client, **overrides):
    kwargs = {
        "model": "requested-alias",
        "messages": [{"role": "user", "content": "question"}],
        "max_output_tokens": 50,
        "local_only": True,
    }
    kwargs.update(overrides)
    return client.generate(**kwargs)


def test_local_text_and_request_contract():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url == "http://127.0.0.1:9999/v1/chat/completions"
        body = json.loads(request.content)
        assert body == {
            "model": "requested-alias",
            "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 50,
            "stream": False,
            "n": 1,
        }
        assert "authorization" not in request.headers
        return httpx.Response(200, json=completion(), headers={"x-request-id": "request-1"})

    result = generate(ProviderClient(profile(), transport=httpx.MockTransport(handle)))
    assert len(requests) == 1
    assert result.text == "answer"
    assert result.model == "actual-revision"
    assert result.request_id == "request-1"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 5
    assert result.fingerprint == "fp-1"


def test_missing_usage_stays_unknown():
    client = ProviderClient(
        profile(),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion(usage=None))),
    )
    result = generate(client)
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None


def test_invalid_usage_is_not_coerced_to_measurement():
    client = ProviderClient(
        profile(),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json=completion(
                    usage={"prompt_tokens": True, "completion_tokens": "17", "total_tokens": -1}
                ),
            )
        ),
    )
    assert generate(client).usage.model_dump() == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_cloud_requires_explicit_policy_before_dispatch():
    def fail(request):
        pytest.fail("policy must reject before making a request")

    client = ProviderClient(
        profile(location="cloud", base_url="https://provider.invalid/v1"),
        transport=httpx.MockTransport(fail),
    )
    with pytest.raises(UnsupportedCapability, match="Local-only"):
        generate(client)


@pytest.mark.parametrize(
    "address", ["http://localhost/v1", "http://192.168.1.4/v1", "http://127.0.0.1.example/v1"]
)
def test_local_requires_numeric_loopback(address):
    with pytest.raises(ProviderConfigurationError):
        ProviderClient(profile(base_url=address)).validate(local_only=True)


@pytest.mark.parametrize("address", ["http://127.0.0.2:123/v1", "http://[::1]:123/v1"])
def test_literal_loopback_supported(address):
    ProviderClient(profile(base_url=address)).validate(local_only=True)


def test_cloud_http_is_rejected():
    with pytest.raises(ProviderConfigurationError, match="HTTPS"):
        ProviderClient(profile(location="cloud", base_url="http://provider.invalid/v1")).validate(
            local_only=False
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@provider.invalid/v1",
        "https://provider.invalid/v1?api_key=secret",
        "https://provider.invalid/v1#token",
        "https://provider.invalid:99999/v1",
        "https://provider.invalid/\nother",
    ],
)
def test_unsafe_urls_rejected(url):
    with pytest.raises(ValidationError):
        profile(base_url=url)


def test_secret_resolved_but_never_serialized(monkeypatch):
    monkeypatch.setenv("P2B_TEST_TOKEN", "secret-marker-783")
    config = profile(
        location="cloud", base_url="https://provider.invalid/v1", api_key_env="P2B_TEST_TOKEN"
    )

    def handle(request):
        assert request.headers["authorization"] == "Bearer secret-marker-783"
        return httpx.Response(200, json=completion())

    generate(ProviderClient(config, transport=httpx.MockTransport(handle)), local_only=False)
    assert "secret-marker-783" not in json.dumps(config.canonical_config())
    assert "secret-marker-783" not in repr(config)


def test_missing_credential_is_sanitized(monkeypatch):
    monkeypatch.delenv("P2B_MISSING_TOKEN", raising=False)
    with pytest.raises(ProviderConfigurationError) as error:
        ProviderClient(profile(api_key_env="P2B_MISSING_TOKEN")).validate(local_only=True)
    assert "P2B_MISSING_TOKEN" not in str(error.value)


@pytest.mark.parametrize("secret", ["secret marker", "secret\U0001f511", "secret\nmarker"])
def test_invalid_header_credential_never_enters_error(monkeypatch, secret):
    monkeypatch.setenv("P2B_TEST_TOKEN", secret)
    with pytest.raises(ProviderConfigurationError) as error:
        generate(ProviderClient(profile(api_key_env="P2B_TEST_TOKEN")))
    assert secret not in str(error.value)


def test_redirect_not_followed_or_retried():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(307, headers={"location": "https://elsewhere.invalid/secret"})

    with pytest.raises(ProviderRequestError, match="redirect"):
        generate(ProviderClient(profile(), transport=httpx.MockTransport(handle)))
    assert len(calls) == 1


def test_timeout_after_dispatch_unknown_not_retried():
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("sensitive provider details", request=request)

    with pytest.raises(OutcomeUnknown) as error:
        generate(ProviderClient(profile(), transport=httpx.MockTransport(handle)))
    assert len(calls) == 1
    assert "sensitive" not in str(error.value)


def test_connect_failure_is_not_reported_as_success():
    def handle(request):
        raise httpx.ConnectError("secret-marker", request=request)

    with pytest.raises(ProviderRequestError) as error:
        generate(ProviderClient(profile(), transport=httpx.MockTransport(handle)))
    assert "secret-marker" not in str(error.value)


def test_server_error_keeps_unknown_outcome_and_request_id():
    client = ProviderClient(
        profile(),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                503, text="secret-error-body", headers={"x-request-id": "req-503"}
            )
        ),
    )
    with pytest.raises(OutcomeUnknown) as error:
        generate(client)
    assert error.value.request_id == "req-503"
    assert "secret-error-body" not in str(error.value)


def test_rejection_does_not_leak_error_body():
    client = ProviderClient(
        profile(),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, text="secret-error-body")),
    )
    with pytest.raises(ProviderRequestError) as error:
        generate(client)
    assert error.value.status_code == 401
    assert "secret-error-body" not in str(error.value)


@pytest.mark.parametrize(
    "payload", [[], {}, {"choices": []}, {"choices": [{"message": {"content": None}}]}]
)
def test_malformed_response_rejected(payload):
    client = ProviderClient(
        profile(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    with pytest.raises(ProviderRequestError):
        generate(client)


def test_tool_output_rejected_explicitly():
    client = ProviderClient(
        profile(),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json=completion(
                    choices=[{"message": {"content": None, "tool_calls": [{"id": "call-1"}]}}]
                ),
            )
        ),
    )
    with pytest.raises(UnsupportedCapability, match="tool calls"):
        generate(client)


def test_multimodal_request_rejected_before_dispatch():
    with pytest.raises(UnsupportedCapability):
        generate(
            ProviderClient(profile()),
            messages=[{"role": "user", "content": [{"type": "image_url"}]}],
        )


def test_invalid_role_type_rejected_before_dispatch():
    with pytest.raises(UnsupportedCapability):
        generate(ProviderClient(profile()), messages=[{"role": ["user"], "content": "text"}])


def test_compressed_response_rejected_before_decoding():
    client = ProviderClient(
        profile(),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json=completion(), headers={"content-encoding": "identity"}
            )
        ),
    )
    assert generate(client).text == "answer"

    def handle(request):
        response = httpx.Response(200, content=b"{}")
        response.headers["content-encoding"] = "gzip"
        return response

    with pytest.raises(ProviderRequestError, match="Compressed"):
        generate(ProviderClient(profile(), transport=httpx.MockTransport(handle)))


def test_response_is_bounded():
    client = ProviderClient(
        profile(), transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 500))
    )
    client.MAX_RESPONSE_BYTES = 100
    with pytest.raises(ProviderRequestError, match="size"):
        generate(client)


@pytest.mark.parametrize("tokens", [0, -1, True, 1.5, 1_000_001])
def test_invalid_token_budget_before_request(tokens):
    with pytest.raises(ProviderConfigurationError):
        generate(ProviderClient(profile()), max_output_tokens=tokens)


def test_profile_immutable_and_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        profile(api_key="never-store-this")
    config = profile()
    with pytest.raises(ValidationError):
        config.location = "cloud"


@pytest.mark.parametrize("explicit_timeout", [None, 0.05])
def test_slow_trickle_obeys_whole_request_deadline_and_closes_stream(explicit_timeout):
    state = {"calls": 0, "yielded": 0, "closed": False, "finished": False}

    class TrickleStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for byte in json.dumps(completion()).encode():
                # Each chunk arrives well within the profile's per-I/O timeout.
                await asyncio.sleep(0.01)
                state["yielded"] += 1
                yield bytes([byte])
            state["finished"] = True

        async def aclose(self):
            state["closed"] = True

    async def handle(request):
        state["calls"] += 1
        return httpx.Response(
            200, stream=TrickleStream(), headers={"x-request-id": "trickle-request-1"}
        )

    client = ProviderClient(profile(timeout=0.1), transport=httpx.MockTransport(handle))
    started = time.monotonic()
    with pytest.raises(OutcomeUnknown, match="total deadline") as error:
        generate(client, total_timeout=explicit_timeout)
    assert time.monotonic() - started < 1.0
    assert state["calls"] == 1
    assert state["yielded"] > 0
    assert state["closed"] is True
    assert state["finished"] is False
    assert error.value.request_id == "trickle-request-1"


def test_total_deadline_covers_wait_for_response_headers():
    state = {"calls": 0, "cancelled": False}

    async def handle(request):
        state["calls"] += 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return httpx.Response(200, json=completion())

    client = ProviderClient(profile(timeout=1.0), transport=httpx.MockTransport(handle))
    with pytest.raises(OutcomeUnknown, match="total deadline") as error:
        generate(client, total_timeout=0.02)
    assert state == {"calls": 1, "cancelled": True}
    assert error.value.request_id is None


def test_async_applications_have_native_entrypoint_and_sync_fails_before_dispatch():
    state = {"calls": 0}

    def handle(request):
        state["calls"] += 1
        return httpx.Response(200, json=completion())

    client = ProviderClient(profile(), transport=httpx.MockTransport(handle))

    async def application():
        with pytest.raises(ProviderConfigurationError, match="await agenerate"):
            generate(client)
        assert state["calls"] == 0
        result = await client.agenerate(
            model="fixture",
            messages=[{"role": "user", "content": "question"}],
            max_output_tokens=50,
            local_only=True,
            total_timeout=1.0,
        )
        assert result.text == "answer"

    asyncio.run(application())
    assert state["calls"] == 1


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "1"])
def test_invalid_total_deadline_rejected_before_dispatch(timeout):
    def never_dispatch(request):
        pytest.fail("invalid deadlines must fail before dispatch")

    client = ProviderClient(profile(), transport=httpx.MockTransport(never_dispatch))
    with pytest.raises(ProviderConfigurationError, match="total_timeout"):
        generate(client, total_timeout=timeout)
