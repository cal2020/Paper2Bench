"""Backward-compatible chat facade with explicit scoped local/cloud routing.

Legacy calls retain name-based routing. Active profiles select role bindings;
historical default model arguments cannot silently override a local profile.
"""
from __future__ import annotations
import os
import math
from dataclasses import replace
from typing import Optional
from .providers import ChatResponse, ProviderError, get_adapter
from .runtime_config import ModelRef, get_context, get_role


def detect_provider(model: str) -> str:
    name = (model or "").lower()
    return "anthropic" if name.startswith("claude-") or name.startswith("anthropic/") else "openai"


def chat(model: str, messages: list[dict], *, max_tokens: int = 4096, temperature: float = 0.0,
         api_key: Optional[str] = None, role: Optional[str] = None,
         timeout_seconds: Optional[float] = None) -> ChatResponse:
    context = get_context()
    selected_role = role or get_role()
    if context:
        ref = context.resolve(selected_role)
        key = context.credential(ref, api_key)
    else:
        provider = detect_provider(model)
        ref = ModelRef.from_dict({"provider": provider, "model": model})
        key = api_key or os.getenv("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ProviderError("invalid_request", "timeout_seconds must be finite and positive")
        ref = replace(ref, timeout_seconds=min(ref.timeout_seconds, timeout_seconds))
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ProviderError("invalid_request", "max_tokens must be a positive integer")
    if ref.output_tokens is not None and max_tokens > ref.output_tokens:
        raise ProviderError("context_overflow", "requested output exceeds configured output token limit")
    if not isinstance(messages, list) or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
        raise ProviderError("invalid_request", "only text message lists are supported")
    # Conservative byte estimate when a verified endpoint tokenizer is absent.
    input_estimate = sum(len(m["content"].encode("utf-8")) + 64 for m in messages)
    if ref.context_tokens is not None and input_estimate + max_tokens > ref.context_tokens:
        raise ProviderError("context_overflow", "prompt plus output exceeds conservative context estimate; shorten evidence or increase context")
    adapter = get_adapter(ref.provider)
    if any(adapter.capabilities().get(name) != "supported" for name in ref.required_capabilities):
        raise ProviderError("unsupported_capability", "binding requires a capability unsupported by this adapter")
    # Proven local validation failures do not consume durable billed-work bounds.
    adapter.validate_request(ref, messages, max_tokens=max_tokens, temperature=temperature, api_key=key)
    reservation = context.ledger.reserve(selected_role, input_estimate, max_tokens) if context else None
    try:
        response = adapter.generate(ref, messages, max_tokens=max_tokens, temperature=temperature, api_key=key)
    except Exception:
        if reservation:
            # Possibly dispatched/billed calls retain conservative reservations.
            context.ledger.settle(reservation, None, None)
        raise
    if reservation:
        context.ledger.settle(reservation, response.metadata.get("input_tokens_measured"), response.metadata.get("output_tokens_measured"))
        response.metadata["budget_pool"] = reservation.pool
        response.metadata["usage_estimate"] = {"input_tokens": input_estimate, "output_tokens_reserved": max_tokens}
    response.metadata["role"] = selected_role
    return response


def _chat_openai(model, messages, max_tokens, temperature, api_key):
    ref = ModelRef.from_dict({"provider": "openai", "model": model})
    return get_adapter("openai").generate(ref, messages, max_tokens=max_tokens, temperature=temperature,
                                          api_key=api_key or os.getenv("OPENAI_API_KEY"))


def _chat_anthropic(model, messages, max_tokens, temperature, api_key):
    ref = ModelRef.from_dict({"provider": "anthropic", "model": model})
    return get_adapter("anthropic").generate(ref, messages, max_tokens=max_tokens, temperature=temperature,
                                             api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))


__all__ = ["chat", "ChatResponse", "ProviderError", "detect_provider"]
