"""Unified LLM wrapper that dispatches to OpenAI or Anthropic.

Model-name routing:
  - ``claude-*``                        → Anthropic
  - ``gpt-*``, ``o1-*``, ``o3-*``, ``o4-*`` (and anything else)  → OpenAI

API keys are read from the env vars ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``
(or can be passed explicitly).  The ``anthropic`` package is a soft
dependency — only required when a Claude model is used.

Call sites pass OpenAI-style messages (``[{"role": "system"|"user"|"assistant",
"content": "..."}, ...]``); the wrapper handles the Anthropic convention of
taking the system prompt as a separate argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str) -> str:
    """Return ``"anthropic"`` or ``"openai"`` based on the model name."""
    name = (model or "").lower()
    if name.startswith("claude-") or name.startswith("anthropic/"):
        return "anthropic"
    return "openai"


# ---------------------------------------------------------------------------
# Unified response type
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = "openai"
    model: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> ChatResponse:
    """Send a chat-completion request to whichever provider owns ``model``.

    Parameters
    ----------
    model : str
        Model ID (e.g. ``"gpt-4o"``, ``"claude-opus-4-6"``).
    messages : list[dict]
        OpenAI-style messages.  ``role`` in ``{"system", "user", "assistant"}``.
        For Anthropic, system messages are extracted and passed via the
        dedicated ``system`` argument.
    max_tokens : int
        Upper bound on output tokens.
    temperature : float
        Sampling temperature.
    api_key : str, optional
        Explicit key; falls back to the provider's env var.

    Returns
    -------
    ChatResponse
    """
    provider = detect_provider(model)
    if provider == "anthropic":
        return _chat_anthropic(model, messages, max_tokens, temperature, api_key)
    return _chat_openai(model, messages, max_tokens, temperature, api_key)


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

def _chat_openai(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    api_key: Optional[str],
) -> ChatResponse:
    try:
        import openai
    except ImportError as e:
        raise RuntimeError("openai package required (pip install openai)") from e

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set for model " + model)

    client = openai.OpenAI(api_key=key)

    # Newer OpenAI models accept only max_completion_tokens; older accept
    # max_tokens.  Try the newer one first, fall back on TypeError for
    # pinned older SDKs.
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    try:
        resp = client.chat.completions.create(
            max_completion_tokens=max_tokens, **kwargs,
        )
    except TypeError:
        resp = client.chat.completions.create(
            max_tokens=max_tokens, **kwargs,
        )

    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return ChatResponse(
        content=content,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        provider="openai",
        model=model,
    )


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

def _chat_anthropic(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    api_key: Optional[str],
) -> ChatResponse:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package required for Claude models "
            "(pip install anthropic)"
        ) from e

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY must be set for model " + model)

    # Strip any 'anthropic/' prefix
    actual_model = model.split("/", 1)[1] if model.startswith("anthropic/") else model

    # Anthropic takes system as a dedicated arg, not a message role.
    system_parts = []
    converted = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content", ""))
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": m.get("content", "")})
        # ignore other roles

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=actual_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="\n\n".join(p for p in system_parts if p) or None,
        messages=converted,
    )

    # Concatenate all text blocks in the response
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    )

    usage = getattr(resp, "usage", None)
    return ChatResponse(
        content=text,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=(getattr(usage, "input_tokens", 0) or 0)
                     + (getattr(usage, "output_tokens", 0) or 0),
        provider="anthropic",
        model=model,
    )
