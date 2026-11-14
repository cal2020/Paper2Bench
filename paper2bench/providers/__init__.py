"""Explicit text-generation adapter registry; no task-driven imports."""
from .base import ChatResponse, ProviderError
from .http import TextHTTPAdapter


def get_adapter(provider: str) -> TextHTTPAdapter:
    if provider not in {"openai", "anthropic", "ollama", "openai_compatible"}:
        raise ProviderError("unsupported_provider", "provider adapter is not installed")
    return TextHTTPAdapter(provider)


__all__ = ["ChatResponse", "ProviderError", "get_adapter"]
