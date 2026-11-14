"""Provider contracts with explicit usage missingness and safe errors."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


class ProviderError(RuntimeError):
    """Stable error code without provider bodies or credentials."""
    def __init__(self, code: str, message: str, *, retryable: bool = False, outcome_unknown: bool = False):
        super().__init__(message)
        self.code, self.retryable, self.outcome_unknown = code, retryable, outcome_unknown


@dataclass
class ChatResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = "openai"
    model: str = ""
    usage_complete: bool = False
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    resolved_model: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> dict:
        return {"input_tokens": self.metadata.get("input_tokens_measured"),
                "output_tokens": self.metadata.get("output_tokens_measured"), "complete": self.usage_complete}
