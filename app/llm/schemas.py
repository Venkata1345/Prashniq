"""Provider-agnostic types for the LLM boundary.

Nothing in `app.interview` should ever import a provider SDK; it talks to the
gateway using these types only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant"]


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class LLMUsage(BaseModel):
    """Token accounting. Carried on every response so metering can be added
    later without touching the interview domain."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0


class LLMCallContext(BaseModel):
    """Correlation/attribution data that travels with a call but is not part of
    the prompt. Later this is where BYOK credentials and provider selection go.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    interview_id: str | None = None
    purpose: str = "generic"


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: int = 0


class LLMError(RuntimeError):
    """Base class for every failure that crosses the gateway boundary."""


class LLMTimeoutError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMStructuredOutputError(LLMError):
    """The provider could not produce output matching the requested schema,
    even after retries."""

    def __init__(self, message: str, *, attempts: int, last_raw: str | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw
