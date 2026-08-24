"""A deterministic in-process gateway.

Used by the whole test suite and by `LLM_PROVIDER=fake` for local development,
so the core loop can be exercised without paid API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.llm.gateway import LLMGateway
from app.llm.schemas import LLMCallContext, LLMMessage, LLMProviderError, LLMResponse, LLMUsage


@dataclass
class RecordedCall:
    messages: list[LLMMessage]
    system: str | None
    max_tokens: int
    context: LLMCallContext

    @property
    def purpose(self) -> str:
        return self.context.purpose

    @property
    def last_user_message(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""


Responder = Callable[[RecordedCall], str]


@dataclass
class FakeLLMGateway(LLMGateway):
    """Returns scripted text. Either a queue of `responses` or a `responder`
    function that inspects the call (purpose, prompt) and returns text."""

    responses: list[str] | None = None
    responder: Responder | None = None
    model: str = "fake-model"
    structured_attempts: int = 3
    calls: list[RecordedCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        LLMGateway.__init__(self, structured_attempts=self.structured_attempts)
        if self.responses is None and self.responder is None:
            raise ValueError("FakeLLMGateway needs `responses` or `responder`")
        self._queue = list(self.responses or [])

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        context: LLMCallContext | None = None,
    ) -> LLMResponse:
        call = RecordedCall(
            messages=list(messages),
            system=system,
            max_tokens=max_tokens,
            context=context or LLMCallContext(),
        )
        self.calls.append(call)

        if self.responder is not None:
            text = self.responder(call)
        elif self._queue:
            text = self._queue.pop(0)
        else:
            raise LLMProviderError(
                f"FakeLLMGateway ran out of scripted responses (call #{len(self.calls)}, "
                f"purpose={call.purpose})"
            )

        return LLMResponse(
            text=text,
            model=self.model,
            usage=LLMUsage(input_tokens=0, output_tokens=0),
            latency_ms=0,
        )
