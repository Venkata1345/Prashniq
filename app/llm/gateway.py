"""The LLM gateway abstraction.

Responsibilities that live *here* and not in the interview domain:
  - provider selection and credentials (later: app-managed vs BYOK)
  - prompt/response transport
  - structured-output coaxing, validation and retry
  - usage accounting hooks

`generate_structured` returns a validated Pydantic model or raises
`LLMStructuredOutputError`. Callers never see raw provider payloads.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.schemas import (
    LLMCallContext,
    LLMMessage,
    LLMResponse,
    LLMStructuredOutputError,
)

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMGateway(ABC):
    """Base gateway. Adapters implement `complete`; structured output has a
    working default built on top of it, which providers with native structured
    output may override."""

    def __init__(self, *, structured_attempts: int = 3) -> None:
        if structured_attempts < 1:
            raise ValueError("structured_attempts must be >= 1")
        self.structured_attempts = structured_attempts

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        context: LLMCallContext | None = None,
    ) -> LLMResponse:
        """Single completion call. Adapter-specific."""

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        context: LLMCallContext | None = None,
    ) -> str:
        response = await self.complete(
            messages, system=system, max_tokens=max_tokens, context=context
        )
        return response.text

    async def generate_structured(
        self,
        schema: type[SchemaT],
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        context: LLMCallContext | None = None,
    ) -> SchemaT:
        """Ask for JSON matching `schema`, validate it, retry on failure.

        Retries feed the validation error back to the model, which is far more
        effective than resampling blindly.
        """
        conversation = list(messages)
        instructed_system = _with_schema_instructions(system, schema)
        last_raw: str | None = None
        last_error: str = ""

        for attempt in range(1, self.structured_attempts + 1):
            response = await self.complete(
                conversation,
                system=instructed_system,
                max_tokens=max_tokens,
                context=context,
            )
            last_raw = response.text
            try:
                return schema.model_validate(_extract_json(response.text))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "structured_output_invalid schema=%s attempt=%d/%d interview_id=%s error=%s",
                    schema.__name__,
                    attempt,
                    self.structured_attempts,
                    getattr(context, "interview_id", None),
                    last_error,
                )
                if attempt == self.structured_attempts:
                    break
                conversation = [
                    *conversation,
                    LLMMessage(role="assistant", content=response.text),
                    LLMMessage(
                        role="user",
                        content=(
                            "That output was not valid for the required schema:\n"
                            f"{last_error}\n"
                            "Reply with corrected JSON only."
                        ),
                    ),
                ]

        raise LLMStructuredOutputError(
            f"{schema.__name__} could not be produced after "
            f"{self.structured_attempts} attempts: {last_error}",
            attempts=self.structured_attempts,
            last_raw=last_raw,
        )


def _with_schema_instructions(system: str | None, schema: type[BaseModel]) -> str:
    return "\n\n".join(
        part
        for part in (
            system,
            "Respond with a single JSON object and nothing else. No prose, no "
            "code fences. It must validate against this JSON schema:\n"
            + json.dumps(schema.model_json_schema(), indent=2),
        )
        if part
    )


def _extract_json(text: str) -> object:
    """Best-effort JSON extraction: bare JSON, fenced JSON, or JSON with
    surrounding chatter."""
    candidates = [text.strip()]
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("response contained no parsable JSON object")
