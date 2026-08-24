"""LangChain-backed implementation of the LLM gateway.

One adapter covers every chat provider LangChain supports (Anthropic, OpenAI,
and later BYOK providers) selected by a config string -- this replaces the
hand-written per-provider adapters. The rest of the app still talks to the
`LLMGateway` interface and never imports LangChain.

The chat model is injected, which is what makes this testable: tests pass a
`GenericFakeChatModel`, production passes `init_chat_model(...)`.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.llm.gateway import LLMGateway
from app.llm.schemas import (
    LLMCallContext,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMUsage,
)

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def build_chat_model(
    *,
    provider: str,
    model: str,
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> BaseChatModel:
    """Provider selection in one place, via LangChain's registry."""
    from langchain.chat_models import init_chat_model

    kwargs: dict = {"timeout": timeout_seconds, "max_retries": max_retries}
    if api_key:
        kwargs["api_key"] = api_key
    return init_chat_model(model, model_provider=provider, **kwargs)


class LangChainGateway(LLMGateway):
    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        model_name: str = "unknown",
        structured_attempts: int = 3,
    ) -> None:
        super().__init__(structured_attempts=structured_attempts)
        self._chat_model = chat_model
        self.model = model_name

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        context: LLMCallContext | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._chat_model.ainvoke(
                _as_lc_messages(messages, system), max_tokens=max_tokens
            )
        except TimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except Exception as exc:  # provider SDK errors vary; the boundary maps them
            raise _mapped_error(exc) from exc

        usage = _usage(response)
        latency_ms = self._log_usage(usage, started, context)
        return LLMResponse(
            text=_text_of(response),
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def generate_structured(
        self,
        schema: type[SchemaT],
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        context: LLMCallContext | None = None,
    ) -> SchemaT:
        """Native structured output where the provider supports it, with the
        same retry-then-fail contract as every gateway in this app.

        `include_raw=True` keeps the raw message for usage logging and turns
        parse failures into data instead of exceptions. A model with no tool
        support falls back to the base class's prompt-and-parse path.
        """
        try:
            structured = self._chat_model.with_structured_output(schema, include_raw=True)
        except NotImplementedError:
            return await super().generate_structured(
                schema, messages, system=system, max_tokens=max_tokens, context=context
            )
        last_error = ""

        for attempt in range(1, self.structured_attempts + 1):
            started = time.perf_counter()
            try:
                result = await structured.ainvoke(_as_lc_messages(messages, system))
            except TimeoutError as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except Exception as exc:
                # Some providers (Groq) validate tool calls server-side and 400
                # when the model adds extra fields — but ship the rejected JSON
                # in the error body. Pydantic ignores unknown fields, so that
                # payload usually validates; salvage it rather than failing.
                salvaged = _salvage_failed_tool_call(exc, schema)
                if salvaged is not None:
                    logger.info(
                        "structured_output_salvaged schema=%s interview_id=%s",
                        schema.__name__,
                        getattr(context, "interview_id", None),
                    )
                    return salvaged
                if _is_tool_validation_error(exc):
                    last_error = str(exc)[:500]
                    logger.warning(
                        "structured_output_invalid schema=%s attempt=%d/%d "
                        "interview_id=%s error=%s",
                        schema.__name__,
                        attempt,
                        self.structured_attempts,
                        getattr(context, "interview_id", None),
                        last_error,
                    )
                    continue
                raise _mapped_error(exc) from exc

            raw = result.get("raw")
            self._log_usage(_usage(raw), started, context)

            parsed = result.get("parsed")
            if parsed is not None:
                if isinstance(parsed, schema):
                    return parsed
                try:
                    return schema.model_validate(parsed)
                except ValidationError as exc:
                    last_error = str(exc)
            else:
                last_error = str(result.get("parsing_error") or "no parsed output")

            logger.warning(
                "structured_output_invalid schema=%s attempt=%d/%d interview_id=%s error=%s",
                schema.__name__,
                attempt,
                self.structured_attempts,
                getattr(context, "interview_id", None),
                last_error,
            )

        raise LLMStructuredOutputError(
            f"{schema.__name__} could not be produced after "
            f"{self.structured_attempts} attempts: {last_error}",
            attempts=self.structured_attempts,
        )

    def _log_usage(
        self, usage: LLMUsage, started: float, context: LLMCallContext | None
    ) -> int:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "llm_call model=%s purpose=%s interview_id=%s request_id=%s "
            "latency_ms=%d input_tokens=%s output_tokens=%s",
            self.model,
            getattr(context, "purpose", "generic"),
            getattr(context, "interview_id", None),
            getattr(context, "request_id", None),
            latency_ms,
            usage.input_tokens,
            usage.output_tokens,
        )
        return latency_ms


def _as_lc_messages(messages: Sequence[LLMMessage], system: str | None) -> list[BaseMessage]:
    lc_messages: list[BaseMessage] = []
    if system:
        lc_messages.append(SystemMessage(content=system))
    for message in messages:
        if message.role == "assistant":
            lc_messages.append(AIMessage(content=message.content))
        else:
            lc_messages.append(HumanMessage(content=message.content))
    return lc_messages


def _text_of(message: AIMessage | None) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    # Content blocks (Anthropic style): join the text parts.
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _usage(message: object) -> LLMUsage:
    metadata = getattr(message, "usage_metadata", None) or {}
    return LLMUsage(
        input_tokens=metadata.get("input_tokens", 0) or 0,
        output_tokens=metadata.get("output_tokens", 0) or 0,
    )


def _failed_generation(exc: Exception) -> str | None:
    """The provider's copy of the rejected tool call, if the error carries one."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            generation = error.get("failed_generation")
            if isinstance(generation, str):
                return generation
    return None


def _salvage_failed_tool_call(exc: Exception, schema: type[SchemaT]) -> SchemaT | None:
    generation = _failed_generation(exc)
    if not generation:
        return None
    import json

    try:
        data = json.loads(generation)
        arguments = data.get("arguments", data) if isinstance(data, dict) else data
        return schema.model_validate(arguments)
    except Exception:
        return None


def _is_tool_validation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tool_use_failed" in text or "tool call validation failed" in text


def _mapped_error(exc: Exception) -> LLMProviderError | LLMTimeoutError:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return LLMTimeoutError(str(exc))
    return LLMProviderError(str(exc))
