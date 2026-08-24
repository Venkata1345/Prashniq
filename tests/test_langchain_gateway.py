"""LangChain gateway adapter: the framework boundary, tested with LangChain's
own fake chat model so no provider or network is involved."""

from __future__ import annotations

import json
from itertools import cycle

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.llm.factory import resolve_model
from app.llm.langchain_gateway import LangChainGateway
from app.llm.schemas import LLMMessage, LLMStructuredOutputError
from tests.conftest import evaluation_json
from app.interview.schemas import AnswerEvaluation


def gateway_returning(*texts: str) -> LangChainGateway:
    model = GenericFakeChatModel(messages=cycle([AIMessage(content=t) for t in texts]))
    return LangChainGateway(model, model_name="fake-lc")


class TestComplete:
    async def test_returns_the_models_text(self) -> None:
        gateway = gateway_returning("An interview question?")
        response = await gateway.complete(
            [LLMMessage(role="user", content="ask me something")], system="be terse"
        )

        assert response.text == "An interview question?"
        assert response.model == "fake-lc"

    async def test_joins_content_blocks(self) -> None:
        model = GenericFakeChatModel(
            messages=cycle(
                [AIMessage(content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}])]
            )
        )
        gateway = LangChainGateway(model, model_name="fake-lc")
        response = await gateway.complete([LLMMessage(role="user", content="hi")])

        assert response.text == "part one part two"

    async def test_conversation_roles_are_mapped(self) -> None:
        captured: dict = {}

        class Capturing(GenericFakeChatModel):
            def _generate(self, messages, *args, **kwargs):
                captured["types"] = [type(m).__name__ for m in messages]
                return super()._generate(messages, *args, **kwargs)

        model = Capturing(messages=cycle([AIMessage(content="ok")]))
        gateway = LangChainGateway(model, model_name="fake-lc")
        await gateway.complete(
            [
                LLMMessage(role="user", content="q"),
                LLMMessage(role="assistant", content="a"),
                LLMMessage(role="user", content="q2"),
            ],
            system="sys",
        )

        assert captured["types"] == [
            "SystemMessage",
            "HumanMessage",
            "AIMessage",
            "HumanMessage",
        ]


class TestStructured:
    """GenericFakeChatModel has no tool support, so these exercise the
    fallback path: prompt for JSON, validate, retry. The contract (validated
    model or LLMStructuredOutputError) is identical on the native path."""

    async def test_a_model_without_tool_support_still_produces_validated_output(
        self,
    ) -> None:
        gateway = gateway_returning(evaluation_json(correctness=8.0))
        evaluation = await gateway.generate_structured(
            AnswerEvaluation, [LLMMessage(role="user", content="evaluate")]
        )

        assert evaluation.correctness == 8.0

    async def test_the_retry_loop_survives_one_bad_completion(self) -> None:
        gateway = gateway_returning("not json at all", evaluation_json(correctness=6.0))
        evaluation = await gateway.generate_structured(
            AnswerEvaluation, [LLMMessage(role="user", content="evaluate")]
        )

        assert evaluation.correctness == 6.0

    async def test_structured_failure_raises_the_domain_error(self) -> None:
        gateway = gateway_returning("not json at all")
        with pytest.raises(LLMStructuredOutputError):
            await gateway.generate_structured(
                AnswerEvaluation, [LLMMessage(role="user", content="evaluate")]
            )


def test_model_resolution_prevents_cross_provider_404s() -> None:
    assert resolve_model("openai", "claude-opus-5") == "gpt-4o-mini"
    assert resolve_model("anthropic", "gpt-4o") == "claude-opus-5"
    assert resolve_model("openai", "gpt-4o") == "gpt-4o"
    assert resolve_model("anthropic", "claude-opus-5") == "claude-opus-5"
    assert resolve_model("groq", "gpt-4o-mini") == "openai/gpt-oss-120b"
    assert resolve_model("groq", "claude-opus-5") == "openai/gpt-oss-120b"
    assert resolve_model("groq", "llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"
    assert resolve_model("groq", "openai/gpt-oss-120b") == "openai/gpt-oss-120b"


def test_failed_tool_call_is_salvaged_from_the_error_body() -> None:
    """Groq 400s when the model adds extra fields but ships the JSON in the
    error body; the gateway recovers the payload instead of failing the turn."""
    import json

    from pydantic import BaseModel

    from app.llm.langchain_gateway import (
        _is_tool_validation_error,
        _salvage_failed_tool_call,
    )

    class Evaluation(BaseModel):
        correctness: float
        rationale: str

    class FakeGroqError(Exception):
        def __init__(self, body: dict) -> None:
            super().__init__("Error code: 400 - tool_use_failed")
            self.body = body

    generation = json.dumps(
        {
            "name": "Evaluation",
            # extra top-level field, exactly what Groq rejects
            "arguments": {"correctness": 7.0, "rationale": "solid", "reasoning": 5},
        }
    )
    exc = FakeGroqError({"error": {"code": "tool_use_failed", "failed_generation": generation}})

    salvaged = _salvage_failed_tool_call(exc, Evaluation)
    assert salvaged is not None
    assert salvaged.correctness == 7.0
    assert salvaged.rationale == "solid"
    assert _is_tool_validation_error(exc)

    # Unsalvageable garbage still identifies as retryable, not fatal.
    bad = FakeGroqError({"error": {"code": "tool_use_failed", "failed_generation": "not json"}})
    assert _salvage_failed_tool_call(bad, Evaluation) is None
    assert _is_tool_validation_error(bad)


def test_groq_provider_builds_a_langchain_gateway() -> None:
    from app.core.config import Settings
    from app.llm.factory import build_gateway
    from app.llm.langchain_gateway import LangChainGateway

    gateway = build_gateway(
        Settings(
            llm_provider="groq",
            llm_model="llama-3.3-70b-versatile",
            groq_api_key="gsk-test-not-real",
            embedding_provider="fake",
            vector_store="memory",
            database_url=None,
        )
    )
    assert isinstance(gateway, LangChainGateway)
    assert gateway.model == "llama-3.3-70b-versatile"
