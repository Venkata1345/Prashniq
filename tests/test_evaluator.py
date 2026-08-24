"""Evaluator: schema validation, structured-output retry, and failure handling."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.interview.evaluator import Evaluator
from app.interview.modes import get_mode
from app.interview.schemas import AnswerEvaluation, Question
from app.interview.state import InterviewState
from app.llm.fake import FakeLLMGateway
from app.llm.gateway import LLMGateway
from app.llm.schemas import LLMMessage, LLMStructuredOutputError
from tests.conftest import evaluation_json

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def make_question() -> Question:
    return Question(
        id="q1",
        index=1,
        text="Explain the bias-variance tradeoff.",
        topic="bias-variance tradeoff",
        difficulty=3,
        action="open_question",
        asked_at=NOW,
    )


def make_state() -> InterviewState:
    mode = get_mode("ml_fundamentals")
    return InterviewState(
        interview_id="interview-1",
        interview_type=mode.key,
        remaining_topics=list(mode.topics),
        created_at=NOW,
    )


class TestSchemaValidation:
    def test_accepts_a_well_formed_evaluation(self) -> None:
        evaluation = AnswerEvaluation.model_validate_json(evaluation_json())
        assert evaluation.correctness == 6.0
        assert evaluation.recommended_action == "probe_deeper"
        assert evaluation.scores_by_dimension()["tradeoff_awareness"] == 4.0

    @pytest.mark.parametrize("score", [-1.0, 10.5, 99])
    def test_rejects_out_of_range_scores(self, score: float) -> None:
        with pytest.raises(ValidationError):
            AnswerEvaluation.model_validate_json(evaluation_json(correctness=score))

    def test_rejects_unknown_action(self) -> None:
        with pytest.raises(ValidationError):
            AnswerEvaluation.model_validate_json(
                evaluation_json(recommended_action="fire_the_candidate")
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AnswerEvaluation.model_validate_json(evaluation_json(hire=True))

    def test_requires_an_action(self) -> None:
        payload = json.loads(evaluation_json())
        del payload["recommended_action"]
        with pytest.raises(ValidationError):
            AnswerEvaluation.model_validate(payload)


class TestStructuredOutputRetry:
    async def test_extracts_json_from_a_fenced_response(self) -> None:
        gateway = FakeLLMGateway(
            responses=[f"Here you go:\n```json\n{evaluation_json()}\n```"]
        )
        evaluation = await _generate(gateway)
        assert evaluation.correctness == 6.0
        assert len(gateway.calls) == 1

    async def test_retries_with_the_validation_error_then_succeeds(self) -> None:
        gateway = FakeLLMGateway(
            responses=["not json at all", evaluation_json(correctness=20), evaluation_json()]
        )
        evaluation = await _generate(gateway)

        assert evaluation.correctness == 6.0
        assert len(gateway.calls) == 3
        # The repair prompt must carry the reason the previous attempt failed.
        assert "not valid for the required schema" in gateway.calls[-1].messages[-1].content

    async def test_raises_after_exhausting_attempts(self) -> None:
        gateway = FakeLLMGateway(responses=["nope"] * 3, structured_attempts=3)
        with pytest.raises(LLMStructuredOutputError) as excinfo:
            await _generate(gateway)
        assert excinfo.value.attempts == 3
        assert len(gateway.calls) == 3


class TestEvaluator:
    async def test_returns_a_validated_evaluation(self) -> None:
        gateway = FakeLLMGateway(responses=[evaluation_json(correctness=8.0)])
        outcome = await Evaluator(gateway).evaluate(
            question=make_question(),
            answer="Bias is error from wrong assumptions; variance is sensitivity to data.",
            state=make_state(),
            mode=get_mode("ml_fundamentals"),
        )

        assert outcome.degraded is False
        assert outcome.evaluation.correctness == 8.0

    async def test_prompt_carries_the_question_and_answer(self) -> None:
        gateway = FakeLLMGateway(responses=[evaluation_json()])
        await Evaluator(gateway).evaluate(
            question=make_question(),
            answer="My answer about variance.",
            state=make_state(),
            mode=get_mode("ml_fundamentals"),
        )

        prompt = gateway.calls[0].last_user_message
        assert "Explain the bias-variance tradeoff." in prompt
        assert "My answer about variance." in prompt
        assert gateway.calls[0].context.purpose == "evaluate_answer"

    async def test_degrades_gracefully_when_the_provider_cannot_comply(self) -> None:
        gateway = FakeLLMGateway(responses=["garbage"] * 3)
        outcome = await Evaluator(gateway).evaluate(
            question=make_question(),
            answer="anything",
            state=make_state(),
            mode=get_mode("ml_fundamentals"),
        )

        # The interview survives, but the turn is explicitly marked untrusted
        # so scoring can exclude it.
        assert outcome.degraded is True
        assert outcome.evaluation.recommended_action == "change_topic"


async def _generate(gateway: LLMGateway) -> AnswerEvaluation:
    return await gateway.generate_structured(
        AnswerEvaluation, [LLMMessage(role="user", content="evaluate this")]
    )
