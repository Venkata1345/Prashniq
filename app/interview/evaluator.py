"""Answer evaluation: the LLM's first responsibility.

It produces a validated `AnswerEvaluation` and nothing else. It does not touch
state and its `recommended_action` is advisory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.interview.grounding import Grounding
from app.interview.modes import InterviewMode
from app.interview.schemas import AnswerEvaluation, Question
from app.interview.state import InterviewState
from app.llm.gateway import LLMGateway
from app.llm.schemas import LLMCallContext, LLMError, LLMMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the evaluation component of a technical interview \
system for AI/ML engineering roles.

You do not talk to the candidate. You analyse one answer and return structured \
data for the interview engine.

Interview: {display_name} — {focus}
Current difficulty: {difficulty} of 5 (1 = junior, 5 = staff level)

Score every dimension from 0 to 10:
- correctness: factual and conceptual accuracy
- depth: how far past a surface-level answer the candidate went
- communication: structure, precision and clarity
{extra_dimensions}

Also return, for this specific interview mode, a `dimension_scores` entry for \
each of: {dimension_list}.

Be calibrated and strict. A fluent answer that avoids specifics is not a good \
answer. Do not inflate scores.

`recommended_action` is a recommendation to the interview engine, which may \
override it. Choose the action a strong human interviewer would take next.
"""

_FALLBACK_RATIONALE = (
    "Automatic evaluation failed; a neutral placeholder was recorded so the "
    "interview could continue."
)


@dataclass(frozen=True)
class EvaluationOutcome:
    evaluation: AnswerEvaluation
    degraded: bool


class Evaluator:
    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def evaluate(
        self,
        *,
        question: Question,
        answer: str,
        state: InterviewState,
        mode: InterviewMode,
        grounding: Grounding | None = None,
        request_id: str | None = None,
    ) -> EvaluationOutcome:
        context = LLMCallContext(
            request_id=request_id, interview_id=state.interview_id, purpose="evaluate_answer"
        )
        try:
            evaluation = await self._gateway.generate_structured(
                AnswerEvaluation,
                [
                    LLMMessage(
                        role="user",
                        content=_user_prompt(
                            question, answer, state, grounding or Grounding()
                        ),
                    )
                ],
                system=_system_prompt(mode, state),
                context=context,
            )
            return EvaluationOutcome(evaluation=evaluation, degraded=False)
        except LLMError as exc:
            # Avoid silent failure: log loudly, degrade explicitly, keep going.
            logger.error(
                "evaluation_failed interview_id=%s question_index=%d error=%s",
                state.interview_id,
                question.index,
                exc,
            )
            return EvaluationOutcome(evaluation=fallback_evaluation(), degraded=True)


def fallback_evaluation() -> AnswerEvaluation:
    """Neutral, non-scoring evaluation used when the LLM cannot be trusted.

    It deliberately moves the interview on rather than probing a thread we have
    no reliable analysis of.
    """
    return AnswerEvaluation(
        correctness=5.0,
        depth=5.0,
        communication=5.0,
        recommended_action="change_topic",
        rationale=_FALLBACK_RATIONALE,
    )


def _system_prompt(mode: InterviewMode, state: InterviewState) -> str:
    extra = mode.extra_dimensions
    return SYSTEM_PROMPT.format(
        display_name=mode.display_name,
        focus=mode.focus,
        difficulty=state.difficulty,
        extra_dimensions="\n".join(f"- {name}" for name in extra),
        dimension_list=", ".join(mode.dimensions),
    )


def _user_prompt(
    question: Question, answer: str, state: InterviewState, grounding: Grounding
) -> str:
    covered = ", ".join(state.topics_covered) or "none yet"
    target_skill = state.target_skill(question.topic)
    target_line = (
        f"Skill the target role requires: {target_skill}\n" if target_skill else ""
    )
    weaknesses = ", ".join(state.weaknesses[:8]) or "none recorded"
    return (
        f"Topic: {question.topic}\n"
        f"Questions answered so far: {state.answered_count}\n"
        f"Follow-up depth on this thread: {state.follow_up_depth}\n"
        f"Topics already covered: {covered}\n"
        f"{target_line}"
        f"Weaknesses observed so far: {weaknesses}\n"
        f"{_reference_block(grounding)}"
        f"{_claims_block(grounding)}"
        f"\nQUESTION:\n{question.text}\n\n"
        f"CANDIDATE ANSWER:\n{answer.strip() or '(no answer given)'}"
    )


def _reference_block(grounding: Grounding) -> str:
    """Retrieved reference points. Grounding what "missing concepts" is measured
    against is the whole reason the knowledge base exists."""
    if not grounding.knowledge_notes:
        return ""
    notes = "\n".join(f"- {note}" for note in grounding.knowledge_notes)
    return (
        "\nREFERENCE POINTS for this topic. Judge completeness against these and "
        "report what the answer missed relative to them. These are notes for you, "
        "not a rubric to quote:\n" + notes + "\n"
    )


def _claims_block(grounding: Grounding) -> str:
    if not grounding.resume_evidence:
        return ""
    claims = "\n".join(f"- {claim}" for claim in grounding.resume_evidence)
    return (
        "\nThe candidate claims the following on their resume; weigh the answer "
        "against what someone who did this work should know:\n" + claims + "\n"
    )
