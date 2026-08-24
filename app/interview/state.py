"""Explicit interview state and pure transitions.

The conversation transcript is *not* the memory of this system; this model is.
Every function here is pure: it takes a state and returns a new one, so state
changes are testable without an LLM, a clock or a database.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.context.schemas import InterviewBlueprint, ResumeProfile
from app.interview.schemas import (
    AnswerEvaluation,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    InterviewStatus,
    Question,
    Turn,
)


class InterviewState(BaseModel):
    interview_id: str
    candidate_id: str | None = None
    interview_type: str
    status: InterviewStatus = InterviewStatus.CREATED

    # Set at creation when a resume/job description was supplied; it is what the
    # topic list, its ordering and the interviewer's grounding come from.
    context_id: str | None = None
    blueprint: InterviewBlueprint | None = None
    resume: ResumeProfile | None = None

    difficulty: int = Field(default=3, ge=DIFFICULTY_MIN, le=DIFFICULTY_MAX)
    current_topic: str | None = None
    topics_covered: list[str] = Field(default_factory=list)
    remaining_topics: list[str] = Field(default_factory=list)

    questions_asked: list[Question] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    pending_question: Question | None = None

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    concept_scores: dict[str, float] = Field(default_factory=dict)

    follow_up_depth: int = 0

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    time_limit_seconds: int = 45 * 60

    @property
    def answered_count(self) -> int:
        return len(self.turns)

    def elapsed_seconds(self, now: datetime) -> int:
        if self.started_at is None:
            return 0
        end = self.completed_at or now
        return max(0, int((end - self.started_at).total_seconds()))

    def remaining_seconds(self, now: datetime) -> int:
        return max(0, self.time_limit_seconds - self.elapsed_seconds(now))

    def last_question(self) -> Question | None:
        return self.questions_asked[-1] if self.questions_asked else None

    def topic_evidence(self, topic: str | None) -> list[str]:
        """Resume claims the blueprint attached to a planned topic.

        Only what was planned: finding evidence for an unplanned topic is
        retrieval, and belongs in `app.interview.grounding`.
        """
        planned = self.blueprint.find(topic) if self.blueprint else None
        return list(planned.evidence) if planned else []

    def target_skill(self, topic: str | None) -> str | None:
        planned = self.blueprint.find(topic) if self.blueprint else None
        return planned.target_skill if planned else None


def start(state: InterviewState, now: datetime) -> InterviewState:
    return state.model_copy(
        update={"status": InterviewStatus.IN_PROGRESS, "started_at": now}
    )


def record_question(state: InterviewState, question: Question) -> InterviewState:
    topics_covered = list(state.topics_covered)
    if question.topic not in topics_covered:
        topics_covered.append(question.topic)
    remaining = [t for t in state.remaining_topics if t != question.topic]

    return state.model_copy(
        update={
            "questions_asked": [*state.questions_asked, question],
            "pending_question": question,
            "current_topic": question.topic,
            "topics_covered": topics_covered,
            "remaining_topics": remaining,
            "difficulty": question.difficulty,
        }
    )


def record_answer(
    state: InterviewState,
    *,
    answer: str,
    evaluation: AnswerEvaluation,
    degraded: bool = False,
) -> InterviewState:
    """Fold an evaluated answer into state. The evaluation never sets control
    fields (difficulty, topic) — the planner does that via `apply_plan`."""
    if state.pending_question is None:
        raise InvalidInterviewState("no question is awaiting an answer")

    turn = Turn(
        question=state.pending_question,
        answer=answer,
        evaluation=evaluation,
        evaluation_degraded=degraded,
    )

    return state.model_copy(
        update={
            "turns": [*state.turns, turn],
            "pending_question": None,
            "concept_scores": _updated_concept_scores(state, evaluation),
            "strengths": _merge(state.strengths, evaluation.concepts_covered),
            "weaknesses": _merge(
                state.weaknesses, evaluation.missing_concepts + evaluation.misconceptions
            ),
        }
    )


def apply_plan(
    state: InterviewState, *, difficulty: int, follow_up_depth: int
) -> InterviewState:
    return state.model_copy(
        update={
            "difficulty": max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, difficulty)),
            "follow_up_depth": max(0, follow_up_depth),
        }
    )


def complete(state: InterviewState, now: datetime) -> InterviewState:
    if state.status is InterviewStatus.COMPLETED:
        return state
    return state.model_copy(
        update={
            "status": InterviewStatus.COMPLETED,
            "completed_at": now,
            "pending_question": None,
        }
    )


class InvalidInterviewState(RuntimeError):
    """Raised when a transition is attempted from a state that does not allow
    it (e.g. answering when no question is outstanding)."""


def _updated_concept_scores(
    state: InterviewState, evaluation: AnswerEvaluation
) -> dict[str, float]:
    """Running mean per concept. Covered concepts are credited with the answer's
    correctness; missing concepts and misconceptions score 0 for this turn."""
    scores = dict(state.concept_scores)
    observations: list[tuple[str, float]] = [
        *((c, evaluation.correctness) for c in evaluation.concepts_covered),
        *((c, 0.0) for c in evaluation.missing_concepts),
        *((c, 0.0) for c in evaluation.misconceptions),
    ]
    for concept, value in observations:
        key = concept.strip().lower()
        if not key:
            continue
        previous = scores.get(key)
        scores[key] = value if previous is None else round((previous + value) / 2, 2)
    return scores


def _merge(existing: list[str], new: list[str]) -> list[str]:
    merged = list(existing)
    seen = {item.strip().lower() for item in merged}
    for item in new:
        key = item.strip().lower()
        if key and key not in seen:
            merged.append(item.strip())
            seen.add(key)
    return merged
