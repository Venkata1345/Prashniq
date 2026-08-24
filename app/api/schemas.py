"""Request/response DTOs.

Separate from the domain models so the wire contract can evolve independently
-- and so evaluations never leak to the candidate mid-interview.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.interview.schemas import InterviewStatus
from app.interview.state import InterviewState


class CreateInterviewRequest(BaseModel):
    interview_type: str = "ml_fundamentals"
    candidate_id: str | None = None
    # Required by resume_deep_dive and jd_targeted; optional elsewhere, where it
    # re-prioritises the mode's standing topics.
    context_id: str | None = None


class SubmitAnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=20_000)


class QuestionResponse(BaseModel):
    id: str
    index: int
    text: str
    topic: str
    difficulty: int
    asked_at: datetime


class InterviewResponse(BaseModel):
    interview_id: str
    interview_type: str
    status: InterviewStatus
    difficulty: int
    current_topic: str | None
    topics_covered: list[str]
    questions_asked: int
    questions_answered: int
    context_id: str | None
    pending_question: QuestionResponse | None

    @classmethod
    def from_state(cls, state: InterviewState) -> "InterviewResponse":
        return cls(
            interview_id=state.interview_id,
            interview_type=state.interview_type,
            status=state.status,
            difficulty=state.difficulty,
            current_topic=state.current_topic,
            topics_covered=state.topics_covered,
            questions_asked=len(state.questions_asked),
            questions_answered=state.answered_count,
            context_id=state.context_id,
            pending_question=(
                QuestionResponse(**state.pending_question.model_dump())
                if state.pending_question
                else None
            ),
        )


class AnswerAcceptedResponse(BaseModel):
    """Interview first, feedback later.

    The candidate gets the next question, not the evaluation. Scores and
    coaching arrive in the final report.
    """

    interview_id: str
    status: InterviewStatus
    next_question: QuestionResponse | None
    interview_complete: bool
