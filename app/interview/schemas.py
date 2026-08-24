"""Domain types for the interview loop.

`AnswerEvaluation` and `QuestionDraft` are the *only* structures the LLM is
allowed to produce. Everything else (state, difficulty, scoring, control flow)
is computed by code.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCORE_MIN = 0.0
SCORE_MAX = 10.0

NextAction = Literal[
    "probe_deeper",
    "clarify",
    "challenge_assumption",
    "ask_tradeoff",
    "increase_difficulty",
    "decrease_difficulty",
    "change_topic",
    "end_topic",
    "end_interview",
]

FOLLOW_UP_ACTIONS: frozenset[str] = frozenset(
    {"probe_deeper", "clarify", "challenge_assumption", "ask_tradeoff"}
)

DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 5


class InterviewStatus(str, Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class DimensionScore(BaseModel):
    """Mode-specific score. A list rather than a dict so the JSON schema stays
    strict-mode friendly for providers with native structured output."""

    model_config = ConfigDict(extra="forbid")

    name: str
    score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)


class AnswerEvaluation(BaseModel):
    """Validated structured output of the evaluator.

    `recommended_action` is *advice*. The planner decides what actually
    happens; see `app.interview.planner`.
    """

    model_config = ConfigDict(extra="forbid")

    correctness: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    depth: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    communication: float = Field(ge=SCORE_MIN, le=SCORE_MAX)

    dimension_scores: list[DimensionScore] = Field(default_factory=list)

    concepts_covered: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)

    recommended_action: NextAction
    follow_up_topic: str | None = None
    rationale: str = ""

    def scores_by_dimension(self) -> dict[str, float]:
        scores = {
            "technical_correctness": self.correctness,
            "technical_depth": self.depth,
            "communication": self.communication,
        }
        for dimension in self.dimension_scores:
            scores.setdefault(dimension.name, dimension.score)
        return scores


class QuestionDraft(BaseModel):
    """Structured output of the interviewer. Difficulty/topic bookkeeping is
    applied by code, not taken from the model."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    topic: str = Field(min_length=1)


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    index: int
    text: str
    topic: str
    difficulty: int = Field(ge=DIFFICULTY_MIN, le=DIFFICULTY_MAX)
    action: str
    asked_at: datetime


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: Question
    answer: str
    evaluation: AnswerEvaluation
    evaluation_degraded: bool = False


class InterviewReport(BaseModel):
    interview_id: str
    interview_type: str
    overall_score: float
    dimension_scores: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]
    evidence: list[str]
    missed_concepts: list[str]
    recommended_topics: list[str]
    unaddressed_target_skills: list[str] = []
    questions_answered: int
    duration_seconds: int
