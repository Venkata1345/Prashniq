"""Deterministic interview control.

The evaluator *recommends* an action; this module decides. Keeping the decision
in code means budgets, difficulty bounds and termination are guaranteed rather
than hoped for, and they are testable without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from app.interview.modes import InterviewMode
from app.interview.schemas import (
    AnswerEvaluation,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    FOLLOW_UP_ACTIONS,
    NextAction,
)
from app.interview.state import InterviewState

PlannedActionName = Literal["open_question"] | NextAction

# Below this correctness, raising difficulty is counterproductive regardless of
# what the model asked for.
STRUGGLING_CORRECTNESS = 4.0
STRONG_CORRECTNESS = 8.0


@dataclass(frozen=True)
class PlannedAction:
    action: PlannedActionName
    topic: str | None
    difficulty: int
    follow_up_depth: int
    reason: str

    @property
    def ends_interview(self) -> bool:
        return self.action == "end_interview"


def plan_opening(state: InterviewState, mode: InterviewMode) -> PlannedAction:
    topic = state.remaining_topics[0] if state.remaining_topics else _fallback_topic(mode)
    return PlannedAction(
        action="open_question",
        topic=topic,
        difficulty=state.difficulty,
        follow_up_depth=0,
        reason="opening question",
    )


def choose_next_action(
    *,
    evaluation: AnswerEvaluation,
    state: InterviewState,
    mode: InterviewMode,
    now: datetime,
) -> PlannedAction:
    """Map an evaluation plus the current state onto the next concrete action."""

    if state.answered_count >= mode.max_questions:
        return _end("question budget reached", state)
    if state.remaining_seconds(now) <= 0:
        return _end("time limit reached", state)

    action: PlannedActionName = evaluation.recommended_action
    reason = "evaluator recommendation"

    # Guard against advice that contradicts the evidence.
    if action == "increase_difficulty" and evaluation.correctness < STRUGGLING_CORRECTNESS:
        action, reason = "decrease_difficulty", "low correctness overrides difficulty increase"
    elif action == "decrease_difficulty" and evaluation.correctness >= STRONG_CORRECTNESS:
        action, reason = "probe_deeper", "strong answer overrides difficulty decrease"

    # Cap how long we chase a single thread.
    if action in FOLLOW_UP_ACTIONS and state.follow_up_depth >= mode.max_follow_up_depth:
        action, reason = "change_topic", "follow-up depth exhausted"

    # Only code ends the interview early, and only once enough ground is covered.
    if action == "end_interview" and state.answered_count < _min_questions(mode):
        action, reason = "change_topic", "too early to end the interview"

    if action in ("change_topic", "end_topic"):
        if not state.remaining_topics:
            return _end("no topics left to cover", state)
        return PlannedAction(
            action="change_topic",
            topic=state.remaining_topics[0],
            difficulty=state.difficulty,
            follow_up_depth=0,
            reason=reason,
        )

    if action == "end_interview":
        return _end(reason, state)

    difficulty = state.difficulty
    if action == "increase_difficulty":
        difficulty = min(DIFFICULTY_MAX, difficulty + 1)
    elif action == "decrease_difficulty":
        difficulty = max(DIFFICULTY_MIN, difficulty - 1)

    topic = evaluation.follow_up_topic or state.current_topic
    depth = state.follow_up_depth + 1 if action in FOLLOW_UP_ACTIONS else 0

    return PlannedAction(
        action=action,
        topic=topic,
        difficulty=difficulty,
        follow_up_depth=depth,
        reason=reason,
    )


def _end(reason: str, state: InterviewState) -> PlannedAction:
    return PlannedAction(
        action="end_interview",
        topic=state.current_topic,
        difficulty=state.difficulty,
        follow_up_depth=state.follow_up_depth,
        reason=reason,
    )


def _fallback_topic(mode: InterviewMode) -> str:
    """Blueprint-driven modes have no standing topic list."""
    return mode.topics[0] if mode.topics else mode.display_name


def _min_questions(mode: InterviewMode) -> int:
    return max(2, mode.max_questions // 2)
