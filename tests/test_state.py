"""State transitions. Pure functions, no LLM, no clock of their own."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.interview import state as state_ops
from app.interview.modes import get_mode
from app.interview.schemas import AnswerEvaluation, InterviewStatus, Question
from app.interview.state import InterviewState, InvalidInterviewState

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
MODE = get_mode("ml_fundamentals")


def new_state() -> InterviewState:
    return InterviewState(
        interview_id="interview-1",
        interview_type=MODE.key,
        remaining_topics=list(MODE.topics),
        time_limit_seconds=MODE.time_limit_seconds,
        created_at=NOW,
    )


def question(index: int = 1, topic: str = "regularization") -> Question:
    return Question(
        id=f"q{index}",
        index=index,
        text=f"Question {index}?",
        topic=topic,
        difficulty=3,
        action="open_question",
        asked_at=NOW,
    )


def evaluation(**overrides: object) -> AnswerEvaluation:
    payload: dict[str, object] = {
        "correctness": 7.0,
        "depth": 6.0,
        "communication": 8.0,
        "concepts_covered": ["L2 penalty"],
        "missing_concepts": ["early stopping"],
        "misconceptions": [],
        "recommended_action": "probe_deeper",
        "rationale": "ok",
    }
    payload.update(overrides)
    return AnswerEvaluation.model_validate(payload)


def test_start_marks_the_interview_in_progress() -> None:
    state = state_ops.start(new_state(), NOW)
    assert state.status is InterviewStatus.IN_PROGRESS
    assert state.started_at == NOW


def test_record_question_tracks_topic_coverage_and_pending_question() -> None:
    state = state_ops.record_question(new_state(), question(topic="regularization"))

    assert state.pending_question is not None
    assert state.current_topic == "regularization"
    assert state.topics_covered == ["regularization"]
    assert "regularization" not in state.remaining_topics


def test_record_question_does_not_duplicate_a_revisited_topic() -> None:
    state = state_ops.record_question(new_state(), question(1))
    state = state_ops.record_question(state, question(2))

    assert state.topics_covered == ["regularization"]
    assert len(state.questions_asked) == 2


def test_record_answer_clears_the_pending_question_and_stores_the_turn() -> None:
    state = state_ops.record_question(new_state(), question())
    state = state_ops.record_answer(state, answer="my answer", evaluation=evaluation())

    assert state.pending_question is None
    assert state.answered_count == 1
    assert state.turns[0].answer == "my answer"


def test_record_answer_requires_an_outstanding_question() -> None:
    with pytest.raises(InvalidInterviewState):
        state_ops.record_answer(new_state(), answer="hello", evaluation=evaluation())


def test_record_answer_accumulates_strengths_weaknesses_and_concept_scores() -> None:
    state = state_ops.record_question(new_state(), question())
    state = state_ops.record_answer(state, answer="a", evaluation=evaluation())

    assert state.strengths == ["L2 penalty"]
    assert state.weaknesses == ["early stopping"]
    assert state.concept_scores["l2 penalty"] == 7.0
    assert state.concept_scores["early stopping"] == 0.0


def test_concept_scores_average_across_turns() -> None:
    state = state_ops.record_question(new_state(), question(1))
    state = state_ops.record_answer(state, answer="a", evaluation=evaluation(correctness=8.0))
    state = state_ops.record_question(state, question(2))
    state = state_ops.record_answer(state, answer="b", evaluation=evaluation(correctness=4.0))

    assert state.concept_scores["l2 penalty"] == 6.0
    # Repeated gaps stay at the floor rather than drifting upward.
    assert state.concept_scores["early stopping"] == 0.0


def test_strengths_are_deduplicated_case_insensitively() -> None:
    state = state_ops.record_question(new_state(), question(1))
    state = state_ops.record_answer(state, answer="a", evaluation=evaluation())
    state = state_ops.record_question(state, question(2))
    state = state_ops.record_answer(
        state, answer="b", evaluation=evaluation(concepts_covered=["l2 penalty", "dropout"])
    )

    assert state.strengths == ["L2 penalty", "dropout"]


def test_apply_plan_clamps_difficulty_into_range() -> None:
    state = state_ops.apply_plan(new_state(), difficulty=9, follow_up_depth=2)
    assert state.difficulty == 5

    state = state_ops.apply_plan(state, difficulty=0, follow_up_depth=0)
    assert state.difficulty == 1


def test_complete_is_idempotent_and_drops_the_pending_question() -> None:
    state = state_ops.record_question(state_ops.start(new_state(), NOW), question())
    completed = state_ops.complete(state, NOW + timedelta(minutes=5))

    assert completed.status is InterviewStatus.COMPLETED
    assert completed.pending_question is None
    assert state_ops.complete(completed, NOW + timedelta(hours=1)) == completed


def test_elapsed_and_remaining_time_track_the_clock() -> None:
    state = state_ops.start(new_state(), NOW)
    later = NOW + timedelta(minutes=10)

    assert state.elapsed_seconds(later) == 600
    assert state.remaining_seconds(later) == MODE.time_limit_seconds - 600
    assert state.remaining_seconds(NOW + timedelta(hours=5)) == 0


def test_elapsed_time_freezes_once_completed() -> None:
    state = state_ops.complete(state_ops.start(new_state(), NOW), NOW + timedelta(minutes=3))
    assert state.elapsed_seconds(NOW + timedelta(hours=2)) == 180
