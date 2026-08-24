"""Next-action logic. This is the layer that must not depend on the LLM
behaving sensibly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.interview import state as state_ops
from app.interview.modes import get_mode
from app.interview.planner import choose_next_action, plan_opening
from app.interview.schemas import AnswerEvaluation, Question
from app.interview.state import InterviewState

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
MODE = get_mode("ml_fundamentals")


def state_with(
    *,
    answered: int = 1,
    difficulty: int = 3,
    follow_up_depth: int = 0,
    remaining_topics: list[str] | None = None,
) -> InterviewState:
    state = InterviewState(
        interview_id="interview-1",
        interview_type=MODE.key,
        difficulty=difficulty,
        current_topic="regularization",
        follow_up_depth=follow_up_depth,
        remaining_topics=(
            list(MODE.topics[1:]) if remaining_topics is None else remaining_topics
        ),
        time_limit_seconds=MODE.time_limit_seconds,
        created_at=NOW,
    )
    state = state_ops.start(state, NOW)

    for index in range(1, answered + 1):
        state = state_ops.record_question(
            state,
            Question(
                id=f"q{index}",
                index=index,
                text=f"Question {index}?",
                topic="regularization",
                difficulty=difficulty,
                action="probe_deeper",
                asked_at=NOW,
            ),
        )
        state = state_ops.record_answer(state, answer="a", evaluation=evaluation())

    return state.model_copy(update={"follow_up_depth": follow_up_depth})


def evaluation(**overrides: object) -> AnswerEvaluation:
    payload: dict[str, object] = {
        "correctness": 6.0,
        "depth": 6.0,
        "communication": 6.0,
        "recommended_action": "probe_deeper",
        "rationale": "",
    }
    payload.update(overrides)
    return AnswerEvaluation.model_validate(payload)


def plan(state: InterviewState, evaluation_: AnswerEvaluation, now: datetime = NOW):
    return choose_next_action(evaluation=evaluation_, state=state, mode=MODE, now=now)


def test_opening_plan_takes_the_first_topic() -> None:
    state = InterviewState(
        interview_id="i",
        interview_type=MODE.key,
        remaining_topics=list(MODE.topics),
        created_at=NOW,
    )
    opening = plan_opening(state, MODE)

    assert opening.action == "open_question"
    assert opening.topic == MODE.topics[0]


def test_follow_up_keeps_the_topic_and_increments_depth() -> None:
    result = plan(state_with(follow_up_depth=0), evaluation(recommended_action="probe_deeper"))

    assert result.action == "probe_deeper"
    assert result.topic == "regularization"
    assert result.follow_up_depth == 1


def test_follow_up_depth_cap_forces_a_topic_change() -> None:
    state = state_with(follow_up_depth=MODE.max_follow_up_depth)
    result = plan(state, evaluation(recommended_action="probe_deeper"))

    assert result.action == "change_topic"
    assert result.topic == state.remaining_topics[0]
    assert result.follow_up_depth == 0
    assert "depth exhausted" in result.reason


def test_increase_difficulty_is_ignored_when_the_candidate_is_struggling() -> None:
    result = plan(
        state_with(difficulty=3),
        evaluation(correctness=2.0, recommended_action="increase_difficulty"),
    )

    assert result.action == "decrease_difficulty"
    assert result.difficulty == 2


def test_decrease_difficulty_is_ignored_after_a_strong_answer() -> None:
    result = plan(
        state_with(difficulty=3),
        evaluation(correctness=9.0, recommended_action="decrease_difficulty"),
    )

    assert result.action == "probe_deeper"
    assert result.difficulty == 3


def test_difficulty_is_clamped_at_the_ceiling_and_floor() -> None:
    high = plan(
        state_with(difficulty=5),
        evaluation(correctness=9.0, recommended_action="increase_difficulty"),
    )
    low = plan(
        state_with(difficulty=1),
        evaluation(correctness=1.0, recommended_action="decrease_difficulty"),
    )

    assert high.difficulty == 5
    assert low.difficulty == 1


def test_question_budget_ends_the_interview() -> None:
    result = plan(
        state_with(answered=MODE.max_questions),
        evaluation(recommended_action="probe_deeper"),
    )

    assert result.ends_interview
    assert "question budget" in result.reason


def test_expired_time_limit_ends_the_interview() -> None:
    state = state_with()
    result = plan(state, evaluation(), now=NOW + timedelta(seconds=MODE.time_limit_seconds + 1))

    assert result.ends_interview
    assert "time limit" in result.reason


def test_running_out_of_topics_ends_the_interview() -> None:
    result = plan(
        state_with(remaining_topics=[]), evaluation(recommended_action="change_topic")
    )

    assert result.ends_interview
    assert "no topics left" in result.reason


def test_the_model_cannot_end_the_interview_early() -> None:
    result = plan(state_with(answered=1), evaluation(recommended_action="end_interview"))

    assert result.action == "change_topic"
    assert "too early" in result.reason


def test_the_model_may_end_the_interview_once_enough_ground_is_covered() -> None:
    result = plan(
        state_with(answered=MODE.max_questions - 1),
        evaluation(recommended_action="end_interview"),
    )

    assert result.ends_interview


def test_end_topic_moves_to_the_next_topic() -> None:
    state = state_with()
    result = plan(state, evaluation(recommended_action="end_topic"))

    assert result.action == "change_topic"
    assert result.topic == state.remaining_topics[0]


def test_follow_up_topic_hint_is_used_when_provided() -> None:
    result = plan(
        state_with(),
        evaluation(recommended_action="ask_tradeoff", follow_up_topic="weight decay"),
    )

    assert result.topic == "weight decay"
