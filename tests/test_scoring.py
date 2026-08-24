"""Score aggregation and report assembly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.interview import state as state_ops
from app.interview.modes import get_mode
from app.interview.schemas import AnswerEvaluation, Question
from app.interview.scoring import build_report
from app.interview.state import InterviewState

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
MODE = get_mode("ml_fundamentals")


def evaluation(**overrides: object) -> AnswerEvaluation:
    payload: dict[str, object] = {
        "correctness": 8.0,
        "depth": 6.0,
        "communication": 7.0,
        "dimension_scores": [
            {"name": "reasoning", "score": 7.0},
            {"name": "tradeoff_awareness", "score": 5.0},
        ],
        "concepts_covered": ["L2 penalty"],
        "missing_concepts": ["early stopping"],
        "recommended_action": "probe_deeper",
        "rationale": "Solid.",
    }
    payload.update(overrides)
    return AnswerEvaluation.model_validate(payload)


def state_with_turns(*evaluations: AnswerEvaluation, degraded: set[int] = frozenset()):
    state = state_ops.start(
        InterviewState(
            interview_id="interview-1",
            interview_type=MODE.key,
            remaining_topics=list(MODE.topics),
            created_at=NOW,
        ),
        NOW,
    )
    for index, item in enumerate(evaluations, start=1):
        state = state_ops.record_question(
            state,
            Question(
                id=f"q{index}",
                index=index,
                text=f"Question {index}?",
                topic=MODE.topics[index - 1],
                difficulty=3,
                action="probe_deeper",
                asked_at=NOW,
            ),
        )
        state = state_ops.record_answer(
            state, answer=f"answer {index}", evaluation=item, degraded=index in degraded
        )
    return state_ops.complete(state, NOW + timedelta(minutes=12))


def test_dimension_scores_are_means_across_turns() -> None:
    report = build_report(
        state_with_turns(evaluation(), evaluation(correctness=4.0, depth=4.0)),
        MODE,
        NOW + timedelta(minutes=12),
    )

    assert report.dimension_scores["technical_correctness"] == 6.0
    assert report.dimension_scores["technical_depth"] == 5.0
    assert report.dimension_scores["reasoning"] == 7.0
    assert report.questions_answered == 2


def test_overall_score_is_the_mean_of_the_dimension_scores() -> None:
    report = build_report(state_with_turns(evaluation()), MODE, NOW)
    expected = round(sum(report.dimension_scores.values()) / len(report.dimension_scores), 2)

    assert report.overall_score == expected


def test_only_the_modes_dimensions_are_reported() -> None:
    report = build_report(
        state_with_turns(
            evaluation(dimension_scores=[{"name": "sql_skill", "score": 9.0}])
        ),
        MODE,
        NOW,
    )

    assert set(report.dimension_scores) <= set(MODE.dimensions)
    assert "sql_skill" not in report.dimension_scores


def test_degraded_turns_do_not_contribute_to_scores() -> None:
    report = build_report(
        state_with_turns(
            evaluation(correctness=8.0), evaluation(correctness=0.0), degraded={2}
        ),
        MODE,
        NOW,
    )

    assert report.dimension_scores["technical_correctness"] == 8.0
    # The turn still happened, so it is still counted as answered.
    assert report.questions_answered == 2


def test_report_separates_strengths_from_weaknesses_and_gaps() -> None:
    report = build_report(state_with_turns(evaluation()), MODE, NOW + timedelta(minutes=12))

    assert "L2 penalty".lower() in [s.lower() for s in report.strengths]
    assert "early stopping" in [w.lower() for w in report.weaknesses]
    assert "early stopping" in report.missed_concepts
    assert report.evidence and "Q1:" in report.evidence[0]
    assert report.duration_seconds == 720


def test_recommended_topics_prefer_weak_concepts_then_untouched_topics() -> None:
    report = build_report(state_with_turns(evaluation()), MODE, NOW)

    assert report.recommended_topics[0] == "early stopping"
    assert any(topic in report.recommended_topics for topic in MODE.topics[1:])


def test_report_with_no_scored_turns_does_not_invent_scores() -> None:
    report = build_report(state_with_turns(evaluation(), degraded={1}), MODE, NOW)

    assert report.dimension_scores == {}
    assert report.overall_score == 0.0
