"""Report generation. Pure computation over recorded turns — no LLM call, so
scores are reproducible and explainable.
"""

from __future__ import annotations

from datetime import datetime

from app.interview.modes import InterviewMode
from app.interview.schemas import InterviewReport, Turn
from app.interview.state import InterviewState

MAX_LISTED = 8
WEAK_CONCEPT_THRESHOLD = 5.0


def build_report(
    state: InterviewState, mode: InterviewMode, now: datetime
) -> InterviewReport:
    scored_turns = [turn for turn in state.turns if not turn.evaluation_degraded]
    dimension_scores = _dimension_scores(scored_turns, mode)
    overall = (
        round(sum(dimension_scores.values()) / len(dimension_scores), 2)
        if dimension_scores
        else 0.0
    )

    strong = [c for c, s in state.concept_scores.items() if s >= WEAK_CONCEPT_THRESHOLD]
    weak = [c for c, s in state.concept_scores.items() if s < WEAK_CONCEPT_THRESHOLD]

    return InterviewReport(
        interview_id=state.interview_id,
        interview_type=state.interview_type,
        overall_score=overall,
        dimension_scores=dimension_scores,
        strengths=_top(strong, state.strengths),
        weaknesses=_top(weak, state.weaknesses),
        evidence=_evidence(scored_turns),
        missed_concepts=_top(
            [c for turn in state.turns for c in turn.evaluation.missing_concepts], []
        ),
        recommended_topics=_recommended_topics(state, mode),
        unaddressed_target_skills=_unaddressed_target_skills(state),
        questions_answered=len(state.turns),
        duration_seconds=state.elapsed_seconds(now),
    )


def _dimension_scores(turns: list[Turn], mode: InterviewMode) -> dict[str, float]:
    """Mean per dimension across turns. Dimensions the evaluator never returned
    are omitted rather than defaulted, so the report never invents a score."""
    totals: dict[str, list[float]] = {}
    for turn in turns:
        for name, score in turn.evaluation.scores_by_dimension().items():
            if name in mode.dimensions:
                totals.setdefault(name, []).append(score)
    return {
        name: round(sum(values) / len(values), 2)
        for name, values in totals.items()
        if values
    }


def _evidence(turns: list[Turn]) -> list[str]:
    return [
        f"Q{turn.question.index}: {turn.question.text} — "
        f"correctness {turn.evaluation.correctness:g}/10, "
        f"depth {turn.evaluation.depth:g}/10. {turn.evaluation.rationale}".strip()
        for turn in turns
    ]


def _recommended_topics(state: InterviewState, mode: InterviewMode) -> list[str]:
    weak_concepts = sorted(
        (c for c, s in state.concept_scores.items() if s < WEAK_CONCEPT_THRESHOLD),
        key=lambda c: state.concept_scores[c],
    )
    untouched = [t for t in mode.topics if t not in state.topics_covered]
    return _top(weak_concepts, untouched)


def _unaddressed_target_skills(state: InterviewState) -> list[str]:
    """Skills the blueprint planned to cover but the interview never reached --
    the honest limit on what this report can say about role fit."""
    if state.blueprint is None:
        return []
    asked = {question.topic.strip().lower() for question in state.questions_asked}
    return [
        topic.target_skill
        for topic in state.blueprint.topics
        if topic.target_skill and topic.key.strip().lower() not in asked
    ]


def _top(primary: list[str], secondary: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in [*primary, *secondary]:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
        if len(result) == MAX_LISTED:
            break
    return result
