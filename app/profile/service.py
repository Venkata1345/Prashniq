"""Skill-profile service: record what an interview showed, aggregate on read.

Recording is derived deterministically from the completed interview state:
  - one "topic" observation per interviewed topic (mean correctness/depth of
    its non-degraded turns) -- the coarse areas the roadmap displays
  - one "concept" observation per entry in `state.concept_scores` -- the
    evaluator's fine-grained verdicts

Anonymous interviews (no candidate_id) record nothing: there is no identity to
aggregate under.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from app.interview.schemas import InterviewStatus, Turn
from app.interview.state import InterviewState
from app.profile.repository import SkillObservationRepository
from app.profile.schemas import SkillObservation, SkillProfile, aggregate

logger = logging.getLogger(__name__)

# Correctness matters more than depth for "do they know this", but an answer
# with no depth should not score as mastery.
CORRECTNESS_WEIGHT = 0.7
DEPTH_WEIGHT = 0.3


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SkillProfileService:
    def __init__(
        self, repository: SkillObservationRepository, clock: Clock | None = None
    ) -> None:
        self._repository = repository
        self._clock = clock or SystemClock()

    async def record_interview(self, state: InterviewState) -> int:
        """Fold a completed interview into the candidate's history."""
        if state.status is not InterviewStatus.COMPLETED:
            raise ValueError("only completed interviews are recorded")
        if state.candidate_id is None:
            return 0

        observed_at = state.completed_at or self._clock.now()
        observations = [
            *self._topic_observations(state, observed_at),
            *self._concept_observations(state, observed_at),
        ]
        recorded = await self._repository.add(observations)
        logger.info(
            "skill_observations_recorded interview_id=%s candidate_id=%s "
            "derived=%d recorded=%d",
            state.interview_id,
            state.candidate_id,
            len(observations),
            recorded,
        )
        return recorded

    async def get_profile(self, candidate_id: str) -> SkillProfile:
        observations = await self._repository.for_candidate(candidate_id)
        return aggregate(candidate_id, observations, self._clock.now())

    async def weak_skills(self, candidate_id: str) -> list[str]:
        """Weakest first; what the blueprint should steer toward."""
        return (await self.get_profile(candidate_id)).weak_skills()

    def _topic_observations(
        self, state: InterviewState, observed_at: datetime
    ) -> list[SkillObservation]:
        by_topic: dict[str, list[Turn]] = {}
        for turn in state.turns:
            if turn.evaluation_degraded:
                continue  # an evaluation we could not trust is not evidence
            by_topic.setdefault(turn.question.topic, []).append(turn)

        return [
            SkillObservation(
                candidate_id=state.candidate_id,  # type: ignore[arg-type]
                concept=topic,
                kind="topic",
                score=round(
                    sum(_turn_score(turn) for turn in turns) / len(turns), 2
                ),
                interview_id=state.interview_id,
                interview_type=state.interview_type,
                observed_at=observed_at,
            )
            for topic, turns in by_topic.items()
        ]

    def _concept_observations(
        self, state: InterviewState, observed_at: datetime
    ) -> list[SkillObservation]:
        return [
            SkillObservation(
                candidate_id=state.candidate_id,  # type: ignore[arg-type]
                concept=concept,
                kind="concept",
                score=round(min(10.0, max(0.0, score)), 2),
                interview_id=state.interview_id,
                interview_type=state.interview_type,
                observed_at=observed_at,
            )
            for concept, score in state.concept_scores.items()
        ]


def _turn_score(turn: Turn) -> float:
    return (
        CORRECTNESS_WEIGHT * turn.evaluation.correctness
        + DEPTH_WEIGHT * turn.evaluation.depth
    )
