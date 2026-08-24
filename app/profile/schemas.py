"""Persistent candidate skill profile.

An observation is one interview's verdict on one skill. The profile is a
decayed aggregation over all of a candidate's observations, computed in code so
it is reproducible and explainable.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ObservationKind = Literal["topic", "concept"]

# Skill estimates go stale: an interview from three months ago says half as
# much about today's ability as one from this week.
HALF_LIFE_DAYS = 90.0
WEAK_SKILL_THRESHOLD = 5.0


class SkillObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    concept: str
    kind: ObservationKind
    score: float = Field(ge=0.0, le=10.0)
    interview_id: str
    interview_type: str
    observed_at: datetime


class SkillEntry(BaseModel):
    concept: str
    kind: ObservationKind
    score: float
    observations: int
    last_observed_at: datetime


class SkillProfile(BaseModel):
    candidate_id: str
    skills: list[SkillEntry] = Field(default_factory=list)
    generated_at: datetime

    def topics(self) -> list[SkillEntry]:
        """Coarse interview areas -- what the roadmap's `Transformers 8.1`
        display is made of."""
        return [entry for entry in self.skills if entry.kind == "topic"]

    def weak_skills(self, threshold: float = WEAK_SKILL_THRESHOLD) -> list[str]:
        """Weakest first: what the next interview should practise."""
        weak = [entry for entry in self.skills if entry.score < threshold]
        weak.sort(key=lambda entry: entry.score)
        return [entry.concept for entry in weak]


def decay_weight(observed_at: datetime, now: datetime) -> float:
    """Exponential decay with a 90-day half-life. The future decays nothing."""
    age_days = max(0.0, (now - observed_at).total_seconds() / 86_400)
    return math.pow(0.5, age_days / HALF_LIFE_DAYS)


def aggregate(
    candidate_id: str, observations: list[SkillObservation], now: datetime
) -> SkillProfile:
    """Weighted mean per (kind, concept), newest observations counting most."""
    grouped: dict[tuple[str, str], list[SkillObservation]] = {}
    for observation in observations:
        key = (observation.kind, observation.concept.strip().lower())
        grouped.setdefault(key, []).append(observation)

    skills: list[SkillEntry] = []
    for group in grouped.values():
        weights = [decay_weight(item.observed_at, now) for item in group]
        total = sum(weights)
        if total <= 0.0:
            continue
        score = sum(item.score * weight for item, weight in zip(group, weights)) / total
        newest = max(group, key=lambda item: item.observed_at)
        skills.append(
            SkillEntry(
                concept=newest.concept,
                kind=newest.kind,
                score=round(score, 2),
                observations=len(group),
                last_observed_at=newest.observed_at,
            )
        )

    skills.sort(key=lambda entry: (-entry.score, entry.concept.lower()))
    return SkillProfile(candidate_id=candidate_id, skills=skills, generated_at=now)
