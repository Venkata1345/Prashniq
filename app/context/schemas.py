"""Candidate context: what we know about this candidate before the interview.

Two ingestion paths produce these structures -- a resume and a job description --
and a deterministic blueprint turns them into the topics the interview will
actually cover.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClaimCategory = Literal["project", "experience", "skill", "education", "other"]
Importance = Literal["must_have", "nice_to_have"]

# Deterministic priority weights. Interviewing a must-have skill the candidate
# also claims is the highest-value question we can ask: it is both relevant to
# the role and verifiable against their own words.
IMPORTANCE_WEIGHT: dict[str, float] = {"must_have": 2.0, "nice_to_have": 1.0}
CLAIMED_SKILL_BONUS = 0.5
# A topic the candidate's skill profile says is weak: practise what's weak.
WEAK_SKILL_BONUS = 0.5
RESUME_ONLY_WEIGHT = 0.75
MODE_FALLBACK_WEIGHT = 0.25


class ResumeClaim(BaseModel):
    """One verifiable assertion the candidate makes about themselves.

    Claims -- not raw resume text -- are what the interviewer is allowed to
    press on, which keeps follow-ups grounded in something the candidate wrote.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    skills: list[str] = Field(default_factory=list)
    category: ClaimCategory = "other"


class ResumeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ResumeClaim] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    seniority_signal: str | None = None

    def skills(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for claim in self.claims:
            for skill in claim.skills:
                key = skill.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    ordered.append(skill.strip())
        return ordered

    def merge(self, other: "ResumeProfile") -> "ResumeProfile":
        """Chunks are extracted independently, then merged here -- deterministic
        set union rather than a second LLM pass."""
        claims = list(self.claims)
        seen = {claim.text.strip().lower() for claim in claims}
        for claim in other.claims:
            key = claim.text.strip().lower()
            if key not in seen:
                seen.add(key)
                claims.append(claim)
        return ResumeProfile(
            claims=claims,
            focus_areas=_dedupe(self.focus_areas + other.focus_areas),
            seniority_signal=self.seniority_signal or other.seniority_signal,
        )


class SkillRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(min_length=1)
    importance: Importance = "nice_to_have"
    evidence: str = ""


class JobProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role_title: str | None = None
    requirements: list[SkillRequirement] = Field(default_factory=list)

    def must_haves(self) -> list[SkillRequirement]:
        return [r for r in self.requirements if r.importance == "must_have"]


class BlueprintTopic(BaseModel):
    """A planned topic with its provenance. `evidence` is what the interviewer
    is grounded in; `priority` is why it is in this position."""

    model_config = ConfigDict(frozen=True)

    key: str
    priority: float
    source: Literal["job_description", "resume", "mode"]
    target_skill: str | None = None
    evidence: tuple[str, ...] = ()
    rationale: str = ""


class InterviewBlueprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    interview_type: str
    topics: tuple[BlueprintTopic, ...] = ()

    def topic_keys(self) -> list[str]:
        return [topic.key for topic in self.topics]

    def find(self, key: str | None) -> BlueprintTopic | None:
        if not key:
            return None
        wanted = key.strip().lower()
        for topic in self.topics:
            if topic.key.strip().lower() == wanted:
                return topic
        return None

    def target_skills(self) -> list[str]:
        return _dedupe([t.target_skill for t in self.topics if t.target_skill])


class CandidateContext(BaseModel):
    """Everything ingested for one candidate. Stored separately from the
    interview so several interviews can reuse it."""

    context_id: str
    candidate_id: str | None = None
    resume: ResumeProfile | None = None
    job: JobProfile | None = None
    created_at: datetime

    @property
    def is_empty(self) -> bool:
        return self.resume is None and self.job is None


def _dedupe(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result
