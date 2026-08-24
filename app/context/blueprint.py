"""Interview blueprint construction.

Pure code. Given a mode, a resume and a job description, this decides *what the
interview will cover and in what order* before a single question is generated.

Priority model:
  job requirement          importance weight (must_have 2.0 / nice_to_have 1.0)
  + candidate claims it    +0.5   (relevant AND verifiable against their words)
  + profile says weak      +0.5   (practise what's weak)
  resume-only strength     0.75   (worth probing, but not what the role needs)
  mode default topic       0.25   (filler so an interview is never empty)
"""

from __future__ import annotations

from typing import Protocol, Sequence

from app.context.retrieval import select_claims, select_evidence, tokenize
from app.context.schemas import (
    CLAIMED_SKILL_BONUS,
    IMPORTANCE_WEIGHT,
    MODE_FALLBACK_WEIGHT,
    RESUME_ONLY_WEIGHT,
    WEAK_SKILL_BONUS,
    BlueprintTopic,
    CandidateContext,
    InterviewBlueprint,
    ResumeProfile,
)
from app.interview.modes import InterviewMode

EVIDENCE_PER_TOPIC = 2
RESUME_ONLY_TOPIC_LIMIT = 3


class BlueprintError(RuntimeError):
    """Raised when a mode needs candidate context that was not supplied."""


class EvidenceLookup(Protocol):
    """Finds the resume claims that back a required skill.

    Async because the production implementation is a vector search; the lexical
    implementation below is the fallback when retrieval is unavailable.
    """

    async def claims_for(self, skill: str) -> list[str]: ...


class LexicalEvidenceLookup:
    def __init__(self, resume: ResumeProfile | None) -> None:
        self._resume = resume

    async def claims_for(self, skill: str) -> list[str]:
        return select_evidence(skill, self._resume, limit=EVIDENCE_PER_TOPIC)


async def build_blueprint(
    mode: InterviewMode,
    context: CandidateContext | None = None,
    lookup: EvidenceLookup | None = None,
    focus_skills: Sequence[str] = (),
) -> InterviewBlueprint:
    if mode.requires_candidate_context and (context is None or context.is_empty):
        raise BlueprintError(
            f"interview type '{mode.key}' requires a candidate context "
            "(resume and/or job description)"
        )

    resume = context.resume if context else None
    lookup = lookup or LexicalEvidenceLookup(resume)
    topics: list[BlueprintTopic] = []

    if context and context.job:
        topics.extend(await _from_job_description(context, lookup))
    if resume:
        topics.extend(_from_resume_only(resume, topics))

    topics = _dedupe(topics)
    topics.extend(_from_mode(mode, topics))
    topics = _apply_focus(topics, focus_skills)

    # Stable: equal priorities keep the order they were generated in, which is
    # job requirements, then resume strengths, then mode defaults.
    topics.sort(key=lambda topic: -topic.priority)
    return InterviewBlueprint(
        interview_type=mode.key, topics=tuple(topics[: mode.max_questions])
    )


async def _from_job_description(
    context: CandidateContext, lookup: EvidenceLookup
) -> list[BlueprintTopic]:
    """Requirements first. Whether the candidate actually claims a required
    skill is a retrieval question -- "RAG" should match "built a retrieval
    pipeline" even with no shared tokens."""
    topics: list[BlueprintTopic] = []
    assert context.job is not None

    for requirement in context.job.requirements:
        priority = IMPORTANCE_WEIGHT.get(requirement.importance, 1.0)
        claims = await lookup.claims_for(requirement.skill)
        if claims:
            priority += CLAIMED_SKILL_BONUS
            rationale = (
                f"{requirement.importance.replace('_', ' ')} for the role and claimed "
                "on the resume"
            )
        else:
            rationale = f"{requirement.importance.replace('_', ' ')} for the role"

        topics.append(
            BlueprintTopic(
                key=requirement.skill,
                priority=priority,
                source="job_description",
                target_skill=requirement.skill,
                evidence=tuple(claims),
                rationale=rationale,
            )
        )
    return topics


def _from_resume_only(
    resume: ResumeProfile, existing: list[BlueprintTopic]
) -> list[BlueprintTopic]:
    """Strong resume claims the job description never mentioned. Capped, so a
    long resume cannot crowd out what the role actually requires."""
    taken = {topic.key.strip().lower() for topic in existing}
    topics: list[BlueprintTopic] = []

    for skill in resume.skills():
        if skill.strip().lower() in taken:
            continue
        claims = select_claims(skill, resume.claims, limit=EVIDENCE_PER_TOPIC)
        if not claims:
            continue
        topics.append(
            BlueprintTopic(
                key=skill,
                priority=RESUME_ONLY_WEIGHT,
                source="resume",
                target_skill=skill,
                evidence=tuple(claim.text for claim in claims),
                rationale="claimed on the resume",
            )
        )
        if len(topics) == RESUME_ONLY_TOPIC_LIMIT:
            break
    return topics


def _from_mode(mode: InterviewMode, existing: list[BlueprintTopic]) -> list[BlueprintTopic]:
    taken = {topic.key.strip().lower() for topic in existing}
    return [
        BlueprintTopic(
            key=topic,
            priority=MODE_FALLBACK_WEIGHT,
            source="mode",
            rationale=f"standard {mode.display_name} topic",
        )
        for topic in mode.topics
        if topic.strip().lower() not in taken
    ]


def _apply_focus(
    topics: list[BlueprintTopic], focus_skills: Sequence[str]
) -> list[BlueprintTopic]:
    """Boost topics the candidate's skill profile marked weak."""
    if not focus_skills:
        return topics
    focus_tokens = [(skill, tokenize(skill)) for skill in focus_skills]

    boosted: list[BlueprintTopic] = []
    for topic in topics:
        topic_tokens = tokenize(topic.key)
        match = next(
            (skill for skill, tokens in focus_tokens if tokens and tokens & topic_tokens),
            None,
        )
        if match is None:
            boosted.append(topic)
            continue
        boosted.append(
            topic.model_copy(
                update={
                    "priority": topic.priority + WEAK_SKILL_BONUS,
                    "rationale": f"{topic.rationale}; weak in profile ({match})".strip("; "),
                }
            )
        )
    return boosted


def _dedupe(topics: list[BlueprintTopic]) -> list[BlueprintTopic]:
    """Keep the highest-priority instance of each topic."""
    best: dict[str, BlueprintTopic] = {}
    for topic in topics:
        key = topic.key.strip().lower()
        if key not in best or topic.priority > best[key].priority:
            best[key] = topic
    return list(best.values())
