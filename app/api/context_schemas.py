"""DTOs for candidate context.

The response deliberately returns the *extracted structure*, never the raw
document text that was submitted.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.context.schemas import (
    BlueprintTopic,
    CandidateContext,
    InterviewBlueprint,
    ResumeClaim,
    SkillRequirement,
)


class CreateContextRequest(BaseModel):
    candidate_id: str | None = None
    resume_text: str | None = Field(default=None, max_length=100_000)
    job_description_text: str | None = Field(default=None, max_length=50_000)


class ContextResponse(BaseModel):
    context_id: str
    candidate_id: str | None
    created_at: datetime
    role_title: str | None
    claims: list[ResumeClaim]
    focus_areas: list[str]
    requirements: list[SkillRequirement]

    @classmethod
    def from_context(cls, context: CandidateContext) -> "ContextResponse":
        return cls(
            context_id=context.context_id,
            candidate_id=context.candidate_id,
            created_at=context.created_at,
            role_title=context.job.role_title if context.job else None,
            claims=context.resume.claims if context.resume else [],
            focus_areas=context.resume.focus_areas if context.resume else [],
            requirements=context.job.requirements if context.job else [],
        )


class BlueprintResponse(BaseModel):
    """What the interview intends to cover, and why. Exposed so the plan is
    inspectable before and during the interview rather than implicit."""

    interview_type: str
    topics: list[BlueprintTopic]

    @classmethod
    def from_blueprint(cls, blueprint: InterviewBlueprint) -> "BlueprintResponse":
        return cls(
            interview_type=blueprint.interview_type, topics=list(blueprint.topics)
        )
