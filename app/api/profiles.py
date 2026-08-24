"""Candidate skill-profile routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.profile.schemas import SkillEntry
from app.profile.service import SkillProfileService

router = APIRouter(prefix="/candidates", tags=["skill profile"])


def get_profile_service(request: Request) -> SkillProfileService:
    return request.app.state.profile_service


class ProfileResponse(BaseModel):
    candidate_id: str
    generated_at: datetime
    # Coarse interview areas -- the roadmap's "Transformers 8.1" view.
    topics: list[SkillEntry]
    # Fine-grained concept verdicts from the evaluator.
    concepts: list[SkillEntry]
    # Weakest first: what the next interview will steer toward.
    recommended_focus: list[str]


@router.get("/{candidate_id}/profile", response_model=ProfileResponse)
async def get_profile(
    candidate_id: str,
    service: SkillProfileService = Depends(get_profile_service),
) -> ProfileResponse:
    """The candidate's decayed skill history across all completed interviews.

    An unknown candidate simply has an empty profile -- there is no separate
    registration step to 404 against.
    """
    profile = await service.get_profile(candidate_id)
    return ProfileResponse(
        candidate_id=profile.candidate_id,
        generated_at=profile.generated_at,
        topics=profile.topics(),
        concepts=[entry for entry in profile.skills if entry.kind == "concept"],
        recommended_focus=profile.weak_skills(),
    )
