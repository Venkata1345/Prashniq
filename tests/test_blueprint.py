"""Blueprint construction: what the interview will cover, and in what order.

Entirely deterministic -- no LLM appears in this file.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.context.blueprint import BlueprintError, build_blueprint
from app.context.schemas import (
    CandidateContext,
    JobProfile,
    ResumeClaim,
    ResumeProfile,
    SkillRequirement,
)
from app.interview.modes import get_mode

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
FUNDAMENTALS = get_mode("ml_fundamentals")
JD_TARGETED = get_mode("jd_targeted")
RESUME_MODE = get_mode("resume_deep_dive")


def context(
    *,
    claims: list[tuple[str, list[str]]] | None = None,
    requirements: list[tuple[str, str]] | None = None,
) -> CandidateContext:
    resume = (
        ResumeProfile(
            claims=[
                ResumeClaim(text=text, skills=skills, category="project")
                for text, skills in claims
            ]
        )
        if claims is not None
        else None
    )
    job = (
        JobProfile(
            role_title="AI Engineer",
            requirements=[
                SkillRequirement(skill=skill, importance=importance, evidence="posting")
                for skill, importance in requirements
            ],
        )
        if requirements is not None
        else None
    )
    return CandidateContext(context_id="ctx-1", resume=resume, job=job, created_at=NOW)


async def test_without_context_the_blueprint_is_just_the_modes_topics() -> None:
    blueprint = await build_blueprint(FUNDAMENTALS)

    assert blueprint.topic_keys() == list(FUNDAMENTALS.topics)
    assert {topic.source for topic in blueprint.topics} == {"mode"}
    assert blueprint.target_skills() == []


async def test_must_haves_outrank_nice_to_haves_and_mode_defaults() -> None:
    blueprint = await build_blueprint(
        FUNDAMENTALS,
        context(requirements=[("Kubernetes", "nice_to_have"), ("RAG", "must_have")]),
    )

    assert blueprint.topic_keys()[:2] == ["RAG", "Kubernetes"]
    assert blueprint.topics[0].source == "job_description"
    assert blueprint.topics[-1].source == "mode"


async def test_a_required_skill_the_candidate_claims_ranks_highest() -> None:
    blueprint = await build_blueprint(
        FUNDAMENTALS,
        context(
            claims=[("Built a RAG system using FAISS.", ["RAG", "FAISS"])],
            requirements=[("model serving", "must_have"), ("RAG", "must_have")],
        ),
    )
    top = blueprint.topics[0]

    assert top.key == "RAG"
    assert top.priority > blueprint.topics[1].priority
    assert "claimed on the resume" in top.rationale


async def test_topics_carry_the_resume_claims_that_ground_them() -> None:
    blueprint = await build_blueprint(
        FUNDAMENTALS,
        context(
            claims=[("Built a RAG system using FAISS.", ["RAG", "FAISS"])],
            requirements=[("RAG", "must_have")],
        ),
    )

    assert blueprint.find("RAG").evidence == ("Built a RAG system using FAISS.",)

    # A requirement the resume says nothing about is planned, but ungrounded.
    ungrounded = await build_blueprint(
        FUNDAMENTALS, context(claims=[], requirements=[("RAG", "must_have")])
    )
    assert ungrounded.find("RAG").evidence == ()


async def test_resume_only_skills_are_included_below_role_requirements() -> None:
    blueprint = await build_blueprint(
        FUNDAMENTALS,
        context(
            claims=[("Ran Kafka pipelines in production.", ["Kafka"])],
            requirements=[("RAG", "nice_to_have")],
        ),
    )
    kafka = blueprint.find("Kafka")

    assert kafka is not None
    assert kafka.source == "resume"
    assert kafka.priority < blueprint.find("RAG").priority


async def test_resume_only_topics_cannot_crowd_out_the_role() -> None:
    claims = [(f"Used tool {i} in production.", [f"tool{i}"]) for i in range(10)]
    blueprint = await build_blueprint(
        FUNDAMENTALS, context(claims=claims, requirements=[("RAG", "must_have")])
    )
    resume_topics = [t for t in blueprint.topics if t.source == "resume"]

    assert len(resume_topics) <= 3
    assert blueprint.topics[0].key == "RAG"


async def test_a_skill_in_both_documents_appears_once_at_its_highest_priority() -> None:
    blueprint = await build_blueprint(
        FUNDAMENTALS,
        context(
            claims=[("Built a RAG system.", ["RAG"])],
            requirements=[("RAG", "must_have")],
        ),
    )
    keys = [key.lower() for key in blueprint.topic_keys()]

    assert keys.count("rag") == 1
    assert blueprint.find("RAG").source == "job_description"


async def test_the_blueprint_never_plans_more_topics_than_the_question_budget() -> None:
    requirements = [(f"skill {i}", "must_have") for i in range(20)]
    blueprint = await build_blueprint(FUNDAMENTALS, context(requirements=requirements))

    assert len(blueprint.topics) == FUNDAMENTALS.max_questions


async def test_context_only_modes_require_a_context() -> None:
    with pytest.raises(BlueprintError):
        await build_blueprint(JD_TARGETED)

    with pytest.raises(BlueprintError):
        await build_blueprint(RESUME_MODE, context())


async def test_a_resume_deep_dive_is_built_entirely_from_claims() -> None:
    blueprint = await build_blueprint(
        RESUME_MODE,
        context(claims=[("Built a RAG system using FAISS.", ["RAG", "FAISS"])]),
    )

    assert blueprint.topic_keys() == ["RAG", "FAISS"]
    assert {topic.source for topic in blueprint.topics} == {"resume"}
    assert all(topic.evidence for topic in blueprint.topics)


async def test_target_skills_are_reported_for_role_targeted_interviews() -> None:
    blueprint = await build_blueprint(
        JD_TARGETED, context(requirements=[("RAG", "must_have"), ("PyTorch", "nice_to_have")])
    )

    assert blueprint.target_skills() == ["RAG", "PyTorch"]
