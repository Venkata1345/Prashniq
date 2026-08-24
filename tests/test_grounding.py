"""Grounding: what reaches the interviewer and the evaluator, and what does not."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.context.blueprint import LexicalEvidenceLookup, build_blueprint
from app.context.schemas import (
    CandidateContext,
    JobProfile,
    ResumeClaim,
    ResumeProfile,
    SkillRequirement,
)
from app.interview.grounding import GroundingService
from app.interview.modes import get_mode
from app.interview.planner import PlannedAction
from app.interview.schemas import Question
from app.interview.state import InterviewState
from app.rag.embeddings import DeterministicEmbeddingGateway
from app.rag.indexer import Indexer
from app.rag.knowledge import seed_knowledge_base
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
MODE = get_mode("jd_targeted")
FUNDAMENTALS = get_mode("ml_fundamentals")

RAG_CLAIM = "Built a RAG system using FAISS and FastAPI."
KAFKA_CLAIM = "Ran Kafka ingestion pipelines feeding the feature store."


def candidate_context() -> CandidateContext:
    return CandidateContext(
        context_id="ctx-1",
        resume=ResumeProfile(
            claims=[
                ResumeClaim(text=RAG_CLAIM, skills=["RAG", "FAISS"], category="project"),
                ResumeClaim(text=KAFKA_CLAIM, skills=["Kafka"], category="project"),
            ]
        ),
        job=JobProfile(
            role_title="AI Engineer",
            requirements=[
                SkillRequirement(skill="RAG", importance="must_have", evidence="posting")
            ],
        ),
        created_at=NOW,
    )


async def retrieval_stack(*, seed: bool = True, index_context: bool = True):
    store = InMemoryVectorStore()
    embeddings = DeterministicEmbeddingGateway()
    indexer = Indexer(embeddings=embeddings, store=store)
    retriever = Retriever(embeddings=embeddings, store=store)  # production threshold

    if seed:
        await seed_knowledge_base(indexer)
    if index_context:
        await indexer.index_texts(
            [RAG_CLAIM, KAFKA_CLAIM], collection="resume", owner_id="ctx-1"
        )
    return indexer, retriever


async def interview_state(
    *, service: GroundingService, context: CandidateContext | None, mode=MODE
) -> InterviewState:
    blueprint = await build_blueprint(mode, context, service.evidence_lookup(context))
    return InterviewState(
        interview_id="interview-1",
        interview_type=mode.key,
        context_id=context.context_id if context else None,
        blueprint=blueprint,
        resume=context.resume if context else None,
        remaining_topics=blueprint.topic_keys(),
        current_topic="RAG",
        created_at=NOW,
    )


def plan(topic: str, action: str = "probe_deeper") -> PlannedAction:
    return PlannedAction(
        action=action, topic=topic, difficulty=3, follow_up_depth=1, reason="test"
    )


def question(topic: str) -> Question:
    return Question(
        id="q1",
        index=1,
        text=f"A question about {topic}?",
        topic=topic,
        difficulty=3,
        action="open_question",
        asked_at=NOW,
    )


class TestQuestionGrounding:
    async def test_planned_topics_use_the_blueprints_own_evidence(self) -> None:
        _, retriever = await retrieval_stack()
        service = GroundingService(retriever)
        state = await interview_state(service=service, context=candidate_context())

        grounding = await service.for_question(plan=plan("RAG"), state=state, mode=MODE)

        assert grounding.resume_evidence == [RAG_CLAIM]
        assert grounding.role_requirements == ["must have for the role and claimed on the resume"]

    async def test_an_unplanned_follow_up_topic_falls_back_to_retrieval(self) -> None:
        _, retriever = await retrieval_stack()
        service = GroundingService(retriever)
        state = await interview_state(service=service, context=candidate_context())

        grounding = await service.for_question(
            plan=plan("Kafka ingestion throughput"), state=state, mode=MODE
        )

        assert KAFKA_CLAIM in grounding.resume_evidence
        assert grounding.role_requirements == []

    async def test_knowledge_notes_are_attached_for_the_topic(self) -> None:
        _, retriever = await retrieval_stack()
        service = GroundingService(retriever)
        state = await interview_state(
            service=service, context=candidate_context(), mode=FUNDAMENTALS
        )

        grounding = await service.for_question(
            plan=plan("bias-variance tradeoff"), state=state, mode=FUNDAMENTALS
        )

        assert grounding.knowledge_notes
        assert "bias-variance tradeoff" in grounding.knowledge_notes[0]

    async def test_an_unknown_topic_gets_no_invented_notes(self) -> None:
        _, retriever = await retrieval_stack()
        service = GroundingService(retriever)
        state = await interview_state(service=service, context=candidate_context())

        grounding = await service.for_question(
            plan=plan("underwater basket weaving"), state=state, mode=MODE
        )

        assert grounding.knowledge_notes == []
        assert grounding.resume_evidence == []
        assert grounding.is_empty


class TestEvaluationGrounding:
    async def test_the_evaluator_gets_reference_notes_and_the_candidates_claim(self) -> None:
        _, retriever = await retrieval_stack()
        service = GroundingService(retriever)
        state = await interview_state(service=service, context=candidate_context())

        grounding = await service.for_evaluation(
            question=question("RAG"), state=state, mode=MODE
        )

        assert grounding.resume_evidence == [RAG_CLAIM]
        assert len(grounding.knowledge_notes) <= 2


class TestWithoutRetrieval:
    async def test_grounding_still_works_from_the_blueprint_alone(self) -> None:
        service = GroundingService()
        state = await interview_state(service=service, context=candidate_context())

        grounding = await service.for_question(plan=plan("RAG"), state=state, mode=MODE)

        assert grounding.resume_evidence == [RAG_CLAIM]
        assert grounding.knowledge_notes == []

    async def test_unplanned_topics_fall_back_to_lexical_matching(self) -> None:
        service = GroundingService()
        state = await interview_state(service=service, context=candidate_context())

        grounding = await service.for_question(
            plan=plan("Kafka pipelines"), state=state, mode=MODE
        )

        assert grounding.resume_evidence == [KAFKA_CLAIM]

    async def test_an_interview_with_no_context_has_no_evidence(self) -> None:
        service = GroundingService()
        state = await interview_state(service=service, context=None, mode=FUNDAMENTALS)

        grounding = await service.for_question(
            plan=plan("regularization"), state=state, mode=FUNDAMENTALS
        )

        assert grounding.is_empty


class TestEvidenceLookupSelection:
    async def test_retrieval_is_used_when_available(self) -> None:
        _, retriever = await retrieval_stack()
        lookup = GroundingService(retriever).evidence_lookup(candidate_context())

        assert await lookup.claims_for("FAISS index") == [RAG_CLAIM]

    async def test_lexical_matching_is_used_without_retrieval(self) -> None:
        lookup = GroundingService().evidence_lookup(candidate_context())

        assert isinstance(lookup, LexicalEvidenceLookup)
        assert await lookup.claims_for("FAISS") == [RAG_CLAIM]

    async def test_an_unindexed_context_falls_back_to_lexical_matching(self) -> None:
        # Indexing is best-effort; an indexing outage must cost semantic
        # matching, not all evidence.
        _, retriever = await retrieval_stack(index_context=False)
        lookup = GroundingService(retriever).evidence_lookup(candidate_context())

        assert await lookup.claims_for("FAISS") == [RAG_CLAIM]

    async def test_a_skill_no_claim_supports_still_yields_nothing(self) -> None:
        # The fallback must not weaken the no-invented-evidence property.
        _, retriever = await retrieval_stack()
        lookup = GroundingService(retriever).evidence_lookup(candidate_context())

        assert await lookup.claims_for("Kubernetes autoscaling") == []

    async def test_the_blueprint_bonus_follows_retrieved_evidence(self) -> None:
        _, retriever = await retrieval_stack()
        service = GroundingService(retriever)
        context = candidate_context()

        blueprint = await build_blueprint(MODE, context, service.evidence_lookup(context))
        rag_topic = blueprint.find("RAG")

        assert rag_topic.evidence == (RAG_CLAIM,)
        # Required *and* claimed: 2.0 + 0.5.
        assert rag_topic.priority == pytest.approx(2.5)
