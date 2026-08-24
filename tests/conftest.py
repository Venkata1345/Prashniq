"""Deterministic test fixtures.

No test in this suite touches a real LLM provider: every gateway is a
`FakeLLMGateway`, the clock is fixed and ids are sequential.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.context.ingestion import JobDescriptionIngestor, ResumeIngestor
from app.context.repository import InMemoryCandidateContextRepository
from app.context.service import CandidateContextService
from app.interview.evaluator import Evaluator
from app.interview.grounding import GroundingService
from app.profile.service import SkillProfileService
from app.interview.interviewer import Interviewer
from app.interview.orchestrator import InterviewOrchestrator
from app.interview.repository import InMemoryInterviewRepository
from app.llm.fake import FakeLLMGateway, RecordedCall
from app.rag.embeddings import DeterministicEmbeddingGateway
from app.rag.indexer import Indexer
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


class FixedClock:
    """Advances a fixed amount per call so ordering is observable but stable."""

    def __init__(self, start: datetime = START, step_seconds: int = 30) -> None:
        self._current = start
        self._step = timedelta(seconds=step_seconds)

    def now(self) -> datetime:
        value = self._current
        self._current += self._step
        return value

    def advance(self, seconds: int) -> None:
        self._current += timedelta(seconds=seconds)


def evaluation_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "correctness": 6.0,
        "depth": 5.0,
        "communication": 7.0,
        "dimension_scores": [
            {"name": "reasoning", "score": 6.0},
            {"name": "tradeoff_awareness", "score": 4.0},
        ],
        "concepts_covered": ["regularization"],
        "missing_concepts": ["weight decay vs L2"],
        "misconceptions": [],
        "recommended_action": "probe_deeper",
        "follow_up_topic": None,
        "rationale": "Correct but shallow.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def question_json(question: str = "Why does that tradeoff hold?", topic: str = "regularization") -> str:
    return json.dumps({"question": question, "topic": topic})


def resume_profile_json(
    claims: list[tuple[str, list[str]]] | None = None,
    focus_areas: list[str] | None = None,
) -> str:
    if claims is None:
        claims = [("Built a RAG system using FAISS and FastAPI.", ["RAG", "FAISS", "FastAPI"])]
    return json.dumps(
        {
            "claims": [
                {"text": text, "skills": skills, "category": "project"}
                for text, skills in claims
            ],
            "focus_areas": focus_areas or ["retrieval systems"],
            "seniority_signal": "mid-level",
        }
    )


def job_profile_json(
    role_title: str | None = "AI Engineer",
    requirements: list[tuple[str, str]] | None = None,
) -> str:
    if requirements is None:
        requirements = [("RAG", "must_have"), ("model serving", "nice_to_have")]
    return json.dumps(
        {
            "role_title": role_title,
            "requirements": [
                {"skill": skill, "importance": importance, "evidence": f"posting: {skill}"}
                for skill, importance in requirements
            ],
        }
    )


def scripted_gateway(
    *, evaluations: list[str] | None = None, questions: list[str] | None = None
) -> FakeLLMGateway:
    """Routes by call purpose so evaluator and interviewer scripts stay
    independent of call ordering."""
    evaluation_queue = list(evaluations or [])
    question_queue = list(questions or [])
    counter = itertools.count(1)

    def responder(call: RecordedCall) -> str:
        if call.purpose == "evaluate_answer":
            return evaluation_queue.pop(0) if evaluation_queue else evaluation_json()
        if question_queue:
            return question_queue.pop(0)
        return question_json(question=f"Generated question {next(counter)}?")

    return FakeLLMGateway(responder=responder)


def build_retrieval_stack(min_score: float | None = None) -> tuple[Indexer, Retriever]:
    """Deterministic embeddings over an in-process store: the retrieval
    pipeline under test, with no provider and no database."""
    store = InMemoryVectorStore()
    embeddings = DeterministicEmbeddingGateway()
    retriever = (
        Retriever(embeddings=embeddings, store=store, min_score=min_score)
        if min_score is not None
        else Retriever(embeddings=embeddings, store=store)
    )
    return Indexer(embeddings=embeddings, store=store), retriever


def build_context_service(
    gateway: FakeLLMGateway,
    clock: FixedClock | None = None,
    indexer: Indexer | None = None,
) -> CandidateContextService:
    ids = itertools.count(1)
    return CandidateContextService(
        resume_ingestor=ResumeIngestor(gateway),
        job_ingestor=JobDescriptionIngestor(gateway),
        repository=InMemoryCandidateContextRepository(),
        clock=clock or FixedClock(),
        id_factory=lambda: f"ctx-{next(ids)}",
        indexer=indexer,
    )


def build_orchestrator(
    gateway: FakeLLMGateway,
    clock: FixedClock | None = None,
    context_service: CandidateContextService | None = None,
    grounding: GroundingService | None = None,
    profile_service: SkillProfileService | None = None,
) -> InterviewOrchestrator:
    ids = itertools.count(1)
    return InterviewOrchestrator(
        evaluator=Evaluator(gateway),
        interviewer=Interviewer(gateway),
        repository=InMemoryInterviewRepository(),
        clock=clock or FixedClock(),
        id_factory=lambda: f"id-{next(ids)}",
        context_service=context_service,
        grounding=grounding,
        profile_service=profile_service,
    )


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def gateway() -> FakeLLMGateway:
    return scripted_gateway()


@pytest.fixture
def orchestrator(gateway: FakeLLMGateway, clock: FixedClock) -> InterviewOrchestrator:
    return build_orchestrator(gateway, clock)
