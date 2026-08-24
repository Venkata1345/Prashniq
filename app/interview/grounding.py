"""Grounding: what the interviewer and evaluator are allowed to lean on.

This is the seam between retrieval and the interview domain. The orchestrator
asks for grounding; the interviewer and evaluator receive it as plain strings
and never learn that a vector store exists.

Everything here degrades to empty grounding rather than raising -- a retrieval
outage costs grounding, not the interview.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.context.blueprint import EvidenceLookup, LexicalEvidenceLookup
from app.context.retrieval import select_evidence
from app.context.schemas import CandidateContext
from app.interview.modes import InterviewMode
from app.interview.planner import PlannedAction
from app.interview.schemas import Question
from app.interview.state import InterviewState
from app.rag.retriever import Retriever, dedupe
from app.rag.schemas import GLOBAL_OWNER

logger = logging.getLogger(__name__)

RESUME_EVIDENCE_LIMIT = 2
KNOWLEDGE_LIMIT_FOR_QUESTION = 1
KNOWLEDGE_LIMIT_FOR_EVALUATION = 2


@dataclass(frozen=True)
class Grounding:
    """Retrieved context for one turn.

    `resume_evidence` is the candidate's own words; `knowledge_notes` are
    reference points for the topic, used to judge completeness -- never to be
    read out to the candidate.
    """

    resume_evidence: list[str] = field(default_factory=list)
    role_requirements: list[str] = field(default_factory=list)
    knowledge_notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.resume_evidence or self.role_requirements or self.knowledge_notes)


class _HybridEvidenceLookup:
    def __init__(self, primary: EvidenceLookup, fallback: EvidenceLookup) -> None:
        self._primary = primary
        self._fallback = fallback

    async def claims_for(self, skill: str) -> list[str]:
        claims = await self._primary.claims_for(skill)
        return claims if claims else await self._fallback.claims_for(skill)


class GroundingService:
    """Builds grounding from the blueprint first, retrieval second.

    Blueprint evidence was already selected when the interview was planned, so
    it is preferred; retrieval fills the gaps and covers follow-up topics the
    blueprint never anticipated.
    """

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever

    def evidence_lookup(self, context: CandidateContext | None) -> EvidenceLookup:
        """How the blueprint should find claims backing a required skill.

        Hybrid: semantic search first, lexical overlap as the fallback --
        embeddings bridge vocabulary gaps, exact terms catch what a small
        embedding model scores too low, and an unindexed context (indexing is
        best-effort) still yields evidence instead of silently losing it.
        """
        lexical = LexicalEvidenceLookup(context.resume if context else None)
        if self._retriever is None or context is None:
            return lexical

        from app.rag.lookup import VectorEvidenceLookup

        return _HybridEvidenceLookup(
            VectorEvidenceLookup(self._retriever, owner_id=context.context_id), lexical
        )

    async def for_question(
        self, *, plan: PlannedAction, state: InterviewState, mode: InterviewMode
    ) -> Grounding:
        topic = plan.topic or state.current_topic or mode.display_name
        return Grounding(
            resume_evidence=await self._resume_evidence(topic, state),
            role_requirements=self._role_requirements(topic, state),
            knowledge_notes=await self._knowledge(topic, KNOWLEDGE_LIMIT_FOR_QUESTION),
        )

    async def for_evaluation(
        self, *, question: Question, state: InterviewState, mode: InterviewMode
    ) -> Grounding:
        return Grounding(
            resume_evidence=await self._resume_evidence(question.topic, state),
            role_requirements=self._role_requirements(question.topic, state),
            knowledge_notes=await self._knowledge(
                question.topic, KNOWLEDGE_LIMIT_FOR_EVALUATION
            ),
        )

    async def _resume_evidence(self, topic: str | None, state: InterviewState) -> list[str]:
        planned = state.topic_evidence(topic)
        if planned:
            return planned[:RESUME_EVIDENCE_LIMIT]
        if self._retriever is not None and state.context_id is not None:
            hits = dedupe(
                await self._retriever.texts(
                    topic or "",
                    collection="resume",
                    owner_id=state.context_id,
                    limit=RESUME_EVIDENCE_LIMIT,
                )
            )
            if hits:
                return hits
        # No retrieval configured (or nothing indexed): fall back to lexical
        # overlap over the claims carried on the state.
        return select_evidence(topic or "", state.resume, limit=RESUME_EVIDENCE_LIMIT)

    def _role_requirements(self, topic: str | None, state: InterviewState) -> list[str]:
        """The requirement text behind a planned topic, straight from the
        blueprint -- no retrieval needed for a handful of requirements."""
        planned = state.blueprint.find(topic) if state.blueprint else None
        if planned is None or planned.source != "job_description":
            return []
        return [planned.rationale] if planned.rationale else []

    async def _knowledge(self, topic: str | None, limit: int) -> list[str]:
        if self._retriever is None or not (topic or "").strip():
            return []
        return dedupe(
            await self._retriever.texts(
                topic or "", collection="knowledge", owner_id=GLOBAL_OWNER, limit=limit
            )
        )
