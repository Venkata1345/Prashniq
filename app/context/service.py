"""Candidate-context service: ingest documents, store the structure.

The raw resume and job-description text is used to build the profile and then
dropped -- only extracted claims and requirements are retained.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Protocol

from app.context.ingestion import (
    DocumentIngestionError,
    JobDescriptionIngestor,
    ResumeIngestor,
)
from app.context.repository import CandidateContextRepository
from app.context.schemas import CandidateContext
from app.context.schemas import ResumeClaim
from app.rag.indexer import Indexer, stable_chunk_id
from app.rag.schemas import DocumentChunk, EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class CandidateContextService:
    def __init__(
        self,
        *,
        resume_ingestor: ResumeIngestor,
        job_ingestor: JobDescriptionIngestor,
        repository: CandidateContextRepository,
        clock: Clock | None = None,
        id_factory: Callable[[], str] | None = None,
        indexer: Indexer | None = None,
    ) -> None:
        self._resume_ingestor = resume_ingestor
        self._job_ingestor = job_ingestor
        self._repository = repository
        self._clock = clock or SystemClock()
        self._new_id = id_factory or (lambda: uuid.uuid4().hex)
        self._indexer = indexer

    async def create(
        self,
        *,
        resume_text: str | None = None,
        job_description_text: str | None = None,
        candidate_id: str | None = None,
        request_id: str | None = None,
    ) -> CandidateContext:
        if not (resume_text or "").strip() and not (job_description_text or "").strip():
            raise DocumentIngestionError(
                "provide a resume, a job description, or both"
            )

        resume = (
            await self._resume_ingestor.ingest(resume_text, request_id=request_id)
            if (resume_text or "").strip()
            else None
        )
        job = (
            await self._job_ingestor.ingest(job_description_text, request_id=request_id)
            if (job_description_text or "").strip()
            else None
        )

        context = CandidateContext(
            context_id=self._new_id(),
            candidate_id=candidate_id,
            resume=resume,
            job=job,
            created_at=self._clock.now(),
        )
        await self._repository.add(context)
        await self._index(context)
        logger.info(
            "candidate_context_created context_id=%s has_resume=%s has_job=%s",
            context.context_id,
            resume is not None,
            job is not None,
        )
        return context

    async def get(self, context_id: str) -> CandidateContext:
        return await self._repository.get(context_id)

    async def _index(self, context: CandidateContext) -> None:
        """Index claims and requirements for retrieval.

        Best-effort: a context that cannot be indexed is still usable, it just
        falls back to lexical grounding.
        """
        if self._indexer is None:
            return
        try:
            if context.resume:
                await self._indexer.index_chunks(
                    [
                        DocumentChunk(
                            chunk_id=stable_chunk_id(claim.text),
                            collection="resume",
                            owner_id=context.context_id,
                            text=claim.text,
                            source="resume",
                            embedding_text=claim_embedding_text(claim),
                        )
                        for claim in context.resume.claims
                    ]
                )
            if context.job:
                await self._indexer.index_texts(
                    [
                        f"{requirement.skill}: {requirement.evidence}".strip(": ")
                        for requirement in context.job.requirements
                    ],
                    collection="job_description",
                    owner_id=context.context_id,
                    source="job description",
                )
        except (EmbeddingError, VectorStoreError) as exc:
            logger.error(
                "context_indexing_failed context_id=%s error=%s", context.context_id, exc
            )


def claim_embedding_text(claim: ResumeClaim) -> str:
    """Embed the claim together with its extracted skill tags.

    A claim like "Built a RAG pipeline using FAISS" never says "vector search",
    so a JD skill phrased that way misses it. The tags bridge the vocabulary
    gap; the displayed evidence stays the candidate's own words.
    """
    if not claim.skills:
        return claim.text
    return f"{claim.text} Skills demonstrated: {', '.join(claim.skills)}."
