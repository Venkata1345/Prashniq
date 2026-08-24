"""Resume and job-description ingestion.

Both are the same shape: chunk the document, ask the LLM for validated
structure, merge deterministically. Neither ever sees interview state, and the
raw document text is never stored or logged -- only the extracted structure.
"""

from __future__ import annotations

import logging

from app.context.chunking import DEFAULT_MAX_CHARS, chunk_text
from app.context.schemas import JobProfile, ResumeProfile
from app.llm.gateway import LLMGateway
from app.llm.schemas import LLMCallContext, LLMError, LLMMessage

logger = logging.getLogger(__name__)

MAX_CHUNKS = 8

RESUME_SYSTEM_PROMPT = """You extract structured claims from a candidate's resume \
for an AI/ML interview system.

A claim is one concrete, checkable assertion the candidate makes about their own \
work -- a project, a system they built, a responsibility, a technology they used.

Rules:
- Copy claims close to the candidate's own wording. Do not embellish.
- Do not invent claims that are not in the text.
- `skills` lists the concrete technologies or techniques named in that claim.
- Skip contact details, addresses and other personal data entirely.
"""

JOB_SYSTEM_PROMPT = """You extract the skill requirements from a job description \
for an AI/ML interview system.

Rules:
- `must_have` is for requirements the posting treats as required; everything \
else is `nice_to_have`.
- Name skills concretely ("RAG evaluation", "PyTorch", "model serving latency"), \
not vaguely ("communication", "team player").
- `evidence` quotes the phrase from the posting the requirement came from.
- Do not invent requirements that are not in the text.
"""


class DocumentIngestionError(RuntimeError):
    """Raised when a document could not be turned into usable structure."""


class ResumeIngestor:
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        max_chunks: int = MAX_CHUNKS,
        max_chunk_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._gateway = gateway
        self._max_chunks = max_chunks
        self._max_chunk_chars = max_chunk_chars

    async def ingest(self, text: str, *, request_id: str | None = None) -> ResumeProfile:
        chunks = chunk_text(text, max_chars=self._max_chunk_chars)
        if not chunks:
            raise DocumentIngestionError("resume is empty")
        if len(chunks) > self._max_chunks:
            logger.warning(
                "resume_truncated chunks=%d kept=%d", len(chunks), self._max_chunks
            )
            chunks = chunks[: self._max_chunks]

        profile = ResumeProfile()
        for index, chunk in enumerate(chunks, start=1):
            extracted = await self._extract_chunk(chunk, index, len(chunks), request_id)
            profile = profile.merge(extracted)

        if not profile.claims:
            raise DocumentIngestionError("no claims could be extracted from the resume")

        # Log volume, never content.
        logger.info(
            "resume_ingested chunks=%d claims=%d skills=%d",
            len(chunks),
            len(profile.claims),
            len(profile.skills()),
        )
        return profile

    async def _extract_chunk(
        self, chunk: str, index: int, total: int, request_id: str | None
    ) -> ResumeProfile:
        try:
            return await self._gateway.generate_structured(
                ResumeProfile,
                [
                    LLMMessage(
                        role="user",
                        content=f"Resume section {index} of {total}:\n\n{chunk}",
                    )
                ],
                system=RESUME_SYSTEM_PROMPT,
                context=LLMCallContext(request_id=request_id, purpose="ingest_resume"),
            )
        except LLMError as exc:
            # One unusable section should not lose the rest of the resume.
            logger.error("resume_chunk_failed index=%d error=%s", index, exc)
            return ResumeProfile()


class JobDescriptionIngestor:
    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def ingest(self, text: str, *, request_id: str | None = None) -> JobProfile:
        chunks = chunk_text(text)
        if not chunks:
            raise DocumentIngestionError("job description is empty")

        # Job descriptions are short; requirements must be weighed against each
        # other, so this is one call over the whole document.
        try:
            profile = await self._gateway.generate_structured(
                JobProfile,
                [LLMMessage(role="user", content=f"Job description:\n\n{text.strip()}")],
                system=JOB_SYSTEM_PROMPT,
                context=LLMCallContext(request_id=request_id, purpose="ingest_job_description"),
            )
        except LLMError as exc:
            raise DocumentIngestionError(
                "the job description could not be analysed"
            ) from exc

        if not profile.requirements:
            raise DocumentIngestionError(
                "no skill requirements could be extracted from the job description"
            )

        logger.info(
            "job_description_ingested requirements=%d must_have=%d",
            len(profile.requirements),
            len(profile.must_haves()),
        )
        return profile
