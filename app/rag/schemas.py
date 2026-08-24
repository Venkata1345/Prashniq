"""Retrieval types.

One vector collection per kind of content, partitioned by owner so a
candidate's resume can never surface in someone else's interview.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Collection = Literal["resume", "job_description", "knowledge"]

# Knowledge is shared by every interview; candidate content is scoped to a
# candidate context id.
GLOBAL_OWNER = "global"


class DocumentChunk(BaseModel):
    """A unit of text to index. `chunk_id` is stable for a given source, so
    re-indexing updates in place rather than duplicating."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    collection: Collection
    owner_id: str = GLOBAL_OWNER
    text: str = Field(min_length=1)
    topic: str | None = None
    source: str | None = None
    # What gets embedded, when it should differ from what gets displayed.
    # Live calibration (2026-08-24, text-embedding-3-small) showed bare resume
    # claims score ~0.2 against the skills they demonstrate; embedding the
    # claim together with its skill tags closes that gap without the tags ever
    # appearing in a prompt. Transient: never persisted.
    embedding_text: str | None = None

    @property
    def text_for_embedding(self) -> str:
        return self.embedding_text or self.text


class EmbeddedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    embedding: tuple[float, ...]


class RetrievedChunk(BaseModel):
    """A search hit. `score` is cosine similarity in [-1, 1]; higher is closer."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


class EmbeddingError(RuntimeError):
    """Embedding provider failure. Retrieval degrades rather than failing the
    interview, so callers catch this."""


class VectorStoreError(RuntimeError):
    """Vector store failure, treated the same way."""
