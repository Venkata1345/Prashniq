"""Indexing: text -> chunks -> embeddings -> store."""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from app.context.chunking import chunk_text
from app.rag.embeddings import EmbeddingGateway
from app.rag.schemas import (
    GLOBAL_OWNER,
    Collection,
    DocumentChunk,
    EmbeddedChunk,
)
from app.rag.store import VectorStore

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, *, embeddings: EmbeddingGateway, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    async def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        if not chunks:
            return 0

        vectors = await self._embeddings.embed_documents(
            [chunk.text_for_embedding for chunk in chunks]
        )
        written = await self._store.upsert(
            [
                EmbeddedChunk(chunk=chunk, embedding=vector)
                for chunk, vector in zip(chunks, vectors)
            ]
        )
        # Volume only: indexed content can be a candidate's resume.
        logger.info(
            "indexed collection=%s owner=%s chunks=%d",
            chunks[0].collection,
            chunks[0].owner_id,
            written,
        )
        return written

    async def index_texts(
        self,
        texts: Sequence[str],
        *,
        collection: Collection,
        owner_id: str = GLOBAL_OWNER,
        topic: str | None = None,
        source: str | None = None,
    ) -> int:
        """Index short, already-atomic texts (a resume claim, a requirement)."""
        return await self.index_chunks(
            [
                DocumentChunk(
                    chunk_id=stable_chunk_id(text),
                    collection=collection,
                    owner_id=owner_id,
                    text=text,
                    topic=topic,
                    source=source,
                )
                for text in texts
                if text.strip()
            ]
        )

    async def index_document(
        self,
        text: str,
        *,
        collection: Collection,
        owner_id: str = GLOBAL_OWNER,
        topic: str | None = None,
        source: str | None = None,
        max_chars: int = 800,
    ) -> int:
        """Index a longer document, split on its own structure first."""
        return await self.index_texts(
            chunk_text(text, max_chars=max_chars),
            collection=collection,
            owner_id=owner_id,
            topic=topic,
            source=source,
        )


def stable_chunk_id(text: str) -> str:
    """Content-addressed, so re-indexing the same claim updates its row instead
    of duplicating it."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:32]
