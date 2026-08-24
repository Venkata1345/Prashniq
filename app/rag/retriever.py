"""Retrieval: query -> embedding -> nearest chunks.

Every method degrades to an empty result on provider or store failure. A
retrieval outage must never end an interview -- an ungrounded question is worse
than a grounded one, but far better than a 500.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from app.rag.embeddings import EmbeddingGateway
from app.rag.schemas import (
    GLOBAL_OWNER,
    Collection,
    EmbeddingError,
    RetrievedChunk,
    VectorStoreError,
)
from app.rag.store import VectorStore

logger = logging.getLogger(__name__)

# Cosine similarity below this is noise rather than evidence.
DEFAULT_MIN_SCORE = 0.15
DEFAULT_LIMIT = 3


class Retriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingGateway,
        store: VectorStore,
        min_score: float = DEFAULT_MIN_SCORE,
        per_collection: Mapping[str, float] | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._min_score = min_score
        # Different collections have different score bands: long knowledge
        # notes match high, short claims match low. One global threshold
        # cannot serve both (see the live calibration in the README).
        self._per_collection = dict(per_collection or {})

    async def search(
        self,
        query: str,
        *,
        collection: Collection,
        owner_id: str = GLOBAL_OWNER,
        limit: int = DEFAULT_LIMIT,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        try:
            embedding = await self._embeddings.embed_query(query)
            threshold = (
                min_score
                if min_score is not None
                else self._per_collection.get(collection, self._min_score)
            )
            return await self._store.search(
                embedding,
                collection=collection,
                owner_id=owner_id,
                limit=limit,
                min_score=threshold,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            logger.error(
                "retrieval_failed collection=%s owner=%s error=%s", collection, owner_id, exc
            )
            return []

    async def texts(
        self,
        query: str,
        *,
        collection: Collection,
        owner_id: str = GLOBAL_OWNER,
        limit: int = DEFAULT_LIMIT,
        min_score: float | None = None,
    ) -> list[str]:
        hits = await self.search(
            query, collection=collection, owner_id=owner_id, limit=limit, min_score=min_score
        )
        return [hit.text for hit in hits]


def dedupe(texts: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in texts:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result
