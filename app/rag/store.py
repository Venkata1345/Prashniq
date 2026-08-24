"""Vector store protocol and an in-memory implementation.

`PgVectorStore` (see `pgvector_store.py`) is the production store. The
in-memory one here is a test double -- the same role `FakeLLMGateway` plays for
the LLM -- so the retrieval pipeline can be tested exhaustively without a
database. It is also what `LLM_PROVIDER=fake` uses for a zero-setup local run.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from app.rag.embeddings import cosine_similarity
from app.rag.schemas import Collection, EmbeddedChunk, RetrievedChunk


class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> int:
        """Insert or replace chunks by (collection, owner_id, chunk_id)."""
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        collection: Collection,
        owner_id: str,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Nearest chunks by cosine similarity, best first."""
        ...

    async def delete_owner(self, *, collection: Collection, owner_id: str) -> int:
        """Remove everything belonging to one owner in one collection."""
        ...

    async def count(self, *, collection: Collection, owner_id: str) -> int: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], EmbeddedChunk] = {}

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> int:
        for item in chunks:
            self._rows[_key(item)] = item
        return len(chunks)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        collection: Collection,
        owner_id: str,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=item.chunk, score=cosine_similarity(embedding, item.embedding))
            for (coll, owner, _), item in self._rows.items()
            if coll == collection and owner == owner_id
        ]
        hits = [hit for hit in scored if hit.score >= min_score]
        # Ties break on chunk_id so results are stable across runs.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits[:limit]

    async def delete_owner(self, *, collection: Collection, owner_id: str) -> int:
        doomed = [
            key for key in self._rows if key[0] == collection and key[1] == owner_id
        ]
        for key in doomed:
            del self._rows[key]
        return len(doomed)

    async def count(self, *, collection: Collection, owner_id: str) -> int:
        return sum(
            1 for key in self._rows if key[0] == collection and key[1] == owner_id
        )


def _key(item: EmbeddedChunk) -> tuple[str, str, str]:
    return (item.chunk.collection, item.chunk.owner_id, item.chunk.chunk_id)
