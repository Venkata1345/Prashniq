"""Vector-backed evidence lookup for blueprint construction.

Implements `app.context.blueprint.EvidenceLookup` with semantic search, so a
required skill matches a claim that means the same thing rather than one that
happens to share a word.
"""

from __future__ import annotations

from app.rag.retriever import Retriever, dedupe

CLAIMS_PER_SKILL = 2


class VectorEvidenceLookup:
    def __init__(self, retriever: Retriever, *, owner_id: str) -> None:
        self._retriever = retriever
        self._owner_id = owner_id

    async def claims_for(self, skill: str) -> list[str]:
        return dedupe(
            await self._retriever.texts(
                skill,
                collection="resume",
                owner_id=self._owner_id,
                limit=CLAIMS_PER_SKILL,
            )
        )
