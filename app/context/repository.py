"""Candidate-context persistence.

Same seam as `InterviewRepository`: in memory for now, a protocol so a
SQLAlchemy implementation can replace it without touching the services.
"""

from __future__ import annotations

from typing import Protocol

from app.context.schemas import CandidateContext


class CandidateContextNotFound(KeyError):
    pass


class CandidateContextRepository(Protocol):
    async def add(self, context: CandidateContext) -> None: ...

    async def get(self, context_id: str) -> CandidateContext: ...


class InMemoryCandidateContextRepository:
    def __init__(self) -> None:
        self._contexts: dict[str, CandidateContext] = {}

    async def add(self, context: CandidateContext) -> None:
        self._contexts[context.context_id] = context

    async def get(self, context_id: str) -> CandidateContext:
        try:
            return self._contexts[context_id]
        except KeyError as exc:
            raise CandidateContextNotFound(context_id) from exc
