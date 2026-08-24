"""Interview persistence.

Phase 1 stores state in memory. The protocol is the seam a SQLAlchemy
repository slots into later without the orchestrator changing.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol

from app.interview.state import InterviewState


class InterviewNotFound(KeyError):
    pass


class InterviewRepository(Protocol):
    async def add(self, state: InterviewState) -> None: ...

    async def get(self, interview_id: str) -> InterviewState: ...

    async def save(self, state: InterviewState) -> None: ...

    def lock(self, interview_id: str) -> AsyncIterator[None]: ...


class InMemoryInterviewRepository:
    def __init__(self) -> None:
        self._states: dict[str, InterviewState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def add(self, state: InterviewState) -> None:
        self._states[state.interview_id] = state

    async def get(self, interview_id: str) -> InterviewState:
        try:
            return self._states[interview_id]
        except KeyError as exc:
            raise InterviewNotFound(interview_id) from exc

    async def save(self, state: InterviewState) -> None:
        self._states[state.interview_id] = state

    @asynccontextmanager
    async def lock(self, interview_id: str) -> AsyncIterator[None]:
        """Serialises turns for one interview so a double-submitted answer
        cannot interleave with itself."""
        lock = self._locks.setdefault(interview_id, asyncio.Lock())
        async with lock:
            yield
