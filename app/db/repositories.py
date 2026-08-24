"""Postgres implementations of the persistence protocols.

Drop-in replacements for the in-memory repositories: same protocols, so the
orchestrator and services never change. Interviews survive restarts, and the
interview lock is a Postgres advisory lock, so double-submitted answers are
serialised even across multiple workers.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.context.repository import CandidateContextNotFound
from app.context.schemas import CandidateContext
from app.db.models import candidate_contexts, interviews
from app.interview.repository import InterviewNotFound
from app.interview.state import InterviewState

logger = logging.getLogger(__name__)


class RepositoryError(RuntimeError):
    """Database failure while persisting or loading domain state.

    Unlike retrieval, persistence failures must NOT degrade silently -- losing
    an interview turn is worse than failing the request."""


class PostgresInterviewRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, state: InterviewState) -> None:
        await self._write(state)

    async def save(self, state: InterviewState) -> None:
        await self._write(state)

    async def get(self, interview_id: str) -> InterviewState:
        statement = select(interviews.c.state).where(
            interviews.c.interview_id == interview_id
        )
        try:
            async with self._engine.connect() as connection:
                row = (await connection.execute(statement)).first()
        except SQLAlchemyError as exc:
            raise RepositoryError(f"loading interview failed: {exc}") from exc
        if row is None:
            raise InterviewNotFound(interview_id)
        return InterviewState.model_validate(row[0])

    @asynccontextmanager
    async def lock(self, interview_id: str) -> AsyncIterator[None]:
        """Advisory lock keyed on the interview id, held for the turn.

        Session-scoped, so acquire and release happen on the same connection,
        which stays checked out for the duration of the block."""
        try:
            async with self._engine.connect() as connection:
                await connection.execute(
                    text("SELECT pg_advisory_lock(hashtext(:id))"), {"id": interview_id}
                )
                try:
                    yield
                finally:
                    await connection.execute(
                        text("SELECT pg_advisory_unlock(hashtext(:id))"),
                        {"id": interview_id},
                    )
        except SQLAlchemyError as exc:
            raise RepositoryError(f"interview lock failed: {exc}") from exc

    async def _write(self, state: InterviewState) -> None:
        document = json.loads(state.model_dump_json())
        now = datetime.now(timezone.utc)
        statement = insert(interviews).values(
            interview_id=state.interview_id,
            candidate_id=state.candidate_id,
            interview_type=state.interview_type,
            status=state.status.value,
            created_at=state.created_at,
            updated_at=now,
            state=document,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["interview_id"],
            set_={"status": statement.excluded.status, "updated_at": now,
                  "state": statement.excluded.state},
        )
        try:
            async with self._engine.begin() as connection:
                await connection.execute(statement)
        except SQLAlchemyError as exc:
            raise RepositoryError(f"saving interview failed: {exc}") from exc


class PostgresCandidateContextRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, context: CandidateContext) -> None:
        statement = insert(candidate_contexts).values(
            context_id=context.context_id,
            candidate_id=context.candidate_id,
            created_at=context.created_at,
            data=json.loads(context.model_dump_json()),
        )
        statement = statement.on_conflict_do_update(
            index_elements=["context_id"], set_={"data": statement.excluded.data}
        )
        try:
            async with self._engine.begin() as connection:
                await connection.execute(statement)
        except SQLAlchemyError as exc:
            raise RepositoryError(f"saving candidate context failed: {exc}") from exc

    async def get(self, context_id: str) -> CandidateContext:
        statement = select(candidate_contexts.c.data).where(
            candidate_contexts.c.context_id == context_id
        )
        try:
            async with self._engine.connect() as connection:
                row = (await connection.execute(statement)).first()
        except SQLAlchemyError as exc:
            raise RepositoryError(f"loading candidate context failed: {exc}") from exc
        if row is None:
            raise CandidateContextNotFound(context_id)
        return CandidateContext.model_validate(row[0])
