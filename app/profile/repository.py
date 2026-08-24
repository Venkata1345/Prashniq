"""Skill-observation persistence: protocol, in-memory, and Postgres."""

from __future__ import annotations

from typing import Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import skill_observations
from app.profile.schemas import SkillObservation


class SkillObservationRepository(Protocol):
    async def add(self, observations: Sequence[SkillObservation]) -> int:
        """Store observations; duplicates for the same interview are ignored.
        Returns how many were actually recorded."""
        ...

    async def for_candidate(self, candidate_id: str) -> list[SkillObservation]: ...


class InMemorySkillObservationRepository:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], SkillObservation] = {}

    async def add(self, observations: Sequence[SkillObservation]) -> int:
        added = 0
        for observation in observations:
            key = (observation.interview_id, observation.kind, observation.concept)
            if key not in self._rows:
                self._rows[key] = observation
                added += 1
        return added

    async def for_candidate(self, candidate_id: str) -> list[SkillObservation]:
        return [
            observation
            for observation in self._rows.values()
            if observation.candidate_id == candidate_id
        ]


class PostgresSkillObservationRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, observations: Sequence[SkillObservation]) -> int:
        if not observations:
            return 0
        statement = insert(skill_observations).values(
            [
                {
                    "candidate_id": item.candidate_id,
                    "concept": item.concept,
                    "kind": item.kind,
                    "score": item.score,
                    "interview_id": item.interview_id,
                    "interview_type": item.interview_type,
                    "observed_at": item.observed_at,
                }
                for item in observations
            ]
        # The unique constraint is the idempotency backstop: re-completing an
        # interview records nothing new.
        ).on_conflict_do_nothing(constraint="uq_skill_observations_interview_concept")

        from app.db.repositories import RepositoryError

        try:
            async with self._engine.begin() as connection:
                result = await connection.execute(statement)
        except SQLAlchemyError as exc:
            raise RepositoryError(f"saving skill observations failed: {exc}") from exc
        return result.rowcount or 0

    async def for_candidate(self, candidate_id: str) -> list[SkillObservation]:
        statement = select(skill_observations).where(
            skill_observations.c.candidate_id == candidate_id
        )
        from app.db.repositories import RepositoryError

        try:
            async with self._engine.connect() as connection:
                rows = (await connection.execute(statement)).mappings().all()
        except SQLAlchemyError as exc:
            raise RepositoryError(f"loading skill observations failed: {exc}") from exc
        return [
            SkillObservation(
                candidate_id=row["candidate_id"],
                concept=row["concept"],
                kind=row["kind"],
                score=row["score"],
                interview_id=row["interview_id"],
                interview_type=row["interview_type"],
                observed_at=row["observed_at"],
            )
            for row in rows
        ]
