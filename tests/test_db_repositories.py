"""Contract tests for the Postgres repositories and the Alembic migration.

Like the pgvector tests: they need a live database and are skipped without
TEST_DATABASE_URL. Everything they verify about *behaviour* is already covered
against the in-memory implementations; what only these can verify is the SQL,
the migration, the JSONB round-trip and the advisory lock.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.context.repository import CandidateContextNotFound
from app.context.schemas import CandidateContext, ResumeClaim, ResumeProfile
from app.interview.repository import InterviewNotFound
from app.interview.schemas import InterviewStatus
from app.interview.state import InterviewState
from app.profile.schemas import SkillObservation

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="set TEST_DATABASE_URL to run the Postgres contract tests",
    ),
]

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def migrated() -> None:
    """Apply migrations once per module; a second run must be a no-op."""
    from app.db.migrate import upgrade_to_head

    upgrade_to_head(TEST_DATABASE_URL)
    upgrade_to_head(TEST_DATABASE_URL)


@pytest.fixture
async def engine(migrated):
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE interviews, candidate_contexts, skill_observations")
        )
    yield engine
    await engine.dispose()


def make_state(interview_id: str = "iv-1") -> InterviewState:
    return InterviewState(
        interview_id=interview_id,
        candidate_id="cand-1",
        interview_type="ml_fundamentals",
        remaining_topics=["regularization"],
        created_at=NOW,
    )


class TestInterviewRepository:
    async def test_round_trip_preserves_the_full_state_document(self, engine) -> None:
        from app.db.repositories import PostgresInterviewRepository

        repository = PostgresInterviewRepository(engine)
        state = make_state()
        await repository.add(state)

        loaded = await repository.get("iv-1")
        assert loaded == state

    async def test_save_updates_in_place(self, engine) -> None:
        from app.db.repositories import PostgresInterviewRepository

        repository = PostgresInterviewRepository(engine)
        state = make_state()
        await repository.add(state)
        await repository.save(
            state.model_copy(update={"status": InterviewStatus.COMPLETED})
        )

        assert (await repository.get("iv-1")).status is InterviewStatus.COMPLETED

    async def test_missing_interviews_raise_the_domain_error(self, engine) -> None:
        from app.db.repositories import PostgresInterviewRepository

        with pytest.raises(InterviewNotFound):
            await PostgresInterviewRepository(engine).get("nope")

    async def test_the_advisory_lock_actually_excludes(self, engine) -> None:
        from app.db.repositories import PostgresInterviewRepository

        repository = PostgresInterviewRepository(engine)
        async with repository.lock("iv-1"):
            # A second session must not be able to take the same lock.
            async with engine.connect() as other:
                taken = (
                    await other.execute(
                        text("SELECT pg_try_advisory_lock(hashtext('iv-1'))")
                    )
                ).scalar_one()
                assert taken is False

        # Released after the block.
        async with engine.connect() as other:
            taken = (
                await other.execute(
                    text("SELECT pg_try_advisory_lock(hashtext('iv-1'))")
                )
            ).scalar_one()
            assert taken is True
            await other.execute(text("SELECT pg_advisory_unlock(hashtext('iv-1'))"))

    async def test_concurrent_turns_serialise_instead_of_interleaving(self, engine) -> None:
        from app.db.repositories import PostgresInterviewRepository

        repository = PostgresInterviewRepository(engine)
        await repository.add(make_state())
        order: list[str] = []

        async def turn(name: str) -> None:
            async with repository.lock("iv-1"):
                order.append(f"{name}:in")
                await asyncio.sleep(0.05)
                order.append(f"{name}:out")

        await asyncio.gather(turn("a"), turn("b"))

        # Whichever entered first must exit before the other enters.
        assert order[1].endswith(":out")


class TestCandidateContextRepository:
    async def test_round_trip(self, engine) -> None:
        from app.db.repositories import PostgresCandidateContextRepository

        repository = PostgresCandidateContextRepository(engine)
        context = CandidateContext(
            context_id="ctx-1",
            candidate_id="cand-1",
            resume=ResumeProfile(
                claims=[ResumeClaim(text="Built a RAG system.", skills=["RAG"])]
            ),
            created_at=NOW,
        )
        await repository.add(context)

        assert await repository.get("ctx-1") == context

    async def test_missing_contexts_raise_the_domain_error(self, engine) -> None:
        from app.db.repositories import PostgresCandidateContextRepository

        with pytest.raises(CandidateContextNotFound):
            await PostgresCandidateContextRepository(engine).get("nope")


class TestSkillObservationRepository:
    @staticmethod
    def observation(concept: str = "RAG", interview_id: str = "iv-1") -> SkillObservation:
        return SkillObservation(
            candidate_id="cand-1",
            concept=concept,
            kind="topic",
            score=7.0,
            interview_id=interview_id,
            interview_type="ml_fundamentals",
            observed_at=NOW,
        )

    async def test_add_and_read_back(self, engine) -> None:
        from app.profile.repository import PostgresSkillObservationRepository

        repository = PostgresSkillObservationRepository(engine)
        recorded = await repository.add(
            [self.observation("RAG"), self.observation("FAISS")]
        )

        assert recorded == 2
        loaded = await repository.for_candidate("cand-1")
        assert {item.concept for item in loaded} == {"RAG", "FAISS"}
        assert await repository.for_candidate("someone-else") == []

    async def test_re_recording_an_interview_is_idempotent(self, engine) -> None:
        from app.profile.repository import PostgresSkillObservationRepository

        repository = PostgresSkillObservationRepository(engine)
        await repository.add([self.observation()])
        recorded_again = await repository.add([self.observation()])

        assert recorded_again == 0
        assert len(await repository.for_candidate("cand-1")) == 1
