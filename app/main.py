"""Application wiring: settings -> gateways -> retrieval -> orchestrator -> routes."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

# psycopg's async mode (the graph checkpointer) cannot run on Windows' default
# Proactor event loop; the Selector loop works for everything this app uses.
# Must be set at import time, before any event loop exists.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.contexts import router as contexts_router
from app.api.interviews import catalog_router, install_exception_handlers, router
from app.api.profiles import router as profiles_router
from app.context.ingestion import JobDescriptionIngestor, ResumeIngestor
from app.context.repository import InMemoryCandidateContextRepository
from app.context.service import CandidateContextService
from app.core.config import Settings, get_settings
from app.interview.evaluator import Evaluator
from app.interview.grounding import GroundingService
from app.interview.interviewer import Interviewer
from app.interview.orchestrator import InterviewOrchestrator
from app.interview.repository import InMemoryInterviewRepository
from app.llm.factory import build_gateway
from app.profile.repository import InMemorySkillObservationRepository
from app.profile.service import SkillProfileService
from app.rag.factory import build_embeddings, build_retriever, build_vector_store
from app.rag.indexer import Indexer
from app.rag.knowledge import seed_knowledge_base
from app.rag.schemas import EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    gateway = build_gateway(settings)
    embeddings = build_embeddings(settings)
    store = build_vector_store(settings)
    indexer = Indexer(embeddings=embeddings, store=store)
    retriever = build_retriever(settings, embeddings=embeddings, store=store)

    # One switch decides where domain state lives: with a database, interviews,
    # contexts and skill observations are durable rows; without one, everything
    # is in-process and lost on restart (fine for local dev, said out loud).
    db_engine = None
    checkpointer_factory = None
    checkpointer_pool = None
    if settings.database_url:
        from sqlalchemy.ext.asyncio import create_async_engine

        from app.db.repositories import (
            PostgresCandidateContextRepository,
            PostgresInterviewRepository,
        )
        from app.profile.repository import PostgresSkillObservationRepository

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        db_engine = create_async_engine(settings.database_url)
        # The graph checkpointer speaks psycopg, not asyncpg. The pool opens in
        # lifespan; the saver is built lazily (it needs a running event loop).
        checkpointer_pool = AsyncConnectionPool(
            settings.database_url.replace("+asyncpg", ""),
            open=False,
            kwargs={"autocommit": True},
        )
        checkpointer_factory = lambda: AsyncPostgresSaver(checkpointer_pool)  # noqa: E731
        interview_repository = PostgresInterviewRepository(db_engine)
        context_repository = PostgresCandidateContextRepository(db_engine)
        observation_repository = PostgresSkillObservationRepository(db_engine)
    else:
        logger.warning(
            "persistence=memory - interviews, contexts and skill profiles are "
            "lost on restart; set database_url for durable state"
        )
        interview_repository = InMemoryInterviewRepository()
        context_repository = InMemoryCandidateContextRepository()
        observation_repository = InMemorySkillObservationRepository()

    profile_service = SkillProfileService(observation_repository)
    context_service = CandidateContextService(
        resume_ingestor=ResumeIngestor(gateway),
        job_ingestor=JobDescriptionIngestor(gateway),
        repository=context_repository,
        indexer=indexer,
    )
    orchestrator = InterviewOrchestrator(
        evaluator=Evaluator(gateway),
        interviewer=Interviewer(gateway),
        repository=interview_repository,
        context_service=context_service,
        grounding=GroundingService(retriever),
        profile_service=profile_service,
        checkpointer_factory=checkpointer_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Prepare the vector schema and load the knowledge base.

        Both are idempotent, and a failure here degrades retrieval rather than
        stopping the service -- interviews still run ungrounded.
        """
        if settings.database_url:
            # Alembic owns the relational schema. Run in a thread: alembic's
            # async env starts its own event loop.
            import asyncio

            from app.db.migrate import upgrade_to_head

            await asyncio.to_thread(upgrade_to_head, settings.database_url)
            if checkpointer_pool is not None:
                await checkpointer_pool.open()
                await checkpointer_factory().setup()  # idempotent tables
        try:
            create_schema = getattr(store, "create_schema", None)
            if create_schema is not None:
                await create_schema()
            if settings.seed_knowledge_base:
                await seed_knowledge_base(indexer)
        except (EmbeddingError, VectorStoreError) as exc:
            logger.error("retrieval_startup_failed error=%s", exc)
            app.state.retrieval_ready = False
        else:
            app.state.retrieval_ready = True

        yield

        dispose = getattr(store, "dispose", None)
        if dispose is not None:
            await dispose()
        if checkpointer_pool is not None:
            await checkpointer_pool.close()
        if db_engine is not None:
            await db_engine.dispose()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.retrieval_ready = False
    app.state.context_service = context_service
    app.state.orchestrator = orchestrator
    app.state.profile_service = profile_service

    app.include_router(router)
    app.include_router(contexts_router)
    app.include_router(profiles_router)
    app.include_router(catalog_router)
    install_exception_handlers(app)

    # The frontend: the built React app (frontend/dist) served from the same
    # origin, so it needs no CORS and no separate deployment. Falls back to the
    # legacy no-build SPA in app/static when no build exists.
    react_dist = Path(__file__).parent.parent / "frontend" / "dist"
    static_dir = react_dist if (react_dist / "index.html").exists() else Path(__file__).parent / "static"
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, object]:
        from app.llm.factory import resolve_model

        # Groq free tier, per org (checked 2026-08-24): the demo's capacity.
        rate_limits = (
            {"requests_per_minute": 30, "tokens_per_minute": 6000, "requests_per_day": 14400}
            if settings.llm_provider == "groq"
            else None
        )
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "llm_model": resolve_model(settings.llm_provider, settings.llm_model),
            "llm_rate_limits": rate_limits,
            "embedding_provider": settings.embedding_provider,
            "vector_store": settings.resolved_vector_store,
            "persistence": "postgres" if settings.database_url else "memory",
            "retrieval_ready": app.state.retrieval_ready,
        }

    return app


# Served with `uvicorn app.main:create_app --factory`. There is deliberately no
# module-level app: building one at import time would make a missing provider
# key an import error, including in tests that never touch that provider.
