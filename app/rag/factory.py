"""Builds the retrieval stack for the configured providers.

The one place that knows which embedding provider and which vector store exist.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.rag.embeddings import DeterministicEmbeddingGateway, EmbeddingGateway
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore, VectorStore

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings) -> EmbeddingGateway:
    if settings.embedding_provider == "openai":
        from app.rag.langchain_embeddings import (
            LangChainEmbeddingGateway,
            build_openai_embeddings,
        )

        try:
            lc_embeddings = build_openai_embeddings(
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                api_key=settings.openai_api_key,
            )
        except Exception as exc:
            # The OpenAI client fails at construction without a key; say
            # plainly what to do about it.
            raise ValueError(
                "embedding_provider=openai needs OPENAI_API_KEY. Set "
                "EMBEDDING_PROVIDER=fake to run locally with deterministic "
                "vectors."
            ) from exc
        return LangChainEmbeddingGateway(
            lc_embeddings,
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        )

    if settings.embedding_provider == "fake":
        return DeterministicEmbeddingGateway(
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        )

    raise ValueError(f"unsupported embedding_provider: {settings.embedding_provider}")


def build_vector_store(settings: Settings) -> VectorStore:
    backend = settings.resolved_vector_store
    if backend == "pgvector":
        from app.rag.pgvector_store import PgVectorStore

        if not settings.database_url:
            raise ValueError("vector_store=pgvector requires database_url")
        logger.info("vector_store=pgvector")
        return PgVectorStore.from_url(
            settings.database_url, dimensions=settings.embedding_dimensions
        )

    logger.warning(
        "vector_store=memory - retrieval is in-process and lost on restart; "
        "set database_url to use pgvector"
    )
    return InMemoryVectorStore()


def build_retriever(
    settings: Settings, *, embeddings: EmbeddingGateway, store: VectorStore
) -> Retriever:
    return Retriever(
        embeddings=embeddings,
        store=store,
        min_score=settings.retrieval_min_score,
        per_collection={
            "knowledge": settings.knowledge_min_score,
            "resume": settings.claim_min_score,
            "job_description": settings.claim_min_score,
        },
    )
