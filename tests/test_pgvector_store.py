"""pgvector contract tests.

These are the only tests that need infrastructure. They run against a real
Postgres with the `vector` extension and are skipped otherwise:

    docker compose up -d db
    TEST_DATABASE_URL=postgresql+asyncpg://interview:interview@localhost:5432/interview \\
        .venv/Scripts/pytest tests/test_pgvector_store.py

Everything else in the suite uses `InMemoryVectorStore`, so a missing database
never hides a regression in the retrieval logic itself -- only in the SQL.
"""

from __future__ import annotations

import os

import pytest

from app.rag.embeddings import DeterministicEmbeddingGateway
from app.rag.indexer import Indexer
from app.rag.retriever import Retriever
from app.rag.schemas import GLOBAL_OWNER, DocumentChunk, EmbeddedChunk

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="set TEST_DATABASE_URL to run the pgvector contract tests",
    ),
]

# A dedicated table at a small width: the contract tests must never touch the
# table a running app uses.
DIMENSIONS = 64
TABLE_NAME = "rag_chunks_test"
CLAIMS = [
    "Built a RAG system using FAISS and FastAPI.",
    "Fine-tuned a transformer for document classification.",
    "Mentored two junior engineers.",
]


@pytest.fixture
async def store():
    from app.rag.pgvector_store import PgVectorStore

    store = PgVectorStore.from_url(
        TEST_DATABASE_URL, dimensions=DIMENSIONS, table_name=TABLE_NAME
    )
    await store.create_schema()
    for collection in ("resume", "job_description", "knowledge"):
        for owner in ("ctx-1", "ctx-2", GLOBAL_OWNER):
            await store.delete_owner(collection=collection, owner_id=owner)
    yield store
    await store.dispose()


@pytest.fixture
def embeddings() -> DeterministicEmbeddingGateway:
    return DeterministicEmbeddingGateway(dimensions=DIMENSIONS)


async def test_schema_creation_is_idempotent(store) -> None:
    await store.create_schema()
    assert await store.count(collection="resume", owner_id="ctx-1") == 0


async def test_upsert_search_round_trip(store, embeddings) -> None:
    indexer = Indexer(embeddings=embeddings, store=store)
    retriever = Retriever(embeddings=embeddings, store=store, min_score=0.05)

    await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")
    hits = await retriever.search("FAISS retrieval", collection="resume", owner_id="ctx-1")

    assert hits
    assert hits[0].text.startswith("Built a RAG system")
    # pgvector returns distance; the store must hand back similarity.
    assert 0.0 < hits[0].score <= 1.0


async def test_reindexing_updates_in_place(store, embeddings) -> None:
    indexer = Indexer(embeddings=embeddings, store=store)

    await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")
    await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")

    assert await store.count(collection="resume", owner_id="ctx-1") == len(CLAIMS)


async def test_owner_and_collection_isolation(store, embeddings) -> None:
    indexer = Indexer(embeddings=embeddings, store=store)
    retriever = Retriever(embeddings=embeddings, store=store, min_score=0.0)

    await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")
    await indexer.index_texts(["Wrote Terraform."], collection="resume", owner_id="ctx-2")

    assert len(await retriever.search("FAISS", collection="resume", owner_id="ctx-2")) == 1
    assert await retriever.search("FAISS", collection="knowledge") == []


async def test_metadata_and_deletion(store, embeddings) -> None:
    vector = await embeddings.embed_query("reranking")
    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    chunk_id="c1",
                    collection="knowledge",
                    text="Rerank with a cross-encoder.",
                    topic="reranking",
                    source="curated",
                ),
                embedding=vector,
            )
        ]
    )

    hit = (
        await store.search(vector, collection="knowledge", owner_id=GLOBAL_OWNER, limit=1)
    )[0]
    assert hit.chunk.topic == "reranking"
    assert hit.chunk.source == "curated"

    assert await store.delete_owner(collection="knowledge", owner_id=GLOBAL_OWNER) == 1
    assert await store.count(collection="knowledge", owner_id=GLOBAL_OWNER) == 0


async def test_empty_upsert_is_a_no_op(store) -> None:
    assert await store.upsert([]) == 0


async def test_a_dimension_change_fails_loudly_instead_of_at_write_time(store) -> None:
    """`CREATE TABLE IF NOT EXISTS` happily leaves a table built for a different
    embedding width in place; without this check the first write fails with an
    opaque driver error instead."""
    from app.rag.pgvector_store import PgVectorStore
    from app.rag.schemas import VectorStoreError

    wrong = PgVectorStore.from_url(
        TEST_DATABASE_URL, dimensions=DIMENSIONS * 2, table_name=TABLE_NAME
    )
    try:
        with pytest.raises(VectorStoreError, match="requires re-indexing"):
            await wrong.create_schema()
    finally:
        await wrong.dispose()
