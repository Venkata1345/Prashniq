"""Vector store, indexer and retriever, end to end over deterministic vectors."""

from __future__ import annotations

from typing import Sequence

import pytest

from app.rag.embeddings import DeterministicEmbeddingGateway
from app.rag.indexer import Indexer, stable_chunk_id
from app.rag.knowledge import KNOWLEDGE_NOTES, seed_knowledge_base
from app.rag.retriever import Retriever, dedupe
from app.rag.schemas import (
    GLOBAL_OWNER,
    Collection,
    DocumentChunk,
    EmbeddingError,
    RetrievedChunk,
    VectorStoreError,
)
from app.rag.store import InMemoryVectorStore

CLAIMS = [
    "Built a RAG system using FAISS and FastAPI.",
    "Fine-tuned a transformer for document classification.",
    "Mentored two junior engineers.",
]


def embeddings() -> DeterministicEmbeddingGateway:
    return DeterministicEmbeddingGateway()


def pipeline() -> tuple[Indexer, Retriever, InMemoryVectorStore]:
    store = InMemoryVectorStore()
    gateway = embeddings()
    return (
        Indexer(embeddings=gateway, store=store),
        Retriever(embeddings=gateway, store=store, min_score=0.05),
        store,
    )


class TestVectorStore:
    async def test_upsert_is_idempotent_on_chunk_identity(self) -> None:
        indexer, _, store = pipeline()

        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")

        assert await store.count(collection="resume", owner_id="ctx-1") == 3

    async def test_owners_are_isolated_from_each_other(self) -> None:
        indexer, retriever, store = pipeline()
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")
        await indexer.index_texts(["Wrote Terraform modules."], collection="resume", owner_id="ctx-2")

        hits = await retriever.texts("FAISS", collection="resume", owner_id="ctx-2")

        assert hits == []
        assert await store.count(collection="resume", owner_id="ctx-2") == 1

    async def test_collections_are_isolated_from_each_other(self) -> None:
        indexer, retriever, _ = pipeline()
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")

        assert await retriever.texts("FAISS", collection="knowledge") == []

    async def test_deleting_an_owner_removes_only_their_rows(self) -> None:
        indexer, _, store = pipeline()
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")
        await indexer.index_texts(["Wrote Terraform."], collection="resume", owner_id="ctx-2")

        removed = await store.delete_owner(collection="resume", owner_id="ctx-1")

        assert removed == 3
        assert await store.count(collection="resume", owner_id="ctx-1") == 0
        assert await store.count(collection="resume", owner_id="ctx-2") == 1

    async def test_results_are_ordered_by_similarity_and_capped(self) -> None:
        indexer, retriever, _ = pipeline()
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")

        # Overlaps both technical claims, and neither the mentoring one.
        hits = await retriever.search(
            "FAISS transformer classification", collection="resume", owner_id="ctx-1", limit=3
        )

        assert len(hits) == 2
        assert hits[0].text.startswith("Fine-tuned a transformer")
        assert hits[0].score >= hits[1].score
        assert all("Mentored" not in hit.text for hit in hits)

    async def test_the_limit_caps_the_number_of_results(self) -> None:
        indexer, retriever, _ = pipeline()
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")

        hits = await retriever.search(
            "FAISS transformer classification", collection="resume", owner_id="ctx-1", limit=1
        )

        assert len(hits) == 1

    async def test_the_score_threshold_filters_out_noise(self) -> None:
        indexer, retriever, _ = pipeline()
        await indexer.index_texts(CLAIMS, collection="resume", owner_id="ctx-1")

        assert (
            await retriever.texts(
                "FAISS", collection="resume", owner_id="ctx-1", min_score=0.99
            )
            == []
        )


class TestIndexer:
    async def test_chunk_ids_are_content_addressed(self) -> None:
        assert stable_chunk_id("Built a RAG system.") == stable_chunk_id(
            "  built a rag system.  "
        )
        assert stable_chunk_id("a") != stable_chunk_id("b")

    async def test_blank_texts_are_skipped(self) -> None:
        indexer, _, store = pipeline()

        written = await indexer.index_texts(
            ["A real claim.", "   ", ""], collection="resume", owner_id="ctx-1"
        )

        assert written == 1
        assert await store.count(collection="resume", owner_id="ctx-1") == 1

    async def test_long_documents_are_chunked_before_indexing(self) -> None:
        indexer, _, store = pipeline()
        document = "\n\n".join(f"Paragraph {i} about retrieval systems." for i in range(6))

        await indexer.index_document(
            document, collection="knowledge", owner_id=GLOBAL_OWNER, max_chars=60
        )

        assert await store.count(collection="knowledge", owner_id=GLOBAL_OWNER) > 1

    async def test_metadata_survives_the_round_trip(self) -> None:
        indexer, retriever, _ = pipeline()
        await indexer.index_chunks(
            [
                DocumentChunk(
                    chunk_id="c1",
                    collection="knowledge",
                    text="Reranking uses a cross-encoder over a shallow candidate set.",
                    topic="reranking",
                    source="curated",
                )
            ]
        )

        hit = (await retriever.search("reranking cross-encoder", collection="knowledge"))[0]

        assert hit.chunk.topic == "reranking"
        assert hit.chunk.source == "curated"


class TestKnowledgeBase:
    async def test_seeding_indexes_every_note_and_is_repeatable(self) -> None:
        indexer, _, store = pipeline()

        first = await seed_knowledge_base(indexer)
        await seed_knowledge_base(indexer)

        assert first == len(KNOWLEDGE_NOTES)
        assert await store.count(collection="knowledge", owner_id=GLOBAL_OWNER) == len(
            KNOWLEDGE_NOTES
        )

    async def test_a_topic_retrieves_its_own_reference_note(self) -> None:
        indexer, retriever, _ = pipeline()
        await seed_knowledge_base(indexer)

        hits = await retriever.search("KV caching and inference cost", collection="knowledge")

        assert hits
        assert hits[0].chunk.topic == "KV caching and inference cost"


class TestDegradation:
    """Retrieval failures must cost grounding, never the interview."""

    class BrokenEmbeddings(DeterministicEmbeddingGateway):
        async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            raise EmbeddingError("provider is down")

    class BrokenStore(InMemoryVectorStore):
        async def search(self, *args, **kwargs) -> list[RetrievedChunk]:
            raise VectorStoreError("database is down")

    async def test_an_embedding_outage_returns_no_results(self) -> None:
        retriever = Retriever(embeddings=self.BrokenEmbeddings(), store=InMemoryVectorStore())

        assert await retriever.texts("anything", collection="knowledge") == []

    async def test_a_store_outage_returns_no_results(self) -> None:
        retriever = Retriever(embeddings=embeddings(), store=self.BrokenStore())

        assert await retriever.texts("anything", collection="knowledge") == []

    async def test_an_empty_query_never_reaches_the_provider(self) -> None:
        retriever = Retriever(embeddings=self.BrokenEmbeddings(), store=InMemoryVectorStore())

        assert await retriever.texts("   ", collection="knowledge") == []


def test_dedupe_preserves_order_and_folds_case() -> None:
    assert dedupe(["RAG", "rag ", "FAISS", ""]) == ["RAG", "FAISS"]


@pytest.mark.parametrize("collection", ["resume", "job_description", "knowledge"])
def test_every_collection_is_a_valid_chunk_target(collection: Collection) -> None:
    chunk = DocumentChunk(chunk_id="c", collection=collection, text="text")
    assert chunk.collection == collection


class TestPerCollectionThresholds:
    async def test_each_collection_gets_its_own_threshold(self) -> None:
        store = InMemoryVectorStore()
        gateway = embeddings()
        indexer = Indexer(embeddings=gateway, store=store)
        await indexer.index_texts(["FAISS index tuning"], collection="knowledge")
        await indexer.index_texts(["FAISS index tuning"], collection="resume", owner_id="c1")

        retriever = Retriever(
            embeddings=gateway,
            store=store,
            min_score=0.05,
            per_collection={"knowledge": 1.01},  # unreachable for knowledge only
        )

        assert await retriever.texts("FAISS index tuning", collection="knowledge") == []
        assert await retriever.texts(
            "FAISS index tuning", collection="resume", owner_id="c1"
        ) == ["FAISS index tuning"]

    async def test_an_explicit_min_score_still_overrides_everything(self) -> None:
        store = InMemoryVectorStore()
        gateway = embeddings()
        indexer = Indexer(embeddings=gateway, store=store)
        await indexer.index_texts(["FAISS index tuning"], collection="knowledge")

        retriever = Retriever(
            embeddings=gateway, store=store, per_collection={"knowledge": 1.01}
        )

        hits = await retriever.texts(
            "FAISS index tuning", collection="knowledge", min_score=0.05
        )
        assert hits == ["FAISS index tuning"]


class TestEmbeddingText:
    async def test_search_matches_the_embedding_text_but_returns_the_display_text(
        self,
    ) -> None:
        store = InMemoryVectorStore()
        gateway = embeddings()
        indexer = Indexer(embeddings=gateway, store=store)
        retriever = Retriever(embeddings=gateway, store=store, min_score=0.05)

        await indexer.index_chunks(
            [
                DocumentChunk(
                    chunk_id="c1",
                    collection="resume",
                    owner_id="c1",
                    text="Built an internal documentation assistant.",
                    embedding_text=(
                        "Built an internal documentation assistant. "
                        "Skills demonstrated: RAG, FAISS, retrieval."
                    ),
                )
            ]
        )

        # The query vocabulary only appears in the embedding text.
        hits = await retriever.texts("FAISS retrieval", collection="resume", owner_id="c1")

        assert hits == ["Built an internal documentation assistant."]
