"""Embedding gateway: batching, validation, determinism."""

from __future__ import annotations

from typing import Sequence

import pytest

from app.rag.embeddings import (
    DeterministicEmbeddingGateway,
    EmbeddingGateway,
    cosine_similarity,
)
from app.rag.schemas import EmbeddingError


class RecordingGateway(EmbeddingGateway):
    """Counts batches so batching behaviour is observable."""

    def __init__(self, *, dimensions: int = 4, batch_size: int = 2, width: int | None = None):
        super().__init__(dimensions=dimensions, batch_size=batch_size)
        self.batches: list[list[str]] = []
        self._width = width or dimensions

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1.0] * self._width for _ in texts]


class TruncatingGateway(RecordingGateway):
    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1.0] * self.dimensions for _ in texts][:-1]


class TestGatewayContract:
    async def test_inputs_are_split_into_batches_in_order(self) -> None:
        gateway = RecordingGateway(batch_size=2)
        vectors = await gateway.embed_documents(["a", "b", "c", "d", "e"])

        assert gateway.batches == [["a", "b"], ["c", "d"], ["e"]]
        assert len(vectors) == 5

    async def test_wrong_dimensionality_is_rejected(self) -> None:
        gateway = RecordingGateway(dimensions=4, width=3)
        with pytest.raises(EmbeddingError, match="4-dimensional"):
            await gateway.embed_documents(["a"])

    async def test_a_short_provider_response_is_rejected(self) -> None:
        gateway = TruncatingGateway()
        with pytest.raises(EmbeddingError, match="vectors for"):
            await gateway.embed_documents(["a", "b"])

    async def test_empty_text_is_rejected_rather_than_embedded(self) -> None:
        gateway = RecordingGateway()
        with pytest.raises(EmbeddingError, match="empty text"):
            await gateway.embed_documents(["fine", "   "])

    async def test_no_inputs_means_no_provider_call(self) -> None:
        gateway = RecordingGateway()
        assert await gateway.embed_documents([]) == []
        assert gateway.batches == []

    def test_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            RecordingGateway(dimensions=0)


class TestDeterministicEmbeddings:
    async def test_the_same_text_always_yields_the_same_vector(self) -> None:
        gateway = DeterministicEmbeddingGateway(dimensions=64)
        first = await gateway.embed_query("chunking and indexing strategy")
        second = await gateway.embed_query("chunking and indexing strategy")

        assert first == second
        assert len(first) == 64

    async def test_vectors_are_unit_length(self) -> None:
        gateway = DeterministicEmbeddingGateway(dimensions=64)
        vector = await gateway.embed_query("retrieval augmented generation")

        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    async def test_shared_vocabulary_produces_closer_vectors(self) -> None:
        gateway = DeterministicEmbeddingGateway(dimensions=512)
        query = await gateway.embed_query("FAISS vector index recall")
        related = await gateway.embed_query("Tuned the FAISS index for recall")
        unrelated = await gateway.embed_query("Mentored two junior engineers")

        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    async def test_punctuation_only_text_still_yields_a_valid_vector(self) -> None:
        gateway = DeterministicEmbeddingGateway(dimensions=16)
        vector = await gateway.embed_query("!!!")

        assert len(vector) == 16
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vectors() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
