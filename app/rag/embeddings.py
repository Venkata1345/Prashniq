"""The embedding gateway abstraction.

Mirrors `LLMGateway`: the domain asks for vectors, never for a provider.
Documents and queries are separate methods because some providers (and some
future models) embed them asymmetrically.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Sequence

from app.rag.schemas import EmbeddingError

DEFAULT_BATCH_SIZE = 96


class EmbeddingGateway(ABC):
    """Adapters implement `_embed_batch`; batching and validation are shared."""

    def __init__(self, *, dimensions: int, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.batch_size = batch_size

    @abstractmethod
    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one batch, in order."""

    async def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise EmbeddingError("cannot embed empty text")
        if not cleaned:
            return []

        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            raw = await self._embed_batch(batch)
            if len(raw) != len(batch):
                raise EmbeddingError(
                    f"provider returned {len(raw)} vectors for {len(batch)} inputs"
                )
            vectors.extend(self._validate(vector) for vector in raw)
        return vectors

    async def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    def _validate(self, vector: Sequence[float]) -> tuple[float, ...]:
        if len(vector) != self.dimensions:
            raise EmbeddingError(
                f"expected {self.dimensions}-dimensional vectors, got {len(vector)}"
            )
        return tuple(float(value) for value in vector)


class DeterministicEmbeddingGateway(EmbeddingGateway):
    """Hashed bag-of-tokens embeddings, used by every test and by the `fake`
    provider.

    Each token is hashed to a fixed set of dimensions, so texts sharing tokens
    end up with genuinely similar vectors. That is enough to exercise ranking,
    thresholds and ordering deterministically -- it is not a semantic model, and
    is never the production path.
    """

    # Wide by default: hashing collisions between unrelated texts are what put
    # a floor under the similarity score, and 2048 keeps that floor (~0.08) well
    # below the retriever's threshold.
    DEFAULT_DIMENSIONS = 2048

    def __init__(
        self, *, dimensions: int = DEFAULT_DIMENSIONS, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        super().__init__(dimensions=dimensions, batch_size=batch_size)

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token for token in _tokenize(text) if token]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(3):
                index = int.from_bytes(digest[offset * 4 : offset * 4 + 4], "big")
                sign = 1.0 if digest[12 + offset] % 2 == 0 else -1.0
                vector[index % self.dimensions] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Empty-after-tokenisation text still needs a valid unit vector.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


# Only the deterministic gateway uses this: real embedding models handle
# function words themselves, but hashed bag-of-tokens similarity is swamped by
# them.
_STOPWORDS = frozenset(
    """a an and are as at be by for from in into is it its of on or that the
    their this to with what how why when which would you your""".split()
)


def _tokenize(text: str) -> list[str]:
    words = (
        "".join(character for character in word if character.isalnum())
        for word in text.lower().split()
    )
    return [word for word in words if word and word not in _STOPWORDS]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
