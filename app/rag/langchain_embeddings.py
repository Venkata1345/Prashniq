"""LangChain-backed implementation of the embedding gateway.

Replaces the hand-written OpenAI adapter: any embedding model LangChain
supports plugs in here, selected by config. The rest of the app talks to
`EmbeddingGateway` and never imports LangChain.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from langchain_core.embeddings import Embeddings

from app.rag.embeddings import EmbeddingGateway
from app.rag.schemas import EmbeddingError

logger = logging.getLogger(__name__)


def build_openai_embeddings(
    *,
    model: str,
    dimensions: int,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict = {"model": model, "dimensions": dimensions, "timeout": timeout_seconds}
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


class LangChainEmbeddingGateway(EmbeddingGateway):
    def __init__(
        self,
        embeddings: Embeddings,
        *,
        model_name: str,
        dimensions: int,
        batch_size: int = 96,
    ) -> None:
        super().__init__(dimensions=dimensions, batch_size=batch_size)
        self._embeddings = embeddings
        self.model = model_name

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            vectors = await self._embeddings.aembed_documents(list(texts))
        except Exception as exc:  # provider SDK errors vary; map at the boundary
            raise EmbeddingError(f"embedding provider failed: {exc}") from exc

        logger.info(
            "embedding_call model=%s inputs=%d latency_ms=%d",
            self.model,
            len(texts),
            int((time.perf_counter() - started) * 1000),
        )
        return vectors
