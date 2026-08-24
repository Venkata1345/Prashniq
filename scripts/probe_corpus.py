"""Probe the real knowledge corpus: what does retrieval actually return?

Runs interview-style queries against the configured vector store (pgvector
when DATABASE_URL is set) and prints the top hits with score, topic and
source. Run it after ingesting new material to see what grounding the
interviewer and evaluator will actually receive.

    .venv/Scripts/python -m scripts.probe_corpus
    .venv/Scripts/python -m scripts.probe_corpus "your own question here"
"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.config import Settings
from app.rag.factory import build_embeddings, build_retriever, build_vector_store
from app.rag.schemas import GLOBAL_OWNER

# One probe per interview mode, phrased the way the engine phrases topic
# queries -- not the way a book phrases its headings.
DEFAULT_PROBES = [
    "bias-variance tradeoff",
    "regularization L1 versus L2 weight decay",
    "why divide attention scores by the square root of the head dimension",
    "KV cache memory cost during decoding",
    "chunk size and overlap for retrieval",
    "evaluating a RAG system retrieval versus generation",
    "agent tool use failure modes",
    "monitoring feature drift in production models",
]


async def run(queries: list[str]) -> int:
    settings = Settings()
    embeddings = build_embeddings(settings)
    store = build_vector_store(settings)
    retriever = build_retriever(settings, embeddings=embeddings, store=store)

    total = await store.count(collection="knowledge", owner_id=GLOBAL_OWNER)
    print(f"knowledge collection: {total} chunks "
          f"(store={settings.resolved_vector_store}, "
          f"threshold>={settings.knowledge_min_score})\n")

    empty = 0
    for query in queries:
        hits = await retriever.search(query, collection="knowledge", limit=3)
        print(f'"{query}"')
        if not hits:
            empty += 1
            print("    (no hits above threshold)")
        for hit in hits:
            preview = " ".join(hit.text.split())[:90]
            print(f"    {hit.score:.3f}  [{hit.chunk.topic}]  {hit.chunk.source}")
            print(f"           {preview}...")
        print()

    print(f"{len(queries) - empty}/{len(queries)} probes returned grounding")
    dispose = getattr(store, "dispose", None)
    if dispose is not None:
        await dispose()
    return 0


def main() -> None:
    queries = sys.argv[1:] or DEFAULT_PROBES
    sys.exit(asyncio.run(run(queries)))


if __name__ == "__main__":
    main()
