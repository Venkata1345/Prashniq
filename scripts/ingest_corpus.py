"""Index the knowledge corpus: drop books into corpus/, run this.

    # index new/changed files (needs OPENAI_API_KEY for real embeddings,
    # DATABASE_URL for the durable vector store)
    .venv/Scripts/python -m scripts.ingest_corpus

    # re-embed everything regardless of the manifest
    .venv/Scripts/python -m scripts.ingest_corpus --force

    # after deleting files from corpus/: wipe and rebuild the collection
    .venv/Scripts/python -m scripts.ingest_corpus --rebuild

Files are organised by topic: `corpus/transformers/*.pdf` all index under the
topic "transformers". Supported: .pdf, .md, .txt.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Windows + async psycopg (see app.main).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.config import Settings
from app.rag.corpus import CorpusIngestor
from app.rag.factory import build_embeddings, build_vector_store
from app.rag.indexer import Indexer
from app.rag.knowledge import seed_knowledge_base
from app.rag.schemas import GLOBAL_OWNER

DEFAULT_CORPUS_DIR = Path("corpus")


async def run(corpus_dir: Path, *, force: bool, rebuild: bool) -> int:
    settings = Settings()
    if settings.embedding_provider != "fake" and not settings.openai_api_key:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is needed to embed the corpus (env or .env).")

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s %(message)s")
    embeddings = build_embeddings(settings)
    store = build_vector_store(settings)
    indexer = Indexer(embeddings=embeddings, store=store)

    create_schema = getattr(store, "create_schema", None)
    if create_schema is not None:
        await create_schema()

    if rebuild:
        removed = await store.delete_owner(collection="knowledge", owner_id=GLOBAL_OWNER)
        print(f"rebuild: removed {removed} existing knowledge chunks")
        await seed_knowledge_base(indexer)
        force = True

    if not corpus_dir.exists():
        corpus_dir.mkdir(parents=True)
        print(f"created {corpus_dir}/ - drop PDFs, markdown or text files in it "
              "(subfolder name = topic) and run this again")
        return 0

    ingestor = CorpusIngestor(indexer=indexer, corpus_dir=corpus_dir)
    report = await ingestor.ingest(force=force)

    print()
    print(f"indexed: {len(report.indexed_files)} file(s), {report.chunks_indexed} chunks")
    for name in report.indexed_files:
        print(f"  + {name}")
    if report.skipped_files:
        print(f"unchanged (skipped): {len(report.skipped_files)}")
    for name, error in report.failed_files:
        print(f"  ! FAILED {name}: {error[:120]}")

    total = await store.count(collection="knowledge", owner_id=GLOBAL_OWNER)
    print(f"knowledge collection now holds {total} chunks "
          f"(store={settings.resolved_vector_store})")
    if settings.resolved_vector_store == "memory":
        print("WARNING: in-memory store - this index is lost when the process "
              "exits. Set DATABASE_URL to make it durable.")

    dispose = getattr(store, "dispose", None)
    if dispose is not None:
        await dispose()
    return 1 if report.failed_files else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--force", action="store_true", help="ignore the manifest")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="wipe the knowledge collection, re-seed the curated notes, re-index",
    )
    arguments = parser.parse_args()
    sys.exit(asyncio.run(run(arguments.path, force=arguments.force, rebuild=arguments.rebuild)))


if __name__ == "__main__":
    main()
