"""pgvector-backed vector store.

One table, partitioned logically by (collection, owner_id). Cosine distance is
used for search, so `score` is returned as `1 - distance` to match the
protocol's "higher is closer" contract.

Schema creation is idempotent and runs at startup; there is no migration tool
yet (see the debt list in the README).
"""

from __future__ import annotations

import logging
from typing import Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.rag.schemas import Collection, EmbeddedChunk, RetrievedChunk, VectorStoreError

logger = logging.getLogger(__name__)

metadata = MetaData()


DEFAULT_TABLE_NAME = "rag_chunks"


def build_table(dimensions: int, name: str = DEFAULT_TABLE_NAME) -> Table:
    """The embedding column is fixed-width, so the table is defined against the
    configured model's dimensionality."""
    return Table(
        name,
        metadata,
        Column("collection", String(32), primary_key=True),
        Column("owner_id", String(64), primary_key=True),
        Column("chunk_id", String(128), primary_key=True),
        Column("text", Text, nullable=False),
        Column("topic", String(256), nullable=True),
        Column("source", String(256), nullable=True),
        Column("embedding", Vector(dimensions), nullable=False),
        Column("dimensions", Integer, nullable=False),
        Index(
            f"ix_{name}_collection_owner",
            "collection",
            "owner_id",
        ),
        extend_existing=True,
    )


class PgVectorStore:
    def __init__(
        self,
        *,
        engine: AsyncEngine,
        dimensions: int,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        self._engine = engine
        self._dimensions = dimensions
        self._table = build_table(dimensions, table_name)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        dimensions: int,
        table_name: str = DEFAULT_TABLE_NAME,
        echo: bool = False,
    ) -> "PgVectorStore":
        return cls(
            engine=create_async_engine(url, echo=echo),
            dimensions=dimensions,
            table_name=table_name,
        )

    async def create_schema(self) -> None:
        """Idempotent: safe to call on every boot."""
        try:
            async with self._engine.begin() as connection:
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await connection.run_sync(metadata.create_all, tables=[self._table])
                await self._assert_dimensions(connection)
                # ANN index: exact scans are fine for hundreds of chunks and
                # sloppy for tens of thousands. HNSW keeps cosine search in
                # single-digit ms; inserts maintain it incrementally.
                await connection.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS ix_{self._table.name}_embedding_hnsw "
                        f"ON {self._table.name} "
                        "USING hnsw (embedding vector_cosine_ops)"
                    )
                )
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"could not prepare the vector schema: {exc}") from exc
        logger.info(
            "vector_schema_ready table=%s dimensions=%d",
            self._table.name,
            self._dimensions,
        )

    async def _assert_dimensions(self, connection) -> None:
        """`CREATE TABLE IF NOT EXISTS` silently accepts a table built for a
        different embedding width; every write then fails at runtime with an
        opaque driver error. Fail loudly here instead."""
        result = await connection.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) AS type "
                "FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "WHERE c.relname = :table AND a.attname = 'embedding'"
            ),
            {"table": self._table.name},
        )
        row = result.first()
        if row is None:
            return

        existing = _vector_width(row[0])
        if existing is not None and existing != self._dimensions:
            raise VectorStoreError(
                f"table '{self._table.name}' stores {existing}-dimensional vectors but "
                f"embedding_dimensions is {self._dimensions}. Changing embedding model "
                "or width requires re-indexing: drop the table (or migrate it) and "
                "restart."
            )

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> int:
        if not chunks:
            return 0

        # ON CONFLICT cannot update the same row twice in one statement, and
        # content-addressed ids make within-batch duplicates legitimate (a
        # repeated page header extracts identically). Last one wins.
        unique: dict[tuple[str, str, str], EmbeddedChunk] = {
            (item.chunk.collection, item.chunk.owner_id, item.chunk.chunk_id): item
            for item in chunks
        }
        chunks = list(unique.values())

        rows = [
            {
                "collection": item.chunk.collection,
                "owner_id": item.chunk.owner_id,
                "chunk_id": item.chunk.chunk_id,
                "text": item.chunk.text,
                "topic": item.chunk.topic,
                "source": item.chunk.source,
                "embedding": list(item.embedding),
                "dimensions": len(item.embedding),
            }
            for item in chunks
        ]
        statement = insert(self._table).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["collection", "owner_id", "chunk_id"],
            set_={
                "text": statement.excluded.text,
                "topic": statement.excluded.topic,
                "source": statement.excluded.source,
                "embedding": statement.excluded.embedding,
                "dimensions": statement.excluded.dimensions,
            },
        )

        try:
            async with self._engine.begin() as connection:
                await connection.execute(statement)
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"upsert failed: {exc}") from exc
        return len(rows)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        collection: Collection,
        owner_id: str,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        distance = self._table.c.embedding.cosine_distance(list(embedding))
        statement = (
            select(
                self._table.c.chunk_id,
                self._table.c.collection,
                self._table.c.owner_id,
                self._table.c.text,
                self._table.c.topic,
                self._table.c.source,
                distance.label("distance"),
            )
            .where(
                self._table.c.collection == collection,
                self._table.c.owner_id == owner_id,
            )
            .order_by(distance, self._table.c.chunk_id)
            .limit(limit)
        )

        try:
            async with self._engine.connect() as connection:
                rows = (await connection.execute(statement)).mappings().all()
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"search failed: {exc}") from exc

        hits = [_as_hit(row) for row in rows]
        return [hit for hit in hits if hit.score >= min_score]

    async def delete_owner(self, *, collection: Collection, owner_id: str) -> int:
        statement = delete(self._table).where(
            self._table.c.collection == collection, self._table.c.owner_id == owner_id
        )
        try:
            async with self._engine.begin() as connection:
                result = await connection.execute(statement)
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"delete failed: {exc}") from exc
        return result.rowcount or 0

    async def count(self, *, collection: Collection, owner_id: str) -> int:
        statement = select(func.count()).where(
            self._table.c.collection == collection, self._table.c.owner_id == owner_id
        )
        try:
            async with self._engine.connect() as connection:
                return int((await connection.execute(statement)).scalar_one())
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"count failed: {exc}") from exc


def _vector_width(type_name: str) -> int | None:
    """'vector(1536)' -> 1536."""
    if "(" not in type_name:
        return None
    try:
        return int(type_name.split("(", 1)[1].rstrip(")"))
    except ValueError:
        return None


def _as_hit(row) -> RetrievedChunk:
    from app.rag.schemas import DocumentChunk

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=row["chunk_id"],
            collection=row["collection"],
            owner_id=row["owner_id"],
            text=row["text"],
            topic=row["topic"],
            source=row["source"],
        ),
        # pgvector returns cosine *distance*; the protocol speaks similarity.
        score=1.0 - float(row["distance"]),
    )
