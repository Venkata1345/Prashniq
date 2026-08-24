"""The knowledge corpus pipeline: drop files in a folder, index them.

    corpus/
    ├── transformers/attention-book.pdf     -> topic "transformers"
    ├── rag/retrieval-notes.md              -> topic "rag"
    └── ml-basics.txt                       -> topic "ml-basics"

Loading is LangChain Documents end to end: PDFs via pypdf, text/markdown read
directly, split with RecursiveCharacterTextSplitter, then indexed into the
shared `knowledge` collection alongside the curated notes.

A manifest of file hashes makes re-runs cheap: unchanged files are skipped, so
adding one book re-embeds one book. Chunk ids are content-addressed, so even a
forced re-run only overwrites rows, never duplicates them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.indexer import Indexer, stable_chunk_id
from app.rag.schemas import GLOBAL_OWNER, DocumentChunk

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".text"}
MANIFEST_NAME = ".manifest.json"

# Character-based splitting; ~1200 chars is a solid paragraph-or-two of a
# textbook, small enough to be a precise retrieval unit.
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 150

# PDF extraction emits NUL bytes and other C0 control characters from broken
# font maps; Postgres TEXT rejects NUL outright. Newlines and tabs survive.
_CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    return _CONTROL_CHARS.sub("", text)


@dataclass
class IngestReport:
    indexed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    chunks_indexed: int = 0


def discover_files(corpus_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in corpus_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith(".")
        # A README describes the shelf; it is not on the shelf.
        and path.stem.lower() != "readme"
    )


def topic_for(corpus_dir: Path, file_path: Path) -> str:
    """First-level folder name if the file lives in one, else the file stem --
    so `corpus/transformers/anything.pdf` all lands under one topic."""
    relative = file_path.relative_to(corpus_dir)
    if len(relative.parts) > 1:
        return relative.parts[0].replace("_", " ").replace("-", " ")
    return file_path.stem.replace("_", " ").replace("-", " ")


def load_documents(file_path: Path) -> list[Document]:
    if file_path.suffix.lower() == ".pdf":
        return _load_pdf(file_path)
    text = clean_text(file_path.read_text(encoding="utf-8", errors="replace"))
    return [Document(page_content=text, metadata={"source": file_path.name})]


def _load_pdf(file_path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    documents = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": number},
                )
            )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


class CorpusIngestor:
    def __init__(self, *, indexer: Indexer, corpus_dir: Path) -> None:
        self._indexer = indexer
        self._corpus_dir = corpus_dir
        self._manifest_path = corpus_dir / MANIFEST_NAME

    async def ingest(self, *, force: bool = False) -> IngestReport:
        report = IngestReport()
        manifest = {} if force else self._read_manifest()

        for file_path in discover_files(self._corpus_dir):
            relative = str(file_path.relative_to(self._corpus_dir))
            digest = _file_digest(file_path)
            if manifest.get(relative) == digest:
                report.skipped_files.append(relative)
                continue

            try:
                written = await self._ingest_file(file_path)
            except Exception as exc:
                # One unreadable book must not sink the rest of the shelf.
                logger.error("corpus_file_failed file=%s error=%s", relative, exc)
                report.failed_files.append((relative, str(exc)))
                continue

            manifest[relative] = digest
            report.indexed_files.append(relative)
            report.chunks_indexed += written
            logger.info("corpus_file_indexed file=%s chunks=%d", relative, written)

        self._write_manifest(manifest)
        return report

    async def _ingest_file(self, file_path: Path) -> int:
        topic = topic_for(self._corpus_dir, file_path)
        relative = str(file_path.relative_to(self._corpus_dir))
        chunks = split_documents(load_documents(file_path))

        return await self._indexer.index_chunks(
            [
                DocumentChunk(
                    chunk_id=stable_chunk_id(f"{relative}:{piece.page_content}"),
                    collection="knowledge",
                    owner_id=GLOBAL_OWNER,
                    text=piece.page_content,
                    topic=topic,
                    source=_source_label(relative, piece),
                )
                for piece in chunks
                if piece.page_content.strip()
            ]
        )

    def _read_manifest(self) -> dict[str, str]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("corpus_manifest_unreadable - re-indexing everything")
            return {}

    def _write_manifest(self, manifest: dict[str, str]) -> None:
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )


def _source_label(relative: str, piece: Document) -> str:
    page = piece.metadata.get("page")
    return f"{relative} (p. {page})" if page else relative


def _file_digest(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()
