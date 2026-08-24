"""Corpus pipeline: loading, splitting, topics, manifest skipping, rebuilds."""

from __future__ import annotations

from pathlib import Path

from app.rag.corpus import (
    CorpusIngestor,
    discover_files,
    load_documents,
    split_documents,
    topic_for,
)
from app.rag.retriever import Retriever
from tests.conftest import build_retrieval_stack

def minimal_pdf(text: str = "Attention scaling") -> bytes:
    """A valid single-page PDF with real xref offsets, built at runtime."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


MINIMAL_PDF = minimal_pdf()


def make_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "transformers").mkdir(parents=True)
    (corpus / "transformers" / "notes.md").write_text(
        "# Attention\n\nSelf-attention compares every token against every other "
        "token, which is why cost grows quadratically with sequence length.",
        encoding="utf-8",
    )
    (corpus / "ml-basics.txt").write_text(
        "Bias is error from wrong assumptions. Variance is sensitivity to the "
        "particular training sample.",
        encoding="utf-8",
    )
    (corpus / "ignore.docx").write_text("unsupported", encoding="utf-8")
    (corpus / "README.md").write_text("about this folder", encoding="utf-8")
    return corpus


class TestLoading:
    def test_only_supported_files_are_discovered(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path)
        names = [path.name for path in discover_files(corpus)]

        assert names == ["ml-basics.txt", "notes.md"]

    def test_subfolder_names_become_topics(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path)

        assert topic_for(corpus, corpus / "transformers" / "notes.md") == "transformers"
        assert topic_for(corpus, corpus / "ml-basics.txt") == "ml basics"

    def test_pdf_pages_load_with_page_numbers(self, tmp_path: Path) -> None:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(MINIMAL_PDF)

        documents = load_documents(pdf)

        assert len(documents) == 1
        assert "Attention scaling" in documents[0].page_content
        assert documents[0].metadata["page"] == 1

    def test_long_documents_are_split_with_overlap(self, tmp_path: Path) -> None:
        from langchain_core.documents import Document

        long_text = " ".join(f"Sentence number {i} about retrieval." for i in range(200))
        pieces = split_documents([Document(page_content=long_text, metadata={})])

        assert len(pieces) > 1
        assert all(len(piece.page_content) <= 1_300 for piece in pieces)


class TestIngestion:
    async def test_corpus_content_becomes_retrievable_knowledge(
        self, tmp_path: Path
    ) -> None:
        corpus = make_corpus(tmp_path)
        indexer, retriever = build_retrieval_stack(min_score=0.05)

        report = await CorpusIngestor(indexer=indexer, corpus_dir=corpus).ingest()

        assert len(report.indexed_files) == 2
        assert report.failed_files == []

        hits = await retriever.search(
            "quadratic cost sequence length attention", collection="knowledge"
        )
        assert hits
        assert hits[0].chunk.topic == "transformers"
        assert "notes.md" in hits[0].chunk.source

    async def test_unchanged_files_are_skipped_on_the_next_run(
        self, tmp_path: Path
    ) -> None:
        corpus = make_corpus(tmp_path)
        indexer, _ = build_retrieval_stack()
        ingestor = CorpusIngestor(indexer=indexer, corpus_dir=corpus)

        first = await ingestor.ingest()
        second = await ingestor.ingest()

        assert len(first.indexed_files) == 2
        assert second.indexed_files == []
        assert len(second.skipped_files) == 2

    async def test_a_changed_file_is_reindexed(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path)
        indexer, _ = build_retrieval_stack()
        ingestor = CorpusIngestor(indexer=indexer, corpus_dir=corpus)
        await ingestor.ingest()

        (corpus / "ml-basics.txt").write_text("Completely new content.", encoding="utf-8")
        report = await ingestor.ingest()

        assert report.indexed_files == ["ml-basics.txt"]
        assert report.skipped_files == ["transformers\\notes.md"] or report.skipped_files == [
            "transformers/notes.md"
        ]

    async def test_force_reindexes_everything(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path)
        indexer, _ = build_retrieval_stack()
        ingestor = CorpusIngestor(indexer=indexer, corpus_dir=corpus)
        await ingestor.ingest()

        report = await ingestor.ingest(force=True)

        assert len(report.indexed_files) == 2

    async def test_one_broken_file_does_not_sink_the_rest(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path)
        (corpus / "broken.pdf").write_bytes(b"not a pdf at all")
        indexer, _ = build_retrieval_stack()

        report = await CorpusIngestor(indexer=indexer, corpus_dir=corpus).ingest()

        assert len(report.indexed_files) == 2
        assert len(report.failed_files) == 1
        assert report.failed_files[0][0] == "broken.pdf"

    async def test_reindexing_does_not_duplicate_chunks(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path)
        indexer, retriever = build_retrieval_stack()
        ingestor = CorpusIngestor(indexer=indexer, corpus_dir=corpus)

        await ingestor.ingest()
        first_count = await retriever._store.count(  # noqa: SLF001 - test introspection
            collection="knowledge", owner_id="global"
        )
        await ingestor.ingest(force=True)
        second_count = await retriever._store.count(  # noqa: SLF001
            collection="knowledge", owner_id="global"
        )

        assert first_count == second_count


class TestSanitisation:
    def test_control_characters_are_scrubbed_from_text_files(self, tmp_path: Path) -> None:
        # Postgres TEXT rejects NUL outright; PDF extraction produces them too.
        from app.rag.corpus import clean_text

        assert clean_text("a\x00b\x01c\nd\te") == "abc\nd\te"

        dirty = tmp_path / "dirty.txt"
        dirty.write_bytes("clean start\x00 clean end".encode("utf-8"))
        documents = load_documents(dirty)

        assert "\x00" not in documents[0].page_content
        assert "clean start clean end" in documents[0].page_content

    async def test_duplicate_chunks_within_one_file_do_not_break_ingestion(
        self, tmp_path: Path
    ) -> None:
        # Repeated page headers extract identically -> identical content ids.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "repeats.txt").write_text(
            "Same header line.\n\nSame header line.", encoding="utf-8"
        )
        indexer, retriever = build_retrieval_stack()

        report = await CorpusIngestor(indexer=indexer, corpus_dir=corpus).ingest()

        assert report.failed_files == []
        assert report.indexed_files == ["repeats.txt"]
