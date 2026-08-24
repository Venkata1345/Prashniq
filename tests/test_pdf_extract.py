"""The resume PDF upload endpoint: extraction, validation, limits."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app(
        Settings(
            llm_provider="fake",
            embedding_provider="fake",
            vector_store="memory",
            database_url=None,
        )
    )
    return TestClient(app)


def tiny_pdf(text: str) -> bytes:
    """A minimal valid one-page PDF containing `text`, offsets computed."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


class TestExtractPdf:
    def test_text_is_extracted(self, client: TestClient) -> None:
        pdf = tiny_pdf("Built a RAG system with FAISS")
        response = client.post(
            "/candidate-contexts/extract-pdf",
            files={"file": ("resume.pdf", pdf, "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "Built a RAG system with FAISS" in body["text"]
        assert body["pages"] == 1

    def test_non_pdf_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/candidate-contexts/extract-pdf",
            files={"file": ("resume.txt", b"just some text", "text/plain")},
        )
        assert response.status_code == 422
        assert "not a PDF" in response.json()["detail"]

    def test_unreadable_pdf_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/candidate-contexts/extract-pdf",
            files={"file": ("bad.pdf", b"%PDF-1.4 garbage after the magic", "application/pdf")},
        )
        assert response.status_code == 422

    def test_extracted_text_flows_into_context_creation(self, client: TestClient) -> None:
        """The exact sequence the page drives: extract, then create a context."""
        with client:
            pdf = tiny_pdf("Shipped ML pipelines on AWS")
            extracted = client.post(
                "/candidate-contexts/extract-pdf",
                files={"file": ("resume.pdf", pdf, "application/pdf")},
            ).json()

            response = client.post(
                "/candidate-contexts",
                json={
                    "candidate_id": "pdf-cand",
                    "resume_text": extracted["text"],
                    "job_description_text": None,
                },
            )
            assert response.status_code == 201
