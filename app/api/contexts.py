"""Candidate-context routes: resume and job-description ingestion."""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.api.context_schemas import ContextResponse, CreateContextRequest
from app.context.service import CandidateContextService
from app.rag.corpus import clean_text

router = APIRouter(prefix="/candidate-contexts", tags=["candidate context"])

# A resume or JD is pages, not a book; anything bigger is a wrong file.
_MAX_PDF_BYTES = 10 * 1024 * 1024


class ExtractedText(BaseModel):
    text: str
    pages: int


@router.post("/extract-pdf", response_model=ExtractedText)
async def extract_pdf(file: UploadFile) -> ExtractedText:
    """Extract plain text from an uploaded PDF (resume or job description).

    The text is returned to the client for review, not stored: the page fills
    the textarea with it and context creation proceeds through the normal
    text endpoint.
    """
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    payload = await file.read(_MAX_PDF_BYTES + 1)
    if len(payload) > _MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF larger than 10 MB")
    if not payload.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="not a PDF file")

    try:
        reader = PdfReader(io.BytesIO(payload))
        pages = [clean_text(page.extract_text() or "").strip() for page in reader.pages]
    except PyPdfError:
        raise HTTPException(status_code=422, detail="could not read this PDF")

    text = "\n\n".join(page for page in pages if page).strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="no extractable text in this PDF (is it a scanned image?)",
        )
    return ExtractedText(text=text, pages=len(pages))


def get_context_service(request: Request) -> CandidateContextService:
    return request.app.state.context_service


def _request_id(request: Request) -> str | None:
    return request.headers.get("x-request-id")


@router.post("", response_model=ContextResponse, status_code=status.HTTP_201_CREATED)
async def create_context(
    payload: CreateContextRequest,
    request: Request,
    service: CandidateContextService = Depends(get_context_service),
) -> ContextResponse:
    context = await service.create(
        resume_text=payload.resume_text,
        job_description_text=payload.job_description_text,
        candidate_id=payload.candidate_id,
        request_id=_request_id(request),
    )
    return ContextResponse.from_context(context)


@router.get("/{context_id}", response_model=ContextResponse)
async def get_context(
    context_id: str,
    service: CandidateContextService = Depends(get_context_service),
) -> ContextResponse:
    return ContextResponse.from_context(await service.get(context_id))
