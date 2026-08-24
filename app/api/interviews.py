"""HTTP routes.

Thin by design: translate DTOs, delegate to the orchestrator, map domain errors
onto status codes. No interview logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.api.context_schemas import BlueprintResponse
from app.api.schemas import (
    AnswerAcceptedResponse,
    CreateInterviewRequest,
    InterviewResponse,
    QuestionResponse,
    SubmitAnswerRequest,
)
from app.context.blueprint import BlueprintError
from app.context.ingestion import DocumentIngestionError
from app.context.repository import CandidateContextNotFound
from app.interview.modes import UnknownInterviewMode, available_modes
from app.interview.orchestrator import InterviewOrchestrator
from app.interview.repository import InterviewNotFound
from app.interview.schemas import InterviewReport, InterviewStatus
from app.interview.state import InvalidInterviewState
from app.llm.schemas import LLMError

router = APIRouter(prefix="/interviews", tags=["interviews"])
catalog_router = APIRouter(tags=["catalog"])


def get_orchestrator(request: Request) -> InterviewOrchestrator:
    return request.app.state.orchestrator


def _request_id(request: Request) -> str | None:
    return request.headers.get("x-request-id")


@router.post("", response_model=InterviewResponse, status_code=status.HTTP_201_CREATED)
async def create_interview(
    payload: CreateInterviewRequest,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> InterviewResponse:
    try:
        state = await orchestrator.create_interview(
            interview_type=payload.interview_type,
            candidate_id=payload.candidate_id,
            context_id=payload.context_id,
        )
    except (UnknownInterviewMode, BlueprintError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return InterviewResponse.from_state(state)


@router.post("/{interview_id}/start", response_model=QuestionResponse)
async def start_interview(
    interview_id: str,
    request: Request,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> QuestionResponse:
    question = await orchestrator.start(interview_id, request_id=_request_id(request))
    return QuestionResponse(**question.model_dump())


@router.post("/{interview_id}/answers", response_model=AnswerAcceptedResponse)
async def submit_answer(
    interview_id: str,
    payload: SubmitAnswerRequest,
    request: Request,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> AnswerAcceptedResponse:
    result = await orchestrator.submit_answer(
        interview_id, payload.answer, request_id=_request_id(request)
    )
    return AnswerAcceptedResponse(
        interview_id=interview_id,
        status=result.state.status,
        next_question=(
            QuestionResponse(**result.next_question.model_dump())
            if result.next_question
            else None
        ),
        interview_complete=result.state.status is InterviewStatus.COMPLETED,
    )


@router.get("/{interview_id}", response_model=InterviewResponse)
async def get_interview(
    interview_id: str,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> InterviewResponse:
    return InterviewResponse.from_state(await orchestrator.get_state(interview_id))


@router.get("/{interview_id}/blueprint", response_model=BlueprintResponse)
async def get_blueprint(
    interview_id: str,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> BlueprintResponse:
    """What this interview plans to cover, in order, and why."""
    state = await orchestrator.get_state(interview_id)
    if state.blueprint is None:
        raise HTTPException(404, "this interview has no blueprint")
    return BlueprintResponse.from_blueprint(state.blueprint)


@router.post("/{interview_id}/complete", response_model=InterviewResponse)
async def complete_interview(
    interview_id: str,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> InterviewResponse:
    return InterviewResponse.from_state(await orchestrator.complete(interview_id))


@router.get("/{interview_id}/report", response_model=InterviewReport)
async def get_report(
    interview_id: str,
    orchestrator: InterviewOrchestrator = Depends(get_orchestrator),
) -> InterviewReport:
    return await orchestrator.get_report(interview_id)


@catalog_router.get("/interview-types")
async def list_interview_types() -> list[dict[str, object]]:
    return [
        {
            "key": mode.key,
            "display_name": mode.display_name,
            "topics": list(mode.topics),
            "dimensions": list(mode.dimensions),
            "max_questions": mode.max_questions,
        }
        for mode in available_modes()
    ]


def install_exception_handlers(app: FastAPI) -> None:
    """Domain errors to HTTP status codes, in one place."""

    @app.exception_handler(InterviewNotFound)
    async def _not_found(_: Request, exc: InterviewNotFound) -> JSONResponse:
        return JSONResponse(
            {"detail": f"interview {exc.args[0]} not found"}, status_code=404
        )

    @app.exception_handler(CandidateContextNotFound)
    async def _context_not_found(_: Request, exc: CandidateContextNotFound) -> JSONResponse:
        return JSONResponse(
            {"detail": f"candidate context {exc.args[0]} not found"}, status_code=404
        )

    @app.exception_handler(DocumentIngestionError)
    async def _bad_document(_: Request, exc: DocumentIngestionError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(InvalidInterviewState)
    async def _conflict(_: Request, exc: InvalidInterviewState) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(LLMError)
    async def _upstream(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(
            {"detail": "the interview service is temporarily unavailable"},
            status_code=503,
        )
