"""The interview orchestrator: a facade over the LangGraph interview graph.

The public API is unchanged (create / start / submit_answer / state / complete
/ report); internally each interview is a checkpointed graph thread keyed by
interview id. The repository remains the queryable system of record; the
checkpointer holds the graph's execution position so an interview can resume
exactly where it stopped.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.context.blueprint import build_blueprint
from app.context.schemas import CandidateContext
from app.context.service import CandidateContextService
from app.interview import state as state_ops
from app.interview.evaluator import Evaluator
from app.interview.graph import build_interview_graph
from app.interview.grounding import GroundingService
from app.interview.interviewer import Interviewer
from app.interview.modes import DEFAULT_MODE, get_mode
from app.interview.planner import PlannedAction
from app.interview.repository import InterviewRepository
from app.interview.schemas import (
    AnswerEvaluation,
    InterviewReport,
    InterviewStatus,
    Question,
)
from app.interview.scoring import build_report
from app.interview.state import InterviewState, InvalidInterviewState
from app.profile.service import SkillProfileService

logger = logging.getLogger(__name__)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TurnResult:
    """Everything the loop produced for one submitted answer.

    The evaluation is included for the engine and the report -- the candidate
    API deliberately does not surface it mid-interview.
    """

    evaluation: AnswerEvaluation
    evaluation_degraded: bool
    plan: PlannedAction
    next_question: Question | None
    state: InterviewState


class InterviewOrchestrator:
    def __init__(
        self,
        *,
        evaluator: Evaluator,
        interviewer: Interviewer,
        repository: InterviewRepository,
        clock: Clock | None = None,
        id_factory: Callable[[], str] | None = None,
        context_service: CandidateContextService | None = None,
        grounding: GroundingService | None = None,
        profile_service: SkillProfileService | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        checkpointer_factory: Callable[[], BaseCheckpointSaver] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or SystemClock()
        self._new_id = id_factory or (lambda: uuid.uuid4().hex)
        self._context_service = context_service
        # Without a retriever this still supplies blueprint-planned evidence.
        self._grounding = grounding or GroundingService()
        self._profile_service = profile_service
        self._evaluator = evaluator
        self._interviewer = interviewer
        self._checkpointer = checkpointer
        self._checkpointer_factory = checkpointer_factory
        # Compiled on first use: AsyncPostgresSaver can only be constructed
        # inside a running event loop, and create_app runs before one exists.
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            checkpointer = self._checkpointer
            if checkpointer is None and self._checkpointer_factory is not None:
                checkpointer = self._checkpointer_factory()
            self._graph = build_interview_graph(
                evaluator=self._evaluator,
                interviewer=self._interviewer,
                repository=self._repository,
                grounding=self._grounding,
                clock=self._clock,
                id_factory=self._new_id,
                checkpointer=checkpointer or InMemorySaver(),
                on_completed=self._record_profile,
            )
        return self._graph

    async def create_interview(
        self,
        *,
        interview_type: str = DEFAULT_MODE,
        candidate_id: str | None = None,
        context_id: str | None = None,
    ) -> InterviewState:
        """Create an interview and plan what it will cover.

        The blueprint is built once, up front: with a candidate context it is
        driven by the target role and the candidate's own claims, without one it
        is simply the mode's standing topics.
        """
        mode = get_mode(interview_type)
        context = await self._load_context(context_id)
        resolved_candidate = candidate_id or (context.candidate_id if context else None)
        blueprint = await build_blueprint(
            mode,
            context,
            self._grounding.evidence_lookup(context),
            focus_skills=await self._focus_skills(resolved_candidate),
        )

        state = InterviewState(
            interview_id=self._new_id(),
            candidate_id=resolved_candidate,
            interview_type=mode.key,
            difficulty=mode.starting_difficulty,
            remaining_topics=blueprint.topic_keys(),
            blueprint=blueprint,
            context_id=context.context_id if context else None,
            resume=context.resume if context else None,
            time_limit_seconds=mode.time_limit_seconds,
            created_at=self._clock.now(),
        )
        await self._repository.add(state)
        logger.info(
            "interview_created interview_id=%s type=%s context_id=%s topics=%d",
            state.interview_id,
            mode.key,
            state.context_id,
            len(blueprint.topics),
        )
        return state

    async def start(self, interview_id: str, *, request_id: str | None = None) -> Question:
        """Run the graph to its first pause. Idempotent while a question is
        outstanding, so a reconnecting client gets the same question back."""
        async with self._repository.lock(interview_id):
            state = await self._repository.get(interview_id)
            if state.status is InterviewStatus.COMPLETED:
                raise InvalidInterviewState("interview is already completed")
            if state.pending_question is not None:
                return state.pending_question

            if state.status is InterviewStatus.CREATED:
                state = state_ops.start(state, self._clock.now())
                await self._repository.save(state)

            await self._get_graph().ainvoke(
                {"interview": state, "request_id": request_id},
                self._thread(interview_id),
            )
            state = await self._repository.get(interview_id)
            if state.pending_question is None:  # pragma: no cover - graph shape
                raise InvalidInterviewState("the interview failed to produce a question")
            return state.pending_question

    async def submit_answer(
        self, interview_id: str, answer: str, *, request_id: str | None = None
    ) -> TurnResult:
        """Resume the paused graph with the candidate's answer. The graph runs
        evaluate -> plan -> (next question | finish) and pauses again."""
        async with self._repository.lock(interview_id):
            state = await self._repository.get(interview_id)
            if state.status is not InterviewStatus.IN_PROGRESS:
                raise InvalidInterviewState(
                    f"cannot answer an interview in status '{state.status.value}'"
                )
            if state.pending_question is None:
                raise InvalidInterviewState("no question is awaiting an answer")

            result = await self._get_graph().ainvoke(
                Command(update={"request_id": request_id}, resume=answer),
                self._thread(interview_id),
            )

            state = result["interview"]
            plan: PlannedAction = result["last_plan"]
            last_turn = state.turns[-1]
            return TurnResult(
                evaluation=last_turn.evaluation,
                evaluation_degraded=last_turn.evaluation_degraded,
                plan=plan,
                next_question=state.pending_question,
                state=state,
            )

    async def get_state(self, interview_id: str) -> InterviewState:
        return await self._repository.get(interview_id)

    async def complete(self, interview_id: str) -> InterviewState:
        """End early. Goes straight to the repository: the graph thread simply
        never resumes, which costs nothing."""
        async with self._repository.lock(interview_id):
            before = await self._repository.get(interview_id)
            state = state_ops.complete(before, self._clock.now())
            await self._repository.save(state)
            # Record only on the transition; re-completing changes nothing.
            if before.status is not InterviewStatus.COMPLETED:
                await self._record_profile(state)
            return state

    async def get_report(self, interview_id: str) -> InterviewReport:
        state = await self._repository.get(interview_id)
        if state.status is not InterviewStatus.COMPLETED:
            raise InvalidInterviewState("report is only available once the interview ends")
        return build_report(state, get_mode(state.interview_type), self._clock.now())

    def _thread(self, interview_id: str) -> dict:
        return {"configurable": {"thread_id": interview_id}}

    async def _focus_skills(self, candidate_id: str | None) -> list[str]:
        """The candidate's weak skills, so the blueprint can practise them.

        Best-effort: a profile outage costs personalisation, not the interview.
        """
        if self._profile_service is None or candidate_id is None:
            return []
        try:
            return await self._profile_service.weak_skills(candidate_id)
        except Exception:
            logger.exception("skill_profile_unavailable candidate_id=%s", candidate_id)
            return []

    async def _record_profile(self, state: InterviewState) -> None:
        """Fold a just-completed interview into the candidate's skill history.

        Failure is logged, never raised: the observations derive entirely from
        the persisted interview state, so they can be rebuilt later.
        """
        if self._profile_service is None:
            return
        try:
            await self._profile_service.record_interview(state)
        except Exception:
            logger.exception(
                "skill_profile_record_failed interview_id=%s", state.interview_id
            )

    async def _load_context(self, context_id: str | None) -> CandidateContext | None:
        if context_id is None:
            return None
        if self._context_service is None:
            raise InvalidInterviewState(
                "candidate contexts are not enabled on this deployment"
            )
        return await self._context_service.get(context_id)
