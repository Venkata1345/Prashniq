"""The interview loop as a LangGraph state machine.

    ask -> await_answer -> evaluate -> (ask | finish)

LangGraph supplies the machinery: every step is checkpointed (Postgres in
production), the graph pauses at `await_answer` via interrupt() until the
candidate replies, and a crashed process resumes from the last checkpoint.

The intelligence/control split is unchanged: `evaluate` and `ask` call the LLM
through the gateway, while planning, state transitions, budgets and termination
are the same pure functions as before, now running inside graph nodes. The
graph never lets the model mutate control state.
"""

from __future__ import annotations

import logging
from typing import Callable, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.interview import state as state_ops
from app.interview.evaluator import Evaluator
from app.interview.grounding import GroundingService
from app.interview.interviewer import Interviewer
from app.interview.modes import get_mode
from app.interview.planner import PlannedAction, choose_next_action, plan_opening
from app.interview.repository import InterviewRepository
from app.interview.schemas import Question
from app.interview.state import InterviewState

logger = logging.getLogger(__name__)


class InterviewGraphState(TypedDict, total=False):
    """The graph's channels. `interview` is the domain state, replaced whole by
    each node -- the same pure-transition style as before, now checkpointed."""

    interview: InterviewState
    answer: str
    last_plan: PlannedAction | None
    request_id: str | None


def build_interview_graph(
    *,
    evaluator: Evaluator,
    interviewer: Interviewer,
    repository: InterviewRepository,
    grounding: GroundingService,
    clock,
    id_factory: Callable[[], str],
    checkpointer: BaseCheckpointSaver,
    on_completed: Callable[[InterviewState], "object"] | None = None,
):
    """Compile the interview graph. Dependencies are closed over; the graph
    itself holds no globals."""

    async def ask(graph_state: InterviewGraphState) -> InterviewGraphState:
        state = graph_state["interview"]
        mode = get_mode(state.interview_type)
        plan = graph_state.get("last_plan") or plan_opening(state, mode)

        state = state_ops.apply_plan(
            state, difficulty=plan.difficulty, follow_up_depth=plan.follow_up_depth
        )
        draft = await interviewer.generate_question(
            plan=plan,
            state=state,
            mode=mode,
            grounding=await grounding.for_question(plan=plan, state=state, mode=mode),
            request_id=graph_state.get("request_id"),
        )
        question = Question(
            id=id_factory(),
            index=len(state.questions_asked) + 1,
            text=draft.question.strip(),
            topic=(plan.topic or draft.topic).strip(),
            difficulty=state.difficulty,
            action=plan.action,
            asked_at=clock.now(),
        )
        state = state_ops.record_question(state, question)
        await repository.save(state)
        return {"interview": state}

    async def await_answer(graph_state: InterviewGraphState) -> InterviewGraphState:
        # Pauses the graph here; the checkpointer persists everything. The
        # value handed to Command(resume=...) becomes the return value.
        question = graph_state["interview"].pending_question
        answer = interrupt(
            {
                "question": question.model_dump(mode="json") if question else None,
                "interview_id": graph_state["interview"].interview_id,
            }
        )
        return {"answer": str(answer)}

    async def evaluate(graph_state: InterviewGraphState) -> InterviewGraphState:
        state = graph_state["interview"]
        mode = get_mode(state.interview_type)
        question = state.pending_question
        if question is None:  # defensive; the graph shape prevents this
            raise RuntimeError("evaluate reached with no pending question")

        answer = graph_state.get("answer", "")
        outcome = await evaluator.evaluate(
            question=question,
            answer=answer,
            state=state,
            mode=mode,
            grounding=await grounding.for_evaluation(
                question=question, state=state, mode=mode
            ),
            request_id=graph_state.get("request_id"),
        )
        state = state_ops.record_answer(
            state, answer=answer, evaluation=outcome.evaluation, degraded=outcome.degraded
        )
        plan = choose_next_action(
            evaluation=outcome.evaluation, state=state, mode=mode, now=clock.now()
        )
        logger.info(
            "interview_turn interview_id=%s answered=%d action=%s reason=%s degraded=%s",
            state.interview_id,
            state.answered_count,
            plan.action,
            plan.reason,
            outcome.degraded,
        )
        await repository.save(state)
        return {"interview": state, "last_plan": plan}

    async def finish(graph_state: InterviewGraphState) -> InterviewGraphState:
        state = state_ops.complete(graph_state["interview"], clock.now())
        await repository.save(state)
        if on_completed is not None:
            await on_completed(state)
        # last_plan is kept: the caller reads why the interview ended from it.
        return {"interview": state}

    def route_after_evaluation(graph_state: InterviewGraphState) -> str:
        plan = graph_state.get("last_plan")
        return "finish" if (plan is not None and plan.ends_interview) else "ask"

    graph = StateGraph(InterviewGraphState)
    graph.add_node("ask", ask)
    graph.add_node("await_answer", await_answer)
    graph.add_node("evaluate", evaluate)
    graph.add_node("finish", finish)

    graph.add_edge(START, "ask")
    graph.add_edge("ask", "await_answer")
    graph.add_edge("await_answer", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_evaluation, {"ask": "ask", "finish": "finish"}
    )
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer)
