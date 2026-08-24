"""What LangGraph checkpointing actually buys: an interview survives the death
of the process that was running it.

Two orchestrator instances share a checkpointer and a repository but nothing
in memory -- the second one picks up mid-interview exactly where the first
stopped.
"""

from __future__ import annotations

import itertools

from langgraph.checkpoint.memory import InMemorySaver

from app.interview.evaluator import Evaluator
from app.interview.interviewer import Interviewer
from app.interview.orchestrator import InterviewOrchestrator
from app.interview.repository import InMemoryInterviewRepository
from app.interview.schemas import InterviewStatus
from tests.conftest import FixedClock, evaluation_json, scripted_gateway


def orchestrator_process(
    shared_checkpointer: InMemorySaver,
    shared_repository: InMemoryInterviewRepository,
    *,
    id_start: int = 1,
    evaluations: list[str] | None = None,
) -> InterviewOrchestrator:
    """One 'process': fresh objects everywhere except the durable pieces."""
    ids = itertools.count(id_start)
    return InterviewOrchestrator(
        evaluator=Evaluator(scripted_gateway(evaluations=evaluations)),
        interviewer=Interviewer(scripted_gateway()),
        repository=shared_repository,
        clock=FixedClock(),
        id_factory=lambda: f"id-{next(ids)}",
        checkpointer=shared_checkpointer,
    )


async def test_an_interview_resumes_after_the_process_dies() -> None:
    checkpointer = InMemorySaver()
    repository = InMemoryInterviewRepository()

    # Process 1: start and answer once, then "crash".
    first = orchestrator_process(checkpointer, repository)
    state = await first.create_interview(candidate_id="cand-1")
    question_1 = await first.start(state.interview_id)
    result = await first.submit_answer(state.interview_id, "First answer.")
    assert result.next_question is not None
    del first

    # Process 2: same checkpointer + repository, nothing else shared.
    second = orchestrator_process(checkpointer, repository, id_start=100)
    resumed = await second.submit_answer(state.interview_id, "Second answer.")

    assert resumed.state.answered_count == 2
    assert resumed.state.turns[0].answer == "First answer."
    assert resumed.state.turns[1].answer == "Second answer."
    assert resumed.next_question is not None
    assert resumed.next_question.index == 3
    assert question_1.index == 1


async def test_a_resumed_interview_still_terminates_and_reports() -> None:
    checkpointer = InMemorySaver()
    repository = InMemoryInterviewRepository()

    first = orchestrator_process(checkpointer, repository)
    state = await first.create_interview()
    await first.start(state.interview_id)
    await first.submit_answer(state.interview_id, "Answer one.")
    del first

    second = orchestrator_process(
        checkpointer,
        repository,
        id_start=100,
        evaluations=[evaluation_json(recommended_action="change_topic")] * 20,
    )
    while True:
        result = await second.submit_answer(state.interview_id, "Another answer.")
        if result.state.status is InterviewStatus.COMPLETED:
            break

    report = await second.get_report(state.interview_id)
    assert report.questions_answered == result.state.answered_count
    assert result.next_question is None


async def test_start_remains_idempotent_across_processes() -> None:
    checkpointer = InMemorySaver()
    repository = InMemoryInterviewRepository()

    first = orchestrator_process(checkpointer, repository)
    state = await first.create_interview()
    question = await first.start(state.interview_id)
    del first

    second = orchestrator_process(checkpointer, repository, id_start=100)
    again = await second.start(state.interview_id)

    assert again == question
