"""Integration: the full adaptive loop, through the orchestrator and the API.

start -> question -> answer -> evaluation -> adaptive follow-up -> report,
driven entirely by the fake gateway.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.interview.modes import get_mode
from app.interview.schemas import InterviewStatus
from app.interview.state import InvalidInterviewState
from app.main import create_app
from tests.conftest import (
    FixedClock,
    build_orchestrator,
    evaluation_json,
    question_json,
    scripted_gateway,
)

MODE = get_mode("ml_fundamentals")


class TestOrchestratorLoop:
    async def test_start_asks_an_opening_question_on_the_first_topic(self) -> None:
        gateway = scripted_gateway(questions=[question_json("What is regularization for?")])
        orchestrator = build_orchestrator(gateway)

        state = await orchestrator.create_interview(interview_type="ml_fundamentals")
        question = await orchestrator.start(state.interview_id)

        assert question.index == 1
        assert question.text == "What is regularization for?"
        assert question.topic == MODE.topics[0]
        assert question.action == "open_question"

        stored = await orchestrator.get_state(state.interview_id)
        assert stored.status is InterviewStatus.IN_PROGRESS
        assert stored.pending_question == question

    async def test_start_is_idempotent_for_a_reconnecting_client(self) -> None:
        gateway = scripted_gateway()
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview()

        first = await orchestrator.start(state.interview_id)
        second = await orchestrator.start(state.interview_id)

        assert first == second
        assert len([c for c in gateway.calls if c.purpose == "generate_question"]) == 1

    async def test_answer_produces_an_adaptive_follow_up_on_the_same_thread(self) -> None:
        gateway = scripted_gateway(
            evaluations=[
                evaluation_json(
                    correctness=6.0,
                    recommended_action="probe_deeper",
                    missing_concepts=["weight decay vs L2"],
                )
            ],
            questions=[
                question_json("Opening question?"),
                question_json("How does weight decay differ from L2?"),
            ],
        )
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview()
        await orchestrator.start(state.interview_id)

        result = await orchestrator.submit_answer(
            state.interview_id, "Regularization prevents overfitting."
        )

        assert result.plan.action == "probe_deeper"
        assert result.next_question is not None
        assert result.next_question.index == 2
        assert result.next_question.topic == MODE.topics[0]
        assert result.state.follow_up_depth == 1
        assert result.state.answered_count == 1

        # The follow-up prompt is grounded in the gap the evaluator found.
        follow_up_prompt = gateway.calls[-1].last_user_message
        assert "weight decay vs L2" in follow_up_prompt
        assert "Regularization prevents overfitting." in follow_up_prompt

    async def test_a_weak_answer_lowers_difficulty_and_a_strong_one_raises_it(self) -> None:
        gateway = scripted_gateway(
            evaluations=[
                evaluation_json(correctness=2.0, recommended_action="decrease_difficulty"),
                evaluation_json(correctness=9.0, recommended_action="increase_difficulty"),
            ]
        )
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview()
        await orchestrator.start(state.interview_id)

        weak = await orchestrator.submit_answer(state.interview_id, "I do not know.")
        assert weak.next_question is not None and weak.next_question.difficulty == 2

        strong = await orchestrator.submit_answer(state.interview_id, "A precise answer.")
        assert strong.next_question is not None and strong.next_question.difficulty == 3

    async def test_topic_change_moves_to_the_next_topic(self) -> None:
        gateway = scripted_gateway(
            evaluations=[evaluation_json(recommended_action="change_topic")]
        )
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview()
        await orchestrator.start(state.interview_id)

        result = await orchestrator.submit_answer(state.interview_id, "An answer.")

        assert result.next_question is not None
        assert result.next_question.topic == MODE.topics[1]
        assert result.state.topics_covered == list(MODE.topics[:2])

    async def test_the_loop_terminates_and_produces_a_report(self) -> None:
        gateway = scripted_gateway()
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview()
        await orchestrator.start(state.interview_id)

        answers = 0
        result = None
        while answers < MODE.max_questions + 2:
            result = await orchestrator.submit_answer(state.interview_id, f"Answer {answers}")
            answers += 1
            if result.state.status is InterviewStatus.COMPLETED:
                break

        assert result is not None
        assert result.state.status is InterviewStatus.COMPLETED
        assert result.next_question is None
        assert answers == MODE.max_questions

        report = await orchestrator.get_report(state.interview_id)
        assert report.questions_answered == MODE.max_questions
        assert 0 < report.overall_score <= 10
        assert set(report.dimension_scores) <= set(MODE.dimensions)

    async def test_answering_without_an_outstanding_question_is_rejected(self) -> None:
        orchestrator = build_orchestrator(scripted_gateway())
        state = await orchestrator.create_interview()

        with pytest.raises(InvalidInterviewState):
            await orchestrator.submit_answer(state.interview_id, "too early")

    async def test_report_is_unavailable_before_completion(self) -> None:
        orchestrator = build_orchestrator(scripted_gateway())
        state = await orchestrator.create_interview()
        await orchestrator.start(state.interview_id)

        with pytest.raises(InvalidInterviewState):
            await orchestrator.get_report(state.interview_id)

    async def test_a_failing_evaluator_does_not_break_the_interview(self) -> None:
        gateway = scripted_gateway(evaluations=["not json"] * 3)
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview()
        await orchestrator.start(state.interview_id)

        result = await orchestrator.submit_answer(state.interview_id, "An answer.")

        assert result.evaluation_degraded is True
        assert result.next_question is not None
        assert result.state.turns[0].evaluation_degraded is True


class TestApi:
    @pytest.fixture
    def client(self) -> TestClient:
        app = create_app(Settings(llm_provider="fake", embedding_provider="fake", vector_store="memory", database_url=None))
        app.state.orchestrator = build_orchestrator(scripted_gateway(), FixedClock())
        return TestClient(app)

    def test_full_http_round_trip(self, client: TestClient) -> None:
        created = client.post("/interviews", json={"interview_type": "ml_fundamentals"})
        assert created.status_code == 201
        interview_id = created.json()["interview_id"]

        started = client.post(f"/interviews/{interview_id}/start")
        assert started.status_code == 200
        assert started.json()["index"] == 1

        answered = client.post(
            f"/interviews/{interview_id}/answers", json={"answer": "My answer."}
        )
        assert answered.status_code == 200
        body = answered.json()
        assert body["interview_complete"] is False
        assert body["next_question"]["index"] == 2
        # Interview first, feedback later: no scores leak mid-interview.
        assert "evaluation" not in body

        state = client.get(f"/interviews/{interview_id}").json()
        assert state["questions_answered"] == 1
        assert state["status"] == "in_progress"

        completed = client.post(f"/interviews/{interview_id}/complete")
        assert completed.json()["status"] == "completed"

        report = client.get(f"/interviews/{interview_id}/report")
        assert report.status_code == 200
        assert report.json()["questions_answered"] == 1

    def test_unknown_interview_is_a_404(self, client: TestClient) -> None:
        assert client.get("/interviews/does-not-exist").status_code == 404

    def test_answering_a_completed_interview_is_a_409(self, client: TestClient) -> None:
        interview_id = client.post("/interviews", json={}).json()["interview_id"]
        client.post(f"/interviews/{interview_id}/start")
        client.post(f"/interviews/{interview_id}/complete")

        response = client.post(
            f"/interviews/{interview_id}/answers", json={"answer": "hello"}
        )
        assert response.status_code == 409

    def test_unknown_interview_type_is_rejected(self, client: TestClient) -> None:
        response = client.post("/interviews", json={"interview_type": "underwater_basket"})
        assert response.status_code == 422

    def test_empty_answers_are_rejected_by_validation(self, client: TestClient) -> None:
        interview_id = client.post("/interviews", json={}).json()["interview_id"]
        client.post(f"/interviews/{interview_id}/start")

        assert (
            client.post(f"/interviews/{interview_id}/answers", json={"answer": ""}).status_code
            == 422
        )

    def test_interview_types_are_discoverable(self, client: TestClient) -> None:
        keys = {mode["key"] for mode in client.get("/interview-types").json()}
        assert {"ml_fundamentals", "ml_system_design"} <= keys


def test_fake_provider_app_runs_the_loop_without_wiring_overrides() -> None:
    """The `fake` provider makes the whole service runnable with no API key."""
    client = TestClient(create_app(Settings(llm_provider="fake", embedding_provider="fake", vector_store="memory", database_url=None)))

    interview_id = client.post("/interviews", json={}).json()["interview_id"]
    question = client.post(f"/interviews/{interview_id}/start").json()
    answer = client.post(
        f"/interviews/{interview_id}/answers", json={"answer": "An answer."}
    ).json()

    assert question["text"].startswith("Stubbed question")
    assert answer["next_question"]["index"] == 2
    assert client.get("/health").json()["llm_provider"] == "fake"


async def test_a_topic_change_hides_the_previous_answer_from_the_interviewer() -> None:
    """Telling the model not to reference the old topic while showing it the
    old answer does not work; the prompt must simply omit it."""
    gateway = scripted_gateway(
        evaluations=[evaluation_json(recommended_action="change_topic")]
    )
    orchestrator = build_orchestrator(gateway)
    state = await orchestrator.create_interview()
    await orchestrator.start(state.interview_id)

    await orchestrator.submit_answer(state.interview_id, "My distinctive answer text.")
    change_topic_prompt = gateway.calls[-1].last_user_message

    assert "My distinctive answer text." not in change_topic_prompt
    assert "Previous question" not in change_topic_prompt
