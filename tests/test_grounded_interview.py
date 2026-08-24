"""Integration: resume + job description -> blueprint -> grounded interview.

The Phase 2 payoff is that a follow-up can quote the candidate's own claim, so
these tests assert on what actually reaches the interviewer's prompt.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.context.ingestion import DocumentIngestionError
from app.core.config import Settings
from app.interview.grounding import GroundingService
from app.interview.schemas import InterviewStatus
from app.interview.state import InvalidInterviewState
from app.rag.knowledge import seed_knowledge_base
from app.rag.schemas import VectorStoreError
from app.llm.fake import FakeLLMGateway, RecordedCall
from app.main import create_app
from tests.conftest import (
    FixedClock,
    build_context_service,
    build_retrieval_stack,
    build_orchestrator,
    evaluation_json,
    job_profile_json,
    question_json,
    resume_profile_json,
)

RESUME = "Built a RAG system using FAISS and FastAPI serving 200 requests per second."
JOB = "We need an AI engineer with production RAG experience."

CLAIM = "Built a RAG system using FAISS and FastAPI."


def context_gateway(
    *,
    claims: list[tuple[str, list[str]]] | None = None,
    requirements: list[tuple[str, str]] | None = None,
    evaluations: list[str] | None = None,
    questions: list[str] | None = None,
) -> FakeLLMGateway:
    """One gateway routing all four call purposes."""
    evaluation_queue = list(evaluations or [])
    question_queue = list(questions or [])

    def responder(call: RecordedCall) -> str:
        if call.purpose == "ingest_resume":
            return resume_profile_json(claims=claims or [(CLAIM, ["RAG", "FAISS"])])
        if call.purpose == "ingest_job_description":
            return job_profile_json(requirements=requirements or [("RAG", "must_have")])
        if call.purpose == "evaluate_answer":
            return evaluation_queue.pop(0) if evaluation_queue else evaluation_json()
        return question_queue.pop(0) if question_queue else question_json(topic="RAG")

    return FakeLLMGateway(responder=responder)


async def grounded_interview(gateway: FakeLLMGateway, interview_type: str = "jd_targeted"):
    clock = FixedClock()
    contexts = build_context_service(gateway, clock)
    orchestrator = build_orchestrator(gateway, clock, contexts)

    context = await contexts.create(resume_text=RESUME, job_description_text=JOB)
    state = await orchestrator.create_interview(
        interview_type=interview_type, context_id=context.context_id
    )
    return orchestrator, state, context


class TestGroundedLoop:
    async def test_the_interview_is_planned_from_the_role_and_the_resume(self) -> None:
        gateway = context_gateway(
            requirements=[("RAG", "must_have"), ("Kubernetes", "nice_to_have")]
        )
        _, state, context = await grounded_interview(gateway)

        assert state.context_id == context.context_id
        assert state.blueprint is not None
        # The required skill the candidate also claims comes first.
        assert state.remaining_topics[:2] == ["RAG", "Kubernetes"]
        assert state.resume is not None

    async def test_the_opening_question_is_grounded_in_a_resume_claim(self) -> None:
        gateway = context_gateway()
        orchestrator, state, _ = await grounded_interview(gateway)

        question = await orchestrator.start(state.interview_id)
        prompt = gateway.calls[-1].last_user_message

        assert question.topic == "RAG"
        assert CLAIM in prompt
        assert "Skill the target role requires: RAG" in prompt

    async def test_the_evaluator_knows_which_role_skill_is_being_tested(self) -> None:
        gateway = context_gateway()
        orchestrator, state, _ = await grounded_interview(gateway)
        await orchestrator.start(state.interview_id)

        await orchestrator.submit_answer(state.interview_id, "We used FAISS with HNSW.")
        evaluation_prompt = next(
            call.last_user_message
            for call in gateway.calls
            if call.purpose == "evaluate_answer"
        )

        assert "Skill the target role requires: RAG" in evaluation_prompt

    async def test_a_follow_up_on_an_unplanned_topic_still_finds_evidence(self) -> None:
        gateway = context_gateway(
            claims=[(CLAIM, ["RAG", "FAISS"]), ("Tuned FastAPI workers.", ["FastAPI"])],
            evaluations=[
                evaluation_json(
                    recommended_action="probe_deeper", follow_up_topic="FastAPI throughput"
                )
            ],
        )
        orchestrator, state, _ = await grounded_interview(gateway)
        await orchestrator.start(state.interview_id)

        result = await orchestrator.submit_answer(state.interview_id, "An answer.")
        prompt = gateway.calls[-1].last_user_message

        assert result.next_question is not None
        assert result.next_question.topic == "FastAPI throughput"
        # The topic was never planned, so this evidence came from retrieval.
        assert "Tuned FastAPI workers." in prompt

    async def test_an_ungrounded_topic_produces_no_invented_evidence(self) -> None:
        gateway = context_gateway(requirements=[("Kubernetes", "must_have")])
        orchestrator, state, _ = await grounded_interview(gateway)

        await orchestrator.start(state.interview_id)
        prompt = gateway.calls[-1].last_user_message

        assert "The candidate claims" not in prompt
        assert CLAIM not in prompt

    async def test_the_report_names_role_skills_the_interview_never_reached(self) -> None:
        # More required skills than an interview that ends at its minimum
        # length can possibly cover.
        gateway = context_gateway(
            requirements=[
                ("RAG", "must_have"),
                ("Kubernetes", "must_have"),
                ("model serving", "must_have"),
                ("feature stores", "must_have"),
                ("drift monitoring", "must_have"),
                ("cost optimisation", "must_have"),
            ],
            evaluations=[evaluation_json(recommended_action="end_interview")] * 20,
        )
        orchestrator, state, _ = await grounded_interview(gateway)
        await orchestrator.start(state.interview_id)

        while True:
            result = await orchestrator.submit_answer(state.interview_id, "An answer.")
            if result.next_question is None:
                break

        report = await orchestrator.get_report(state.interview_id)
        asked = {question.topic for question in result.state.questions_asked}
        planned = set(state.blueprint.target_skills())

        assert report.unaddressed_target_skills
        assert set(report.unaddressed_target_skills) == planned - asked
        assert "RAG" in asked and "RAG" not in report.unaddressed_target_skills

    async def test_a_context_free_mode_still_works_and_reports_no_target_skills(self) -> None:
        gateway = context_gateway()
        orchestrator = build_orchestrator(gateway)
        state = await orchestrator.create_interview(interview_type="ml_fundamentals")

        assert state.blueprint is not None
        assert state.blueprint.target_skills() == []
        assert state.resume is None

    async def test_a_context_only_mode_without_a_context_is_rejected(self) -> None:
        from app.context.blueprint import BlueprintError

        orchestrator = build_orchestrator(context_gateway())
        with pytest.raises(BlueprintError):
            await orchestrator.create_interview(interview_type="resume_deep_dive")

    async def test_a_context_id_without_a_context_service_is_rejected(self) -> None:
        orchestrator = build_orchestrator(context_gateway())
        with pytest.raises(InvalidInterviewState):
            await orchestrator.create_interview(context_id="ctx-1")

    async def test_a_context_needs_at_least_one_document(self) -> None:
        service = build_context_service(context_gateway())
        with pytest.raises(DocumentIngestionError):
            await service.create()


class TestContextApi:
    @pytest.fixture
    def client(self) -> TestClient:
        app = create_app(Settings(llm_provider="fake", embedding_provider="fake", vector_store="memory", database_url=None))
        gateway = context_gateway()
        clock = FixedClock()
        contexts = build_context_service(gateway, clock)
        app.state.context_service = contexts
        app.state.orchestrator = build_orchestrator(gateway, clock, contexts)
        return TestClient(app)

    def test_ingesting_documents_returns_the_extracted_structure(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/candidate-contexts",
            json={"resume_text": RESUME, "job_description_text": JOB},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["role_title"] == "AI Engineer"
        assert body["claims"][0]["text"] == CLAIM
        assert body["requirements"][0]["skill"] == "RAG"
        # The submitted documents are not echoed back or stored.
        assert "resume_text" not in body

    def test_an_interview_can_be_created_from_a_context_and_exposes_its_plan(
        self, client: TestClient
    ) -> None:
        context_id = client.post(
            "/candidate-contexts",
            json={"resume_text": RESUME, "job_description_text": JOB},
        ).json()["context_id"]

        created = client.post(
            "/interviews", json={"interview_type": "jd_targeted", "context_id": context_id}
        )
        assert created.status_code == 201
        interview_id = created.json()["interview_id"]
        assert created.json()["context_id"] == context_id

        blueprint = client.get(f"/interviews/{interview_id}/blueprint").json()
        assert blueprint["topics"][0]["key"] == "RAG"
        assert blueprint["topics"][0]["source"] == "job_description"
        assert blueprint["topics"][0]["evidence"] == [CLAIM]

    def test_an_empty_submission_is_rejected(self, client: TestClient) -> None:
        assert client.post("/candidate-contexts", json={}).status_code == 422

    def test_an_unknown_context_is_a_404(self, client: TestClient) -> None:
        assert client.get("/candidate-contexts/nope").status_code == 404
        assert (
            client.post(
                "/interviews", json={"interview_type": "jd_targeted", "context_id": "nope"}
            ).status_code
            == 404
        )

    def test_a_context_only_mode_without_a_context_is_a_422(self, client: TestClient) -> None:
        response = client.post("/interviews", json={"interview_type": "resume_deep_dive"})
        assert response.status_code == 422

    def test_context_free_interviews_have_no_blueprint_evidence(
        self, client: TestClient
    ) -> None:
        interview_id = client.post("/interviews", json={}).json()["interview_id"]
        blueprint = client.get(f"/interviews/{interview_id}/blueprint").json()

        assert {topic["source"] for topic in blueprint["topics"]} == {"mode"}


class TestRetrievalGroundedInterview:
    """Phase 3: the same loop, with the knowledge base and vector retrieval
    wired in behind the grounding service."""

    async def rag_interview(self, gateway: FakeLLMGateway):
        indexer, retriever = build_retrieval_stack()
        await seed_knowledge_base(indexer)

        clock = FixedClock()
        contexts = build_context_service(gateway, clock, indexer)
        orchestrator = build_orchestrator(
            gateway, clock, contexts, GroundingService(retriever)
        )
        context = await contexts.create(resume_text=RESUME, job_description_text=JOB)
        state = await orchestrator.create_interview(
            interview_type="ml_fundamentals", context_id=context.context_id
        )
        return orchestrator, state

    async def test_the_interviewer_is_given_reference_points_it_must_not_reveal(
        self,
    ) -> None:
        gateway = context_gateway(requirements=[("regularization", "must_have")])
        orchestrator, state = await self.rag_interview(gateway)

        await orchestrator.start(state.interview_id)
        prompt = gateway.calls[-1].last_user_message

        assert "Reference points a complete answer" in prompt
        assert "weight decay" in prompt
        assert "Do NOT state them" in prompt

    async def test_the_evaluator_judges_against_retrieved_reference_points(self) -> None:
        gateway = context_gateway(requirements=[("regularization", "must_have")])
        orchestrator, state = await self.rag_interview(gateway)
        await orchestrator.start(state.interview_id)

        await orchestrator.submit_answer(state.interview_id, "I would add L2.")
        prompt = next(
            call.last_user_message
            for call in gateway.calls
            if call.purpose == "evaluate_answer"
        )

        assert "REFERENCE POINTS" in prompt
        assert "dropout" in prompt
        assert "report what the answer missed relative to them" in prompt

    async def test_resume_claims_are_retrieved_semantically_not_lexically(self) -> None:
        # The requirement shares no tokens with the claim; only retrieval links
        # "vector search" to a claim about FAISS.
        gateway = context_gateway(
            claims=[("Tuned a FAISS HNSW index for recall at 50ms.", ["FAISS", "HNSW"])],
            requirements=[("FAISS HNSW recall tuning", "must_have")],
        )
        orchestrator, state = await self.rag_interview(gateway)

        planned = state.blueprint.find("FAISS HNSW recall tuning")

        assert planned is not None
        assert planned.evidence == ("Tuned a FAISS HNSW index for recall at 50ms.",)
        assert "claimed on the resume" in planned.rationale

    async def test_a_retrieval_outage_costs_grounding_not_the_interview(self) -> None:
        gateway = context_gateway()
        indexer, retriever = build_retrieval_stack()
        await seed_knowledge_base(indexer)

        clock = FixedClock()
        contexts = build_context_service(gateway, clock, indexer)
        orchestrator = build_orchestrator(
            gateway, clock, contexts, GroundingService(retriever)
        )
        context = await contexts.create(resume_text=RESUME, job_description_text=JOB)
        state = await orchestrator.create_interview(
            interview_type="ml_fundamentals", context_id=context.context_id
        )
        await orchestrator.start(state.interview_id)

        # The store goes down mid-interview.
        async def broken_search(*args, **kwargs):
            raise VectorStoreError("database is down")

        retriever._store.search = broken_search  # noqa: SLF001 - simulating an outage

        result = await orchestrator.submit_answer(state.interview_id, "An answer.")

        assert result.next_question is not None
        assert result.state.status is InterviewStatus.IN_PROGRESS
