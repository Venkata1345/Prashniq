"""Phase 4: the persistent candidate skill profile.

Decay math and aggregation are pure functions; recording and personalisation
are exercised through the orchestrator against in-memory repositories.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.context.blueprint import build_blueprint
from app.interview.modes import get_mode
from app.interview.schemas import InterviewStatus
from app.main import create_app
from app.core.config import Settings
from app.profile.repository import InMemorySkillObservationRepository
from app.profile.schemas import (
    HALF_LIFE_DAYS,
    SkillObservation,
    aggregate,
    decay_weight,
)
from app.profile.service import SkillProfileService
from tests.conftest import (
    FixedClock,
    build_orchestrator,
    evaluation_json,
    scripted_gateway,
)

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
MODE = get_mode("ml_fundamentals")


def observation(
    *,
    concept: str = "RAG",
    kind: str = "topic",
    score: float = 7.0,
    interview_id: str = "iv-1",
    days_ago: float = 0.0,
    candidate_id: str = "cand-1",
) -> SkillObservation:
    return SkillObservation(
        candidate_id=candidate_id,
        concept=concept,
        kind=kind,  # type: ignore[arg-type]
        score=score,
        interview_id=interview_id,
        interview_type=MODE.key,
        observed_at=NOW - timedelta(days=days_ago),
    )


class TestDecayAndAggregation:
    def test_a_fresh_observation_has_full_weight(self) -> None:
        assert decay_weight(NOW, NOW) == pytest.approx(1.0)

    def test_one_half_life_halves_the_weight(self) -> None:
        assert decay_weight(NOW - timedelta(days=HALF_LIFE_DAYS), NOW) == pytest.approx(0.5)

    def test_future_observations_do_not_gain_weight(self) -> None:
        assert decay_weight(NOW + timedelta(days=30), NOW) == pytest.approx(1.0)

    def test_newer_observations_dominate_the_score(self) -> None:
        profile = aggregate(
            "cand-1",
            [
                observation(score=2.0, interview_id="old", days_ago=HALF_LIFE_DAYS),
                observation(score=8.0, interview_id="new", days_ago=0.0),
            ],
            NOW,
        )
        entry = profile.skills[0]

        # Weighted mean: (8*1.0 + 2*0.5) / 1.5 = 6.0 -- pulled toward recent.
        assert entry.score == pytest.approx(6.0)
        assert entry.observations == 2
        assert entry.last_observed_at == NOW

    def test_kinds_do_not_mix(self) -> None:
        profile = aggregate(
            "cand-1",
            [
                observation(concept="RAG", kind="topic", score=8.0),
                observation(concept="RAG", kind="concept", score=2.0),
            ],
            NOW,
        )

        assert len(profile.skills) == 2
        assert {entry.kind for entry in profile.skills} == {"topic", "concept"}

    def test_concepts_merge_case_insensitively(self) -> None:
        profile = aggregate(
            "cand-1",
            [
                observation(concept="RAG", score=6.0, interview_id="a"),
                observation(concept="rag", score=8.0, interview_id="b"),
            ],
            NOW,
        )

        assert len(profile.skills) == 1
        assert profile.skills[0].score == pytest.approx(7.0)

    def test_weak_skills_come_back_weakest_first(self) -> None:
        profile = aggregate(
            "cand-1",
            [
                observation(concept="transformers", score=8.1, interview_id="a"),
                observation(concept="ml system design", score=3.2, interview_id="b"),
                observation(concept="mlops", score=4.5, interview_id="c"),
            ],
            NOW,
        )

        assert profile.weak_skills() == ["ml system design", "mlops"]

    def test_an_empty_history_is_an_empty_profile(self) -> None:
        profile = aggregate("cand-1", [], NOW)
        assert profile.skills == []
        assert profile.weak_skills() == []


class TestRecording:
    async def finished_interview(self, service: SkillProfileService, **kwargs):
        """Run a real interview to completion through the orchestrator."""
        gateway = scripted_gateway(
            evaluations=[
                evaluation_json(
                    correctness=8.0,
                    depth=6.0,
                    concepts_covered=["L2 penalty"],
                    missing_concepts=["early stopping"],
                    recommended_action="change_topic",
                )
            ]
            * 20
        )
        orchestrator = build_orchestrator(
            gateway, FixedClock(), profile_service=service, **kwargs
        )
        state = await orchestrator.create_interview(candidate_id="cand-1")
        await orchestrator.start(state.interview_id)
        while True:
            result = await orchestrator.submit_answer(state.interview_id, "answer")
            if result.state.status is InterviewStatus.COMPLETED:
                return orchestrator, result.state

    async def test_a_completed_interview_records_topic_and_concept_observations(
        self,
    ) -> None:
        repository = InMemorySkillObservationRepository()
        service = SkillProfileService(repository, clock=FixedClock())

        _, state = await self.finished_interview(service)
        observations = await repository.for_candidate("cand-1")

        kinds = {item.kind for item in observations}
        assert kinds == {"topic", "concept"}
        topics = [item.concept for item in observations if item.kind == "topic"]
        assert set(topics) <= set(MODE.topics)
        concepts = {item.concept for item in observations if item.kind == "concept"}
        assert {"l2 penalty", "early stopping"} <= concepts
        # correctness 8 * 0.7 + depth 6 * 0.3
        topic_scores = [i.score for i in observations if i.kind == "topic"]
        assert topic_scores and all(score == pytest.approx(7.4) for score in topic_scores)

    async def test_re_completing_records_nothing_new(self) -> None:
        repository = InMemorySkillObservationRepository()
        service = SkillProfileService(repository, clock=FixedClock())

        orchestrator, state = await self.finished_interview(service)
        before = len(await repository.for_candidate("cand-1"))
        await orchestrator.complete(state.interview_id)

        assert len(await repository.for_candidate("cand-1")) == before

    async def test_anonymous_interviews_record_nothing(self) -> None:
        repository = InMemorySkillObservationRepository()
        service = SkillProfileService(repository, clock=FixedClock())
        gateway = scripted_gateway(
            evaluations=[evaluation_json(recommended_action="change_topic")] * 20
        )
        orchestrator = build_orchestrator(gateway, FixedClock(), profile_service=service)

        state = await orchestrator.create_interview()  # no candidate_id
        await orchestrator.start(state.interview_id)
        await orchestrator.complete(state.interview_id)

        assert await repository.for_candidate("cand-1") == []

    async def test_only_completed_interviews_are_recordable(self) -> None:
        service = SkillProfileService(InMemorySkillObservationRepository())
        gateway = scripted_gateway()
        orchestrator = build_orchestrator(gateway, FixedClock())
        state = await orchestrator.create_interview(candidate_id="cand-1")

        with pytest.raises(ValueError):
            await service.record_interview(state)

    async def test_degraded_turns_do_not_become_topic_evidence(self) -> None:
        repository = InMemorySkillObservationRepository()
        service = SkillProfileService(repository, clock=FixedClock())
        # Every evaluation fails -> every turn degraded -> no topic observations.
        gateway = scripted_gateway(evaluations=["not json"] * 60)
        orchestrator = build_orchestrator(gateway, FixedClock(), profile_service=service)

        state = await orchestrator.create_interview(candidate_id="cand-1")
        await orchestrator.start(state.interview_id)
        await orchestrator.submit_answer(state.interview_id, "answer")
        await orchestrator.complete(state.interview_id)

        observations = await repository.for_candidate("cand-1")
        assert [item for item in observations if item.kind == "topic"] == []


class TestPersonalization:
    async def test_weak_skills_boost_matching_mode_topics(self) -> None:
        weak = ["regularization"]
        blueprint = await build_blueprint(MODE, focus_skills=weak)
        boosted = blueprint.find("regularization")

        assert boosted is not None
        assert boosted.priority == pytest.approx(0.75)  # 0.25 mode + 0.5 weak
        assert "weak in profile" in boosted.rationale
        # The weak topic now leads the interview.
        assert blueprint.topic_keys()[0] == "regularization"

    async def test_unrelated_focus_skills_change_nothing(self) -> None:
        plain = await build_blueprint(MODE)
        focused = await build_blueprint(MODE, focus_skills=["underwater basket weaving"])

        assert focused.topic_keys() == plain.topic_keys()
        assert [t.priority for t in focused.topics] == [t.priority for t in plain.topics]

    async def test_the_orchestrator_feeds_the_profile_back_into_the_blueprint(
        self,
    ) -> None:
        repository = InMemorySkillObservationRepository()
        clock = FixedClock()
        service = SkillProfileService(repository, clock=clock)
        await repository.add(
            [observation(concept="regularization", kind="topic", score=2.0)]
        )
        orchestrator = build_orchestrator(
            scripted_gateway(), clock, profile_service=service
        )

        state = await orchestrator.create_interview(candidate_id="cand-1")

        assert state.remaining_topics[0] == "regularization"
        assert "weak in profile" in state.blueprint.find("regularization").rationale

    async def test_a_profile_outage_costs_personalisation_not_the_interview(self) -> None:
        class BrokenRepository(InMemorySkillObservationRepository):
            async def for_candidate(self, candidate_id: str):
                raise RuntimeError("profile store is down")

        service = SkillProfileService(BrokenRepository(), clock=FixedClock())
        orchestrator = build_orchestrator(
            scripted_gateway(), FixedClock(), profile_service=service
        )

        state = await orchestrator.create_interview(candidate_id="cand-1")

        assert state.remaining_topics == list(MODE.topics)


class TestProfileApi:
    @pytest.fixture
    def client(self) -> TestClient:
        app = create_app(
            Settings(llm_provider="fake", embedding_provider="fake", vector_store="memory", database_url=None)
        )
        return TestClient(app)

    def test_profile_builds_up_over_a_completed_interview(self, client: TestClient) -> None:
        with client:
            interview_id = client.post(
                "/interviews", json={"candidate_id": "cand-9"}
            ).json()["interview_id"]
            client.post(f"/interviews/{interview_id}/start")
            client.post(f"/interviews/{interview_id}/answers", json={"answer": "a"})
            client.post(f"/interviews/{interview_id}/complete")

            profile = client.get("/candidates/cand-9/profile").json()

        assert profile["candidate_id"] == "cand-9"
        assert profile["topics"], "expected at least one topic observation"
        assert all(0 <= entry["score"] <= 10 for entry in profile["topics"])

    def test_an_unknown_candidate_has_an_empty_profile_not_a_404(
        self, client: TestClient
    ) -> None:
        with client:
            response = client.get("/candidates/nobody/profile")

        assert response.status_code == 200
        assert response.json()["topics"] == []
        assert response.json()["recommended_focus"] == []
