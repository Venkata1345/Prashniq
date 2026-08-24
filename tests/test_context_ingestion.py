"""Chunking, resume/JD extraction and lexical evidence retrieval."""

from __future__ import annotations

import json

import pytest

from app.context.chunking import chunk_text
from app.context.ingestion import (
    DocumentIngestionError,
    JobDescriptionIngestor,
    ResumeIngestor,
)
from app.context.retrieval import score_claim, select_claims, select_evidence, tokenize
from app.context.schemas import JobProfile, ResumeClaim, ResumeProfile
from app.llm.fake import FakeLLMGateway
from tests.conftest import job_profile_json, resume_profile_json

RESUME = """Abhishek Gullipalli
AI Engineer

Experience
Built a RAG system using FAISS and FastAPI serving 200 requests per second.
Fine-tuned a transformer for document classification.

Education
BSc Computer Science
"""


class TestChunking:
    def test_short_documents_are_a_single_chunk(self) -> None:
        assert chunk_text("one line") == ["one line"]

    def test_empty_input_produces_no_chunks(self) -> None:
        assert chunk_text("   \n\n  ") == []

    def test_splits_on_blank_lines_and_respects_the_budget(self) -> None:
        chunks = chunk_text(RESUME, max_chars=80)

        assert len(chunks) > 1
        assert all(len(chunk) <= 80 for chunk in chunks)

    def test_chunking_is_lossless_and_ordered(self) -> None:
        chunks = chunk_text(RESUME, max_chars=120)
        rejoined = "\n".join(chunks)

        for line in (line.strip() for line in RESUME.splitlines() if line.strip()):
            assert line in rejoined
        assert rejoined.index("Abhishek Gullipalli") < rejoined.index("BSc Computer Science")

    def test_a_single_oversized_line_is_split_rather_than_dropped(self) -> None:
        chunks = chunk_text("x" * 250, max_chars=100)

        assert len(chunks) == 3
        assert "".join(chunks) == "x" * 250

    def test_rejects_a_nonsense_budget(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("anything", max_chars=0)


class TestResumeIngestion:
    async def test_extracts_claims_from_every_chunk_and_merges_them(self) -> None:
        gateway = FakeLLMGateway(
            responses=[
                resume_profile_json(claims=[("Built a RAG system.", ["RAG"])]),
                resume_profile_json(claims=[("Fine-tuned a transformer.", ["transformers"])]),
            ]
        )
        profile = await ResumeIngestor(gateway, max_chunk_chars=200).ingest(RESUME)

        assert len(gateway.calls) == 2
        assert [claim.text for claim in profile.claims] == [
            "Built a RAG system.",
            "Fine-tuned a transformer.",
        ]
        assert profile.skills() == ["RAG", "transformers"]

    async def test_duplicate_claims_across_chunks_are_merged(self) -> None:
        duplicate = resume_profile_json(claims=[("Built a RAG system.", ["RAG"])])
        gateway = FakeLLMGateway(responses=[duplicate, duplicate])
        profile = await ResumeIngestor(gateway, max_chunk_chars=200).ingest(RESUME)

        assert len(profile.claims) == 1

    async def test_one_unreadable_section_does_not_lose_the_rest(self) -> None:
        gateway = FakeLLMGateway(
            responses=[
                "garbage",
                "garbage",
                "garbage",
                resume_profile_json(claims=[("Built a RAG system.", ["RAG"])]),
            ]
        )
        profile = await ResumeIngestor(gateway, max_chunk_chars=200).ingest(RESUME)

        assert [claim.text for claim in profile.claims] == ["Built a RAG system."]

    async def test_an_empty_resume_is_rejected(self) -> None:
        with pytest.raises(DocumentIngestionError):
            await ResumeIngestor(FakeLLMGateway(responses=[])).ingest("   ")

    async def test_a_resume_with_no_extractable_claims_is_rejected(self) -> None:
        gateway = FakeLLMGateway(responder=lambda _: resume_profile_json(claims=[]))
        with pytest.raises(DocumentIngestionError):
            await ResumeIngestor(gateway).ingest("Some text.")

    async def test_long_resumes_are_capped_at_the_chunk_limit(self) -> None:
        gateway = FakeLLMGateway(
            responder=lambda _: resume_profile_json(claims=[("A claim.", ["skill"])])
        )
        long_resume = "\n\n".join(f"Section {i} of the resume." for i in range(50))
        await ResumeIngestor(gateway, max_chunks=3, max_chunk_chars=40).ingest(long_resume)

        assert len(gateway.calls) == 3


class TestJobDescriptionIngestion:
    async def test_extracts_requirements_in_one_call(self) -> None:
        gateway = FakeLLMGateway(
            responses=[
                job_profile_json(
                    role_title="AI Engineer",
                    requirements=[("RAG", "must_have"), ("Kubernetes", "nice_to_have")],
                )
            ]
        )
        profile = await JobDescriptionIngestor(gateway).ingest("We need an AI engineer.")

        assert len(gateway.calls) == 1
        assert profile.role_title == "AI Engineer"
        assert [r.skill for r in profile.must_haves()] == ["RAG"]

    async def test_a_job_description_with_no_requirements_is_rejected(self) -> None:
        gateway = FakeLLMGateway(responses=[job_profile_json(requirements=[])])
        with pytest.raises(DocumentIngestionError):
            await JobDescriptionIngestor(gateway).ingest("We are hiring.")

    async def test_an_unusable_response_becomes_a_domain_error(self) -> None:
        gateway = FakeLLMGateway(responses=["not json"] * 3)
        with pytest.raises(DocumentIngestionError):
            await JobDescriptionIngestor(gateway).ingest("We are hiring.")

    def test_importance_defaults_to_nice_to_have(self) -> None:
        profile = JobProfile.model_validate(json.loads(job_profile_json()))
        assert all(r.importance in {"must_have", "nice_to_have"} for r in profile.requirements)


class TestRetrieval:
    @staticmethod
    def claims() -> list[ResumeClaim]:
        return [
            ResumeClaim(
                text="Built a RAG system using FAISS and FastAPI.",
                skills=["RAG", "FAISS", "FastAPI"],
                category="project",
            ),
            ResumeClaim(
                text="Fine-tuned a transformer for document classification.",
                skills=["transformers", "fine-tuning"],
                category="project",
            ),
            ResumeClaim(text="Mentored two junior engineers.", skills=[], category="experience"),
        ]

    def test_tokenize_drops_stopwords(self) -> None:
        assert tokenize("Built a system using FAISS") == {"faiss"}

    def test_skill_matches_outrank_incidental_text_matches(self) -> None:
        faiss_claim, transformer_claim, _ = self.claims()

        assert score_claim("FAISS", faiss_claim) > score_claim("classification", transformer_claim)

    def test_selects_the_relevant_claim_for_a_skill(self) -> None:
        selected = select_claims("FAISS indexing", self.claims())

        assert selected[0].text.startswith("Built a RAG system")

    def test_irrelevant_claims_are_never_returned(self) -> None:
        assert select_claims("Kubernetes autoscaling", self.claims()) == []

    def test_selection_is_capped_and_deterministic(self) -> None:
        profile = ResumeProfile(claims=self.claims())
        first = select_evidence("RAG FAISS transformer", profile, limit=2)
        second = select_evidence("RAG FAISS transformer", profile, limit=2)

        assert first == second
        assert len(first) == 2

    def test_no_resume_means_no_evidence(self) -> None:
        assert select_evidence("anything", None) == []
