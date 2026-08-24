"""Live smoke test: one real interview against the actual providers.

Everything else in this repo runs against fakes; this script is the one place
the *prompts* get validated -- calibration, terseness, structured-output
compliance, retrieval quality. Run it once after any prompt change.

    # real run (~cents). Keys come from the environment OR from .env
    # (ANTHROPIC_API_KEY / OPENAI_API_KEY).
    .venv/Scripts/python -m scripts.live_smoke

    # harness rehearsal with fakes, no keys, no cost
    .venv/Scripts/python -m scripts.live_smoke --fake

What runs depends on which keys exist:
    Anthropic key   -> interview runs on Claude
    OpenAI key only -> interview runs on OpenAI (gpt-4o-mini by default;
                       override with LLM_MODEL)
    OpenAI key      -> embeddings are real, and the retrieval probe suite
                       runs before the interview
Pass --retrieval-only to stop after the retrieval probes.

It prints the full transcript plus a diagnostics block:
    structured-output retries, degraded evaluations, per-call latency,
    token usage, and what grounding actually reached the prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import sys
import time

from app.context.ingestion import JobDescriptionIngestor, ResumeIngestor
from app.context.repository import InMemoryCandidateContextRepository
from app.context.service import CandidateContextService
from app.core.config import Settings
from app.interview.evaluator import Evaluator
from app.interview.grounding import GroundingService
from app.interview.interviewer import Interviewer
from app.interview.orchestrator import InterviewOrchestrator
from app.interview.repository import InMemoryInterviewRepository
from app.llm.factory import build_gateway
from app.rag.factory import build_embeddings, build_retriever, build_vector_store
from app.rag.indexer import Indexer
from app.rag.knowledge import seed_knowledge_base

SAMPLE_RESUME = """Abhishek G.
AI Engineer, 3 years

Experience
Built a RAG pipeline for internal documentation search using FAISS, a
cross-encoder reranker, and FastAPI, serving ~200 requests per second.
Fine-tuned DeBERTa for contract-clause classification, improving F1 from
0.71 to 0.86.
Deployed models on Kubernetes with canary rollouts and drift monitoring.

Education
BSc Computer Science
"""

SAMPLE_JOB = """AI Engineer - Retrieval Systems

Must have: production RAG experience, retrieval evaluation, Python.
Nice to have: reranking, Kubernetes, model monitoring.
"""

# Deliberately mixed quality, so calibration is observable: if the evaluator
# scores answer 3 anywhere near answer 1, the rubric is broken.
CANNED_ANSWERS = [
    # 1: strong, specific
    "We chunked docs on heading boundaries at roughly 500 tokens with 50-token "
    "overlap, embedded with a bi-encoder, retrieved top-20 with FAISS HNSW, then "
    "reranked to top-3 with a cross-encoder. We measured recall@20 on a labelled "
    "set of 400 queries before touching chunk size, because retrieval failures "
    "dominated end-to-end errors.",
    # 2: plausible but vague -- should get pushed on
    "I would use a vector database for retrieval and make sure the embeddings "
    "are good quality. Testing is also important to make sure it works well.",
    # 3: contains a real misconception
    "ROC-AUC is always the best metric for retrieval because it accounts for "
    "class imbalance, so we optimised our reranker for ROC-AUC directly.",
    # 4: honest ignorance
    "I'm not sure, I haven't worked with that directly.",
]


class Diagnostics(logging.Handler):
    """Collects the gateway's own log lines into hard numbers."""

    def __init__(self) -> None:
        super().__init__()
        self.retries: list[str] = []
        self.latencies_ms: list[int] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "structured_output_invalid" in message:
            self.retries.append(message)
        if record.levelno >= logging.ERROR:
            self.errors.append(message)
        if "llm_call" in message or "embedding_call" in message:
            for part in message.split():
                key, _, value = part.partition("=")
                if key == "latency_ms" and value.isdigit():
                    self.latencies_ms.append(int(value))
                elif key == "input_tokens" and value.isdigit():
                    self.tokens_in += int(value)
                elif key == "output_tokens" and value.isdigit():
                    self.tokens_out += int(value)


def build_settings(fake: bool) -> Settings:
    if fake:
        return Settings(llm_provider="fake", embedding_provider="fake", vector_store="memory")

    # Settings reads .env, so keys pasted there count as much as exported ones.
    base = Settings()
    has_anthropic = bool(base.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(base.openai_api_key or os.environ.get("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        sys.exit(
            "No provider keys found. Put ANTHROPIC_API_KEY and/or OPENAI_API_KEY "
            "in the environment or in .env, or rehearse with --fake."
        )
    if not has_anthropic:
        print("NOTE: no ANTHROPIC_API_KEY - the interview will run on OpenAI.\n")
    if not has_openai:
        print(
            "NOTE: OPENAI_API_KEY not set - using deterministic embeddings; "
            "retrieval quality is NOT being validated this run.\n"
        )
    return base.model_copy(
        update={
            "llm_provider": "anthropic" if has_anthropic else "openai",
            "embedding_provider": "openai" if has_openai else "fake",
            "vector_store": "memory",  # the smoke test validates prompts, not storage
        }
    )


# Phrasings that share almost no vocabulary with the notes they should hit, so
# passing requires actual semantic matching -- token overlap cannot fake it.
RETRIEVAL_PROBES = [
    (
        "my model memorises the training set and fails on new examples",
        "overfitting and validation strategy",
    ),
    (
        "why divide the attention scores by the square root of the head size",
        "self-attention mechanics",
    ),
    (
        "how big should the pieces be when splitting documents for search",
        "chunking and indexing strategy",
    ),
]
NEGATIVE_PROBE = "best sourdough starter feeding schedule"


async def run_retrieval_check(settings: Settings) -> int:
    """OpenAI-only mode: validate real-embedding retrieval, no LLM involved."""
    from app.rag.schemas import EmbeddingError

    embeddings = build_embeddings(settings)
    store = build_vector_store(settings)
    indexer = Indexer(embeddings=embeddings, store=store)
    retriever = build_retriever(settings, embeddings=embeddings, store=store)

    print("=" * 78)
    print(f"RETRIEVAL VALIDATION ({settings.embedding_model}, "
          f"knowledge>={settings.knowledge_min_score}, "
          f"claims>={settings.claim_min_score})")
    print("=" * 78)
    # Indexed exactly as production indexes them: skill tags in the embedding,
    # the candidate's own words as the stored text.
    from app.context.schemas import ResumeClaim
    from app.context.service import claim_embedding_text
    from app.rag.indexer import stable_chunk_id
    from app.rag.schemas import DocumentChunk

    claims = [
        ResumeClaim(
            text="Built a RAG pipeline using FAISS, a cross-encoder reranker and FastAPI.",
            skills=["RAG", "FAISS", "reranking", "FastAPI", "vector search"],
        ),
        ResumeClaim(
            text="Fine-tuned DeBERTa for contract-clause classification.",
            skills=["fine-tuning", "DeBERTa", "text classification"],
        ),
        ResumeClaim(
            text="Deployed models on Kubernetes with canary rollouts.",
            skills=["Kubernetes", "model deployment", "canary releases"],
        ),
    ]
    try:
        await seed_knowledge_base(indexer)
        await indexer.index_chunks(
            [
                DocumentChunk(
                    chunk_id=stable_chunk_id(claim.text),
                    collection="resume",
                    owner_id="smoke",
                    text=claim.text,
                    embedding_text=claim_embedding_text(claim),
                )
                for claim in claims
            ]
        )
    except EmbeddingError as exc:
        sys.exit(f"Embedding provider rejected the request - check the key: {exc}")

    failures = 0
    for query, expected_topic in RETRIEVAL_PROBES:
        hits = await retriever.search(query, collection="knowledge", limit=2)
        top = hits[0].chunk.topic if hits else None
        ok = top == expected_topic
        failures += 0 if ok else 1
        print(f"\n  {'PASS' if ok else 'FAIL'}  \"{query}\"")
        for hit in hits:
            print(f"        {hit.score:.3f}  {hit.chunk.topic}")
        if not ok:
            print(f"        expected: {expected_topic}")

    hits = await retriever.search(NEGATIVE_PROBE, collection="knowledge", limit=2)
    ok = not hits
    failures += 0 if ok else 1
    print(f"\n  {'PASS' if ok else 'FAIL'}  negative probe \"{NEGATIVE_PROBE}\" "
          f"-> {[(round(h.score, 3), h.chunk.topic) for h in hits] or 'no hits'}")
    if not ok:
        print("        an off-topic query cleared the threshold: raise "
              "RETRIEVAL_MIN_SCORE")

    claim_hits = await retriever.search(
        "tuning a vector similarity search index", collection="resume", owner_id="smoke"
    )
    ok = bool(claim_hits) and "FAISS" in claim_hits[0].text
    failures += 0 if ok else 1
    print(f"\n  {'PASS' if ok else 'FAIL'}  resume-claim matching -> "
          f"{[(round(h.score, 3), h.text[:50]) for h in claim_hits] or 'no hits'}")

    print(f"\n  {failures} failure(s). Scores above are real cosine similarities: "
          "if passes sit near a threshold, tune KNOWLEDGE_MIN_SCORE / CLAIM_MIN_SCORE.")
    return 1 if failures else 0


async def run(fake: bool, turns: int, retrieval_only: bool) -> int:
    settings = build_settings(fake)

    retrieval_failures = 0
    if settings.embedding_provider == "openai":
        retrieval_failures = await run_retrieval_check(settings)
        print()
    if retrieval_only:
        return retrieval_failures

    diagnostics = Diagnostics()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("app").addHandler(diagnostics)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    gateway = build_gateway(settings)
    embeddings = build_embeddings(settings)
    store = build_vector_store(settings)
    indexer = Indexer(embeddings=embeddings, store=store)
    retriever = build_retriever(settings, embeddings=embeddings, store=store)
    await seed_knowledge_base(indexer)

    contexts = CandidateContextService(
        resume_ingestor=ResumeIngestor(gateway),
        job_ingestor=JobDescriptionIngestor(gateway),
        repository=InMemoryCandidateContextRepository(),
        indexer=indexer,
    )
    orchestrator = InterviewOrchestrator(
        evaluator=Evaluator(gateway),
        interviewer=Interviewer(gateway),
        repository=InMemoryInterviewRepository(),
        context_service=contexts,
        grounding=GroundingService(retriever),
    )

    started = time.perf_counter()
    print("=" * 78)
    print("INGESTION")
    print("=" * 78)
    context = await contexts.create(
        resume_text=SAMPLE_RESUME, job_description_text=SAMPLE_JOB
    )
    assert context.resume and context.job
    for claim in context.resume.claims:
        print(f"  claim: {claim.text}  [skills: {', '.join(claim.skills) or '-'}]")
    for req in context.job.requirements:
        print(f"  requirement: {req.skill} ({req.importance})")

    state = await orchestrator.create_interview(
        interview_type="jd_targeted", context_id=context.context_id
    )
    print("\nBLUEPRINT")
    for topic in state.blueprint.topics:
        grounded = f" <- {topic.evidence[0][:60]}..." if topic.evidence else ""
        print(f"  {topic.priority:>4}  {topic.key}  ({topic.source}){grounded}")

    print("\n" + "=" * 78)
    model = getattr(gateway, "model", "fake")
    print(f"INTERVIEW (provider={settings.llm_provider}, model={model})")
    print("=" * 78)
    question = await orchestrator.start(state.interview_id)
    print(f"\nQ1 [{question.topic}, difficulty {question.difficulty}]:\n  {question.text}")

    for index, answer in enumerate(CANNED_ANSWERS[:turns], start=1):
        print(f"\nA{index}: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        result = await orchestrator.submit_answer(state.interview_id, answer)
        evaluation = result.evaluation
        flag = "  ** DEGRADED **" if result.evaluation_degraded else ""
        print(
            f"   -> correctness={evaluation.correctness:g} depth={evaluation.depth:g} "
            f"communication={evaluation.communication:g}{flag}"
        )
        print(f"   -> missing: {evaluation.missing_concepts or '-'}")
        print(f"   -> misconceptions: {evaluation.misconceptions or '-'}")
        print(
            f"   -> recommended={evaluation.recommended_action} | "
            f"planner chose={result.plan.action} ({result.plan.reason})"
        )
        if result.next_question is None:
            print("\n   interview ended by the planner")
            break
        next_q = result.next_question
        print(
            f"\nQ{next_q.index} [{next_q.topic}, difficulty {next_q.difficulty}]:"
            f"\n  {next_q.text}"
        )

    await orchestrator.complete(state.interview_id)
    report = await orchestrator.get_report(state.interview_id)

    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"  overall: {report.overall_score}")
    for name, score in report.dimension_scores.items():
        print(f"  {name}: {score}")
    print(f"  weaknesses: {report.weaknesses}")
    print(f"  missed concepts: {report.missed_concepts}")
    print(f"  unaddressed target skills: {report.unaddressed_target_skills}")

    print("\n" + "=" * 78)
    print("DIAGNOSTICS")
    print("=" * 78)
    print(f"  wall clock: {time.perf_counter() - started:.1f}s")
    print(f"  structured-output retries: {len(diagnostics.retries)}")
    for line in diagnostics.retries:
        print(f"    {line[:140]}")
    if diagnostics.latencies_ms:
        print(
            f"  LLM latency ms: p50={statistics.median(diagnostics.latencies_ms):.0f} "
            f"max={max(diagnostics.latencies_ms)} calls={len(diagnostics.latencies_ms)}"
        )
    print(f"  tokens: in={diagnostics.tokens_in} out={diagnostics.tokens_out}")
    print(f"  errors: {len(diagnostics.errors)}")
    for line in diagnostics.errors:
        print(f"    {line[:140]}")

    # Things a human should eyeball, listed so they are not forgotten:
    print(
        "\n  MANUAL CHECKS:\n"
        "   - Did A1 (strong) clearly outscore A2 (vague) and A3 (misconception)?\n"
        "   - Was the ROC-AUC misconception in A3 caught in `misconceptions`?\n"
        "   - Did any question praise, tutor, or reveal reference notes?\n"
        "   - Did follow-ups reference the resume claims, not generic trivia?"
    )
    return 1 if (diagnostics.errors or retrieval_failures) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="rehearse with fakes, no keys")
    parser.add_argument("--turns", type=int, default=4, help="answers to submit (cost control)")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="stop after the retrieval probes"
    )
    arguments = parser.parse_args()
    sys.exit(asyncio.run(run(arguments.fake, arguments.turns, arguments.retrieval_only)))


if __name__ == "__main__":
    main()
