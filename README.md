# AI/ML Technical Interview Simulator

An adaptive AI interview engine built on **LangGraph** (checkpointed interview
loop), **LangChain** (provider-agnostic models, embeddings and document
ingestion), **FastAPI**, **Postgres + pgvector** and **Alembic**.

**Phase 1** — a text-based adaptive interview loop.
**Phase 2** — resume and job-description ingestion, and an interview blueprint built from them.
**Phase 3** — retrieval: embeddings, pgvector, and a knowledge base that grounds evaluation.
**Phase 4** — Postgres persistence and a cross-interview candidate skill profile.
**Phase 4.5** — LangGraph/LangChain migration + textbook corpus pipeline.

No voice, no Redis, no auth yet — see the roadmap in
`ai_ml_interview_simulator_coding_agent_prompt.md`.

## Frontend

A React + Vite + Tailwind single-page app (`frontend/`), built to
`frontend/dist` and served by FastAPI at `/` — same origin as the API, so no
CORS, no proxy, no separate deployment. Setup (mode, optional candidate id,
optional resume/JD) -> chat-style interview -> report with an animated score
ring, dimension bars, strengths/weaknesses, per-question evidence, and the
cross-interview skill profile. Faithful to the product philosophy: no scores
are shown mid-interview. (The legacy no-build SPA in `app/static/` is served
as a fallback when no build exists.)

```bash
cd frontend && npm install && npm run build && cd ..   # once, and after UI changes
.venv/Scripts/python.exe serve.py                      # open http://127.0.0.1:8000
```

`serve.py` exists because uvicorn's default Windows event loop (Proactor)
breaks psycopg's async mode; it points uvicorn at `app.loop:selector_loop`.
For UI development with hot reload: `cd frontend && npm run dev` (proxies API
calls to :8000).

## The interview loop is a LangGraph state machine

```
START -> ask -> await_answer -> evaluate -+-> ask (loop)
                (interrupt)               +-> finish -> END
```

Each interview is a checkpointed graph thread (Postgres-backed in production):
the graph pauses at `await_answer` until the candidate replies, every step is
persisted, and **a crashed or restarted process resumes the interview exactly
where it stopped** — proven by restart tests against real Postgres.

The intelligence/control split survives the framework: `ask` and `evaluate`
call the model, while planning, difficulty, budgets and termination are the
same pure Python functions as before, now running inside graph nodes. LangGraph
supplies the machinery, not the decisions.

One LangChain adapter (`app/llm/langchain_gateway.py`) covers every chat
provider behind the existing `LLMGateway` interface — Anthropic, OpenAI, and
later BYOK providers are a config string. Same move for embeddings. Nothing
outside the adapters imports LangChain model classes.

## The knowledge corpus (drop folder + CLI)

```
corpus/
├── transformers/attention-book.pdf     -> topic "transformers"
├── rag/retrieval-notes.md              -> topic "rag"
└── ml-basics.txt                       -> topic "ml-basics"
```

```bash
.venv/Scripts/python -m scripts.ingest_corpus            # index new/changed files
.venv/Scripts/python -m scripts.ingest_corpus --rebuild  # wipe + re-seed + re-index
```

PDF/markdown/text → LangChain splitter → tagged chunks in the shared
`knowledge` collection, alongside the 18 curated notes. A hash manifest skips
unchanged files, chunk ids are content-addressed so re-runs never duplicate,
and one unreadable file never sinks the rest. `corpus/` is gitignored —
textbooks are not redistributable.

## Persistence + skill profile (Phase 4)

With `DATABASE_URL` set, interviews, candidate contexts, skill observations and
vectors are all durable; Alembic migrations run at startup. Without it,
everything is in-process — fine for local dev, and the log says so out loud.

Completing an interview folds it into the candidate's history as observations:
one per interviewed **topic** (correctness×0.7 + depth×0.3 over its trusted
turns — degraded evaluations are excluded) and one per evaluator **concept**.
The profile is a decayed aggregation computed in code:

```text
weight = 0.5 ^ (age_days / 90)        # an interview from 3 months ago
score  = weighted mean per skill      # counts half as much as one from today
```

`GET /candidates/{id}/profile` returns the roadmap's view — Transformers 8.1,
RAG 7.3 — plus `recommended_focus`, weakest first. That feeds back into the
blueprint: a topic matching a weak skill gets a +0.5 priority boost
(*practise what's weak*), so the next interview steers toward the gaps.
Recording is idempotent (unique per interview+skill, in memory and as a DB
constraint), anonymous interviews record nothing, and a profile-store outage
costs personalisation, never the interview.

The interview lock is now a Postgres advisory lock, so double-submitted
answers serialise even across multiple workers.

## Retrieval (Phase 3)

```
resume claims + role requirements  -> embedded -> pgvector (per candidate context)
curated knowledge notes            -> embedded -> pgvector (shared)

per turn:
  topic -> GroundingService -> resume claims the candidate wrote
                            -> reference points for the topic
                            -> interviewer + evaluator prompts
```

Three things retrieval buys that Phase 2's lexical matching could not:

- **Semantic claim matching.** A required skill now matches a claim that *means*
  the same thing. "FAISS HNSW recall tuning" finds "Tuned a FAISS HNSW index for
  recall at 50ms" and grants the blueprint's *claimed-on-the-resume* priority
  bonus, where token overlap would have missed it.
- **Grounded evaluation.** The evaluator is shown reference points for the topic
  and asked what the answer missed *relative to them*, so `missing_concepts` is
  measured against something written down rather than whatever the model
  recalled. The knowledge base ([knowledge.py](app/rag/knowledge.py)) is 18
  curated notes describing what a complete answer covers.
- **Sharper questions.** The interviewer receives the same notes with an
  explicit instruction not to state or hint at them — they tell it what to press
  on, not what to say.

**Everything degrades.** An embedding outage, a database outage or an unknown
topic costs grounding, never the interview: `Retriever` returns no results,
`GroundingService` falls back to blueprint evidence and then to lexical
matching, and the loop continues.

**Nothing is invented.** A topic with no relevant claim gets no claim, and a
topic outside the knowledge base gets no notes — the similarity thresholds
enforce that. They are **per collection**, calibrated against a live
`text-embedding-3-small` run (2026-08-24): knowledge notes match true topics at
0.46+ while off-topic noise reaches 0.31, so `KNOWLEDGE_MIN_SCORE=0.35`; short
resume claims score in a lower band, so `CLAIM_MIN_SCORE=0.25`. Claims are
embedded together with their extracted skill tags (display text stays the
candidate's own words) to bridge the vocabulary gap, and evidence lookup is
hybrid — semantic first, lexical overlap as the fallback. Re-run
`scripts/live_smoke.py` after changing the embedding model.

## Ingestion (Phase 2)

```
resume + job description
  -> chunked on structural boundaries               (code)
  -> claims and skill requirements extracted        (LLM, per chunk, validated)
  -> merged, then embedded and indexed              (code)
  -> interview blueprint: what to cover, in order   (code)
```

The blueprint scores every candidate topic before a question is asked:

| Signal | Priority |
| --- | --- |
| Must-have in the job description | 2.0 |
| Nice-to-have in the job description | 1.0 |
| ...and the candidate claims it (found by retrieval) | +0.5 |
| Strong resume claim the role never mentions (max 3) | 0.75 |
| Mode default topic (filler) | 0.25 |

> *"Built a RAG system using FAISS and FastAPI"* becomes
> *"What would you change about that FAISS index if the corpus grew 100x?"*

## The loop (Phase 1)

```
create interview
  -> orchestrator asks an opening question        (Interviewer, LLM + grounding)
  -> candidate answers
  -> answer is evaluated into validated JSON      (Evaluator, LLM + grounding)
  -> state is updated                             (code, pure functions)
  -> next action is chosen                        (Planner, code)
  -> adaptive follow-up is generated              (Interviewer, LLM)
  -> repeat until a budget/limit is hit
  -> deterministic report                         (Scoring, code)
```

The LLM has four narrow responsibilities: **extract resume claims**, **extract
role requirements**, **evaluate an answer**, **write a question**. Everything
else — topic planning and ordering, difficulty, follow-up depth, termination,
scoring — is ordinary Python, so it is testable and cannot be derailed by a bad
completion.

## Layout

```
app/
├── api/            HTTP routes + DTOs (thin; no interview logic)
├── static/         legacy no-build frontend (fallback when frontend/dist is absent)
├── context/
│   ├── ingestion.py   resume/JD -> validated structure (LLM)
│   ├── chunking.py    deterministic document splitting
│   ├── blueprint.py   topic plan and priorities; pluggable evidence lookup
│   ├── retrieval.py   lexical claim matching (fallback when RAG is down)
│   ├── service.py     ingest, index, store; raw document text is dropped
│   └── schemas.py     claims, requirements, blueprint
├── rag/
│   ├── embeddings.py        EmbeddingGateway ABC + deterministic test double
│   ├── langchain_embeddings.py  LangChain-backed embeddings adapter
│   ├── corpus.py            drop-folder pipeline: load, split, tag, index
│   ├── store.py             VectorStore protocol + in-memory implementation
│   ├── pgvector_store.py    production store (SQLAlchemy + pgvector)
│   ├── indexer.py           text -> chunks -> embeddings -> store
│   ├── retriever.py         query -> embedding -> nearest chunks (degrading)
│   ├── knowledge.py         curated reference notes + seeding
│   ├── lookup.py            vector-backed blueprint evidence lookup
│   └── factory.py           provider and store selection
├── core/config.py  settings, provider selection
├── db/
│   ├── models.py        relational schema (JSONB state docs + observation rows)
│   ├── repositories.py  Postgres repos + advisory interview lock
│   └── migrate.py       programmatic `alembic upgrade head` at startup
├── profile/
│   ├── schemas.py       observations, decay math, profile aggregation
│   ├── repository.py    observation store (memory + Postgres)
│   └── service.py       record on completion, aggregate on read
├── interview/
│   ├── graph.py         the interview loop as a checkpointed LangGraph graph
│   ├── orchestrator.py  facade over the graph; owns creation and reads
│   ├── grounding.py     the interview <-> retrieval seam
│   ├── evaluator.py     answer -> AnswerEvaluation (LLM)
│   ├── interviewer.py   plan -> QuestionDraft (LLM)
│   ├── planner.py       deterministic next-action decision
│   ├── scoring.py       deterministic report
│   ├── state.py         explicit state + pure transitions
│   ├── modes.py         interview modes as configuration
│   └── schemas.py       domain + LLM output schemas
├── llm/
│   ├── gateway.py            provider-agnostic ABC + structured-output retry
│   ├── langchain_gateway.py  LangChain-backed adapter (all chat providers)
│   ├── fake.py               deterministic gateway for tests and local dev
│   └── factory.py            provider selection
└── main.py         wiring
frontend/           React + Vite + Tailwind SPA; `npm run build` -> frontend/dist
migrations/         Alembic environment + versioned schema
scripts/
├── live_smoke.py      one real interview against live providers (see below)
└── ingest_corpus.py   index dropped textbooks into the knowledge collection
```

## Live smoke test

Everything in `tests/` runs against fakes, so the *prompts* are validated by a
separate one-command live run (a few cents):

```bash
export ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-...   # OpenAI optional
.venv/Scripts/python -m scripts.live_smoke        # --fake rehearses without keys
```

It runs ingestion → blueprint → four mixed-quality canned answers → report,
then prints diagnostics (structured-output retries, latencies, token usage)
and a manual-check list: did the strong answer outscore the vague one, was the
planted ROC-AUC misconception caught, did any question tutor or leak reference
notes. Run it after any prompt change.

## Running

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"

# zero setup: stubbed LLM, deterministic embeddings, in-process vector store
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake \
  .venv/Scripts/python.exe serve.py

# the real thing
docker compose up -d db
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... \
DATABASE_URL=postgresql+asyncpg://interview:interview@localhost:5432/interview \
  .venv/Scripts/python.exe serve.py

# on Windows, plain `uvicorn app.main:create_app --factory` WITHOUT --reload
# crashes at startup: uvicorn picks the Proactor event loop, which psycopg's
# async mode cannot use. serve.py pins the Selector loop (app/loop.py).
```

`app.main` exposes `create_app` rather than a module-level `app`: building one at
import time would turn a missing provider key into an import error.

```bash
.venv/Scripts/pytest                      # 157 tests, no network, no paid calls

# plus the 7 pgvector contract tests, the only ones needing infrastructure
docker compose up -d db
TEST_DATABASE_URL=postgresql+asyncpg://interview:interview@localhost:5432/interview \
  .venv/Scripts/pytest                    # all tests (pgvector + repository contracts)
```

The pgvector tests use their own table (`rag_chunks_test`), so they never touch
the table a running app uses; the repository tests truncate only the migrated
tables.

## API

| Method | Path                          | Purpose                                    |
| ------ | ----------------------------- | ------------------------------------------ |
| POST   | `/candidate-contexts`         | ingest a resume and/or job description     |
| GET    | `/candidate-contexts/{id}`    | extracted claims and requirements          |
| POST   | `/interviews`                 | create an interview                        |
| POST   | `/interviews/{id}/start`      | opening question (idempotent)              |
| POST   | `/interviews/{id}/answers`    | submit an answer, get the follow-up        |
| GET    | `/interviews/{id}`            | interview state                            |
| GET    | `/interviews/{id}/blueprint`  | what the interview plans to cover, and why |
| POST   | `/interviews/{id}/complete`   | end early                                  |
| GET    | `/interviews/{id}/report`     | final multi-dimensional report             |
| GET    | `/candidates/{id}/profile`    | decayed cross-interview skill profile      |
| GET    | `/interview-types`            | available modes                            |
| GET    | `/health`                     | provider, vector store and retrieval status |

`POST /interviews` accepts an optional `context_id`. The `resume_deep_dive` and
`jd_targeted` modes require one; every other mode uses it, when present, to
re-prioritise its standing topics.

`POST /answers` deliberately returns **only the next question** — interview
first, feedback later. Scores and coaching live in the report.

## Design notes

- **Two providers, two gateways.** `LLMGateway` (Anthropic) and
  `EmbeddingGateway` (OpenAI) are separate abstractions with separate adapters,
  because Anthropic ships no embeddings endpoint. Both have deterministic fakes,
  and no other module imports either SDK.
- **Structured output is validated, retried, then degraded.** The gateway feeds
  the validation error back to the model and retries; if it still fails, the
  evaluator records a neutral evaluation flagged `evaluation_degraded`, which
  scoring excludes.
- **`recommended_action` is advice.** The planner overrides it: it refuses to
  raise difficulty for a struggling candidate, caps follow-up depth, clamps
  difficulty to 1–5, and will not let the model end an interview early.
- **State is explicit**, not "the transcript". `InterviewState` carries topics,
  difficulty, concept scores, strengths/weaknesses, follow-up depth and the
  blueprint driving the interview.
- **Modes are data** (`modes.py`), not a growing `if` statement.
- **The plan is inspectable.** `GET /interviews/{id}/blueprint` shows each topic
  with its priority, source and grounding evidence.
- **Documents are not retained.** Only extracted claims and requirements are
  stored and indexed; logs record counts, never resume content.
- **Candidate data is partitioned** by `(collection, owner_id)` in the vector
  store, so one candidate's resume cannot surface in another's interview.
- **Embedding width is checked at startup.** `CREATE TABLE IF NOT EXISTS` will
  happily leave a table built for a different embedding width in place, after
  which every write fails with an opaque driver error. `create_schema()`
  compares the column's actual width against the configured dimensions and
  refuses to start with a message saying what to do.

## Known technical debt

- **`rag_chunks` is still outside Alembic.** (It now carries an HNSW index,
  created idempotently at startup.) Its column width depends on the
  configured embedding model, so `PgVectorStore.create_schema()` keeps owning it
  (with a startup width check). It should fold into migrations once the
  embedding choice settles.
- **`candidate_id` is client-asserted.** Anyone can read or write any profile by
  naming it. Harmless pre-auth, unacceptable after — Phase 7's authentication
  must bind it to a real identity.
- **Weak-skill matching is token overlap.** "vector databases" and "pgvector"
  don't match as a boost; concept normalisation (or embedding similarity) would
  fix the misses.
- Interview writes are **full-state upserts** per turn; fine at this size, but
  a long interview rewrites a growing JSONB document every answer.
- Candidate vectors and contexts have **no expiry or delete endpoint**;
  `delete_owner` exists but nothing calls it.
- Resumes are accepted as **text only** (PDF/DOCX parsing deferred).
- The knowledge base is a **Python literal**, not editable content.
- Embedding calls are **not cached**; chunk ids are content-addressed, so the
  cache key already exists.
- Token and embedding usage is **logged but not metered**.
