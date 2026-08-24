# HANDOFF — AI/ML Interview Simulator

Session handoff, 2026-08-24. Read this first, then `README.md` (user-facing
docs) and `ai_ml_interview_simulator_coding_agent_prompt.md` (the original
roadmap the project follows).

## What this project is

An adaptive AI interview platform for AI/ML engineering roles. The candidate
answers in text; after each answer the system grades it, updates explicit
state, and a deterministic planner decides the next move (probe deeper,
change topic, adjust difficulty, end). Feedback only at the end. Core
principle from the roadmap, enforced everywhere: **LLMs for intelligence,
code for control** — the model only (a) extracts resume claims, (b) extracts
JD requirements, (c) grades an answer into a validated Pydantic schema,
(d) writes one question. Planning, scoring, budgets, termination = pure Python.

## Current state: COMPLETE through Phase 4.5 + frontend

- **Phase 1** – adaptive interview loop (evaluator/interviewer/planner/scoring/state)
- **Phase 2** – resume + JD ingestion -> `InterviewBlueprint` (priority-scored
  topic plan; must-have 2.0, nice-to-have 1.0, +0.5 claimed-on-resume,
  +0.5 weak-in-profile, 0.75 resume-only, 0.25 mode filler)
- **Phase 3** – RAG: pgvector + OpenAI embeddings, 18 curated knowledge notes,
  grounding service feeding interviewer + evaluator prompts
- **Phase 4** – Postgres persistence (Alembic), cross-interview skill profile
  (90-day half-life decay), advisory-lock turn serialisation
- **Phase 4.5** – LangGraph/LangChain migration + textbook corpus:
  - Interview loop = LangGraph `StateGraph` (ask -> await_answer(interrupt) ->
    evaluate -> ask|finish), Postgres-checkpointed; **crash/restart resumes
    mid-interview** (proven live)
  - `LangChainGateway` (init_chat_model) replaced hand-written provider
    adapters; `LangChainEmbeddingGateway` same for embeddings
  - `corpus/` drop-folder + `scripts/ingest_corpus.py` (hash manifest,
    content-addressed chunk ids)
- **Frontend** – React + Vite + Tailwind v4 SPA in `frontend/` (dark theme,
  chat bubbles, animated score ring/bars), built to `frontend/dist`, served by
  FastAPI at `/` -> `/ui/`. Rebuild after UI edits: `cd frontend && npm run
  build`. Legacy no-build SPA in `app/static/` remains as fallback when no
  dist exists. `npm run dev` in frontend/ = hot reload, proxies API to :8000.
- **Corpus is LOADED**: 16,233 chunks in pgvector from 16 legally-free sources
  (ISL-Python, ESL, Bishop PRML, MML, Murphy, UDL Prince, d2l, LBDL,
  Jurafsky&Martin Aug-2026 draft, Foundations of LLMs, Attention paper, RAG
  survey, Weng+Anthropic agent essays, Rules of ML, Hidden Tech Debt).
  Probe result: 8/8 interview-style queries hit the right book pages.

**Tests: 209 passing (`pytest`), +16 more with
`TEST_DATABASE_URL=postgresql+asyncpg://interview:interview@localhost:5432/interview`
(pgvector + Postgres repo contract tests). All tests use fakes; no network.**

## Environment (this machine)

- Windows 11, Git Bash + PowerShell. **NOT a git repo yet** (user was offered
  `git init`, hasn't said yes — recommend again).
- venv at `.venv/` (Python 3.11): fastapi, pydantic v2, langchain 1.3 /
  langgraph 1.2 / langchain-{openai,anthropic,text-splitters},
  langgraph-checkpoint-postgres, psycopg[binary], sqlalchemy+asyncpg,
  pgvector, alembic, openai, anthropic, pypdf, pytest(-asyncio), httpx.
- `.env` exists (gitignored): `OPENAI_API_KEY` (real), `LLM_PROVIDER=openai`,
  `LLM_MODEL`, `DATABASE_URL=postgresql+asyncpg://interview:interview@localhost:5432/interview`.
  **User has NO Anthropic key** — interview brain runs on OpenAI (gpt-4o-mini
  default via `resolve_model`). **Groq is wired as a third provider**
  (`LLM_PROVIDER=groq` + `GROQ_API_KEY`, default model
  llama-3.3-70b-versatile; free tier 30 req/min / 6k tok/min / 14.4k req/day)
  for the planned public demo — Groq has NO embeddings, so retrieval needs
  OpenAI embeddings or runs degraded.
- Postgres via `docker compose up -d db` (pgvector/pgvector:pg16). Container
  name `ai-interview-simulator-db-1`. May be running or stopped; volume
  persists. DB currently holds: migrated tables (interviews,
  candidate_contexts, skill_observations, alembic_version), langgraph
  checkpoint tables, `rag_chunks` (16,233 knowledge chunks, HNSW-indexed,
  real text-embedding-3-small vectors) + `rag_chunks_test`. `interviews`
  table has leftover test rows (cand-42, cand-lg, browser-demo…) — harmless.

## Run / verify commands

```bash
.venv/Scripts/python.exe -m pytest -q                      # 209 tests, no cost
.venv/Scripts/python.exe serve.py                          # app at :8000 (NOT bare uvicorn — see gotcha 8)
.venv/Scripts/python.exe -m scripts.live_smoke             # live prompt validation (~cents; --fake = free rehearsal)
.venv/Scripts/python.exe -m scripts.ingest_corpus          # index new corpus files (manifest skips unchanged)
.venv/Scripts/python.exe -m scripts.probe_corpus           # retrieval quality probes vs real store
```

## Critical gotchas (each cost real debugging time)

1. **Windows + psycopg async**: needs `WindowsSelectorEventLoopPolicy` — set at
   import in `app/main.py` and in scripts that touch the checkpointer. asyncpg
   is fine either way.
2. **Bash heredocs mangle backslash escapes on this setup**: writing Python
   containing `\x00`-style escapes via `python - <<'PY'` once wrote REAL
   control bytes into a source file. Use the Write/Edit tools for any content
   with escape sequences.
3. **PDF extraction emits NUL bytes** (Postgres TEXT rejects) and **duplicate
   page-header chunks** (breaks multi-row upsert). Both fixed:
   `clean_text()` in `app/rag/corpus.py`, batch dedupe in
   `PgVectorStore.upsert`. Don't regress.
4. **Tests must not read `.env`**: always construct
   `Settings(llm_provider="fake", embedding_provider="fake",
   vector_store="memory", database_url=None)` in tests.
5. **`AsyncPostgresSaver` requires a running event loop** — hence
   `checkpointer_factory` + lazy graph compilation in the orchestrator.
6. **Never mix fake and real embeddings in one pgvector collection** (fake
   vectors bury real ones). Verification runs against the DB should use
   `EMBEDDING_PROVIDER=fake DATABASE_URL= VECTOR_STORE=memory` overrides.
7. Retrieval thresholds are **live-calibrated** for text-embedding-3-small:
   `KNOWLEDGE_MIN_SCORE=0.35` (noise ceiling 0.31, true hits 0.46+),
   `CLAIM_MIN_SCORE=0.25`. Claims are embedded WITH skill tags
   (`embedding_text` vs display `text`). Recalibrate via
   `live_smoke --retrieval-only` if the embedding model changes.
8. **Bare `uvicorn app.main:create_app --factory` WITHOUT `--reload` crashes
   at startup on Windows (exit 3, no error printed)**: uvicorn >= 0.36 builds
   its loop from a factory (Proactor on win32), IGNORING the event-loop policy
   set in `app/main.py`; psycopg async then dies in `checkpointer_pool.open()`.
   With `--reload` it happened to work (subprocess -> Selector). Fix shipped:
   run `serve.py` (pins `--loop app.loop:selector_loop`). The error was
   invisible because alembic's `fileConfig` silenced uvicorn's logger — now
   fixed with `disable_existing_loggers=False` in `migrations/env.py`.
9. Port 8000 may be occupied by the user's other project (`nexus-api` docker
   container). Check `docker ps` before blaming the app; `serve.py --port 8001`.

## Architecture map (where things live)

- `app/interview/graph.py` — the LangGraph loop; nodes close over deps
- `app/interview/orchestrator.py` — facade (create/start/submit/complete/report);
  repo stays system-of-record, checkpointer holds graph position
- `app/interview/{planner,scoring,state}.py` — pure logic; planner OVERRIDES
  the evaluator's `recommended_action` (difficulty guards, depth cap ≤
  `mode.max_follow_up_depth`, min-questions before model may end)
- `app/interview/grounding.py` — interview<->RAG seam; blueprint evidence
  first, vector retrieval second, lexical fallback third; failures degrade,
  never raise
- `app/llm/` + `app/rag/` — gateways (ABCs + LangChain adapters + fakes);
  only adapters import provider/framework SDKs
- `app/context/` — ingestion (chunk->extract->merge), blueprint builder
  (pluggable async `EvidenceLookup`)
- `app/profile/` — skill observations (topic 0.7·correctness+0.3·depth,
  concept scores), decayed aggregation, feeds `focus_skills` back into
  blueprint
- `app/db/` — JSONB-document repos + advisory `lock()`, Alembic runner
  (`migrations/versions/0001…`); `rag_chunks` is OUTSIDE Alembic (owned by
  `PgVectorStore.create_schema`, includes startup dimension check + HNSW)
- `app/main.py` — wiring; providers/stores chosen by Settings; lifespan runs
  migrations, checkpointer setup, schema, knowledge seeding
- Degradation ladder everywhere: bad structured output -> retry w/ error
  feedback -> flagged neutral eval (excluded from scoring); retrieval/profile
  outage -> lose grounding/personalisation only.

## What's NOT done (agreed next steps, in rough priority)

1. **git init + first commit + GitHub** (user wants a portfolio piece; offered,
   pending their yes). Remember: never commit `corpus/` or `.env` (gitignored).
2. **Full live interview through the frontend against the real corpus** — user
   should click through http://127.0.0.1:8000; visual polish never verified.
3. **Phase 5 (voice)**: STT -> engine -> TTS, modality-independent engine ready.
4. **Phase 7 (demo/BYOK/auth)**: `candidate_id` is client-asserted — profile
   endpoints are open; auth must bind identity. BYOK slots into
   `LangChainGateway` via config.
5. Debt list at README bottom: rag_chunks->Alembic, per-turn full-state JSONB
   upserts, no vector expiry/delete endpoint, knowledge notes are a Python
   literal, no embedding cache, usage logged-not-metered, weak-skill matching
   is token overlap.

## User preferences (IMPORTANT)

- **Explain in simple English** — saved in memory
  (`prefers-simple-english.md`). Lead with plain-language summaries; jargon
  killed earlier sessions' readability. Technical depth second.
- Decisive: says "run it / start / do it" — proceed, don't over-ask; but ask
  before spending more than a few cents or adding infrastructure.
- Cares about: industry-standard signal (LangChain/LangGraph chosen for
  resume keywords + defensible judgment), landing interviews, low cost.
- Uses OpenAI key only; total spend so far ≈ $0.50 (corpus embedding + smoke
  tests). Corpus re-ingestion is ~free (manifest).
