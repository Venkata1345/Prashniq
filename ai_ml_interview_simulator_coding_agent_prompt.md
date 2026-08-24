# Coding Agent Prompt — AI/ML Technical Interview Simulator

You are a senior AI engineer and product-minded backend architect helping me build a production-quality **AI/ML Technical Interview Simulator** as a real SaaS product.

Your job is not to generate the whole project at once. Work incrementally, preserve architectural quality, and explain important design decisions briefly as you implement them.

## Product Goal

Build an adaptive AI interview platform for candidates preparing for AI/ML Engineer roles.

The platform should eventually support:

- AI/ML fundamentals interviews
- Deep learning and transformers
- RAG and agents
- AI/ML system design
- Resume deep-dive interviews
- Job-description-targeted interviews
- Behavioral interviews
- Voice-based interviews
- Structured scoring and feedback
- Interview history and skill progression
- User accounts
- Free demo usage with quota limits
- Bring Your Own Key (BYOK) provider support
- Usage metering and provider selection

The core differentiator is **adaptive interviewing**.

The system should not simply ask a fixed list of questions.

After each candidate answer, it should:

1. Evaluate the answer.
2. Detect what was covered correctly.
3. Detect missing concepts.
4. Detect misconceptions.
5. Estimate technical depth.
6. Decide whether to:
   - probe deeper
   - ask for clarification
   - challenge an assumption
   - ask for a tradeoff
   - increase difficulty
   - decrease difficulty
   - change topic
   - end the topic
7. Generate the next question accordingly.

## Core Architecture

Use this architecture as the target direction:

```text
Frontend
  ↓
FastAPI Backend
  ↓
Interview Orchestrator
  ├── Interviewer
  ├── Evaluator
  ├── Interview State
  ├── Interview Planner
  ├── Scoring Engine
  └── LLM Gateway
          ↓
       LLM Provider

Later:
  ├── RAG
  ├── Resume/JD ingestion
  ├── pgvector
  ├── Redis
  ├── Voice STT/TTS
  ├── WebSockets
  ├── Provider Gateway / BYOK
  ├── Usage quotas and metering
  └── Background jobs
```

Do not introduce infrastructure before it has a real need.

## Important Engineering Principle

Prefer:

```text
LLMs for intelligence
Code for control and determinism
```

Do NOT build an unnecessary multi-agent system.

For example, do not create 10 agents just because the application uses AI.

The initial architecture should be deterministic Python orchestration around a small number of well-defined LLM responsibilities.

## Initial MVP

Build the MVP as a **text-only interview engine**.

The first complete workflow should be:

```text
Start interview
↓
Generate first question
↓
Candidate submits answer
↓
Evaluator analyzes answer
↓
Evaluator returns validated structured output
↓
Interview state is updated
↓
Orchestrator decides next action
↓
Interviewer generates adaptive follow-up
↓
Repeat
↓
Generate final interview report
```

Do NOT add voice, Redis, authentication, BYOK, usage quotas, or frontend complexity until this core loop works well.

## Suggested Backend Stack

Use:

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL eventually
- pytest
- async code where appropriate

Keep provider-specific LLM code behind an abstraction.

Example:

```python
class LLMGateway:
    async def generate(self, ...):
        ...

    async def generate_structured(self, schema, ...):
        ...
```

The rest of the application should not directly depend on a specific LLM provider.

Design this gateway so it can later support both:

```text
App-managed provider credentials
and
User-provided BYOK credentials
```

Provider selection, authentication material, retries, and usage accounting should stay outside the interview domain logic.

## Suggested Project Structure

Use a structure close to:

```text
app/
├── api/
│   └── interviews.py
│
├── interview/
│   ├── orchestrator.py
│   ├── interviewer.py
│   ├── evaluator.py
│   ├── planner.py
│   ├── scoring.py
│   ├── state.py
│   └── schemas.py
│
├── llm/
│   ├── gateway.py
│   └── schemas.py
│
├── db/
│   ├── models.py
│   ├── repository.py
│   └── session.py
│
├── core/
│   └── config.py
│
└── main.py

tests/
├── test_evaluator.py
├── test_orchestrator.py
└── test_interview_flow.py
```

You may improve this structure if there is a clear reason.

## Interview State

Model interview state explicitly.

It should eventually track fields such as:

```text
interview_id
candidate_id
interview_type
difficulty
current_topic
topics_covered
questions_asked
candidate_answers
strengths
weaknesses
concept_scores
follow_up_depth
elapsed_time
remaining_time
status
```

Do not treat the entire conversation history as the only form of memory.

The application should maintain explicit structured state.

## Evaluator

The evaluator must return validated structured data.

Create a Pydantic schema similar to:

```python
class AnswerEvaluation(BaseModel):
    correctness: float
    depth: float
    communication: float

    concepts_covered: list[str]
    missing_concepts: list[str]
    misconceptions: list[str]

    recommended_action: Literal[
        "probe_deeper",
        "clarify",
        "challenge_assumption",
        "ask_tradeoff",
        "increase_difficulty",
        "decrease_difficulty",
        "change_topic",
        "end_topic",
        "end_interview",
    ]

    follow_up_topic: str | None
    rationale: str
```

Scores should have bounded ranges and validation.

Do not accept arbitrary malformed LLM output.

Handle invalid structured output with:

```text
validation
→ retry if appropriate
→ fallback/error handling
```

## Orchestrator

The orchestrator should own interview control.

Example conceptual flow:

```python
evaluation = await evaluator.evaluate(
    question=current_question,
    answer=candidate_answer,
    state=state,
)

state = update_state(state, evaluation)

action = choose_next_action(
    evaluation=evaluation,
    state=state,
)

next_question = await interviewer.generate_question(
    action=action,
    state=state,
)
```

Do not allow the LLM itself to directly mutate application state.

## Interviewer

The interviewer should behave like a realistic technical interviewer.

It should:

- ask concise questions
- ask relevant follow-ups
- challenge vague answers
- ask why
- ask for tradeoffs
- increase difficulty when appropriate
- avoid immediately revealing the correct answer
- avoid excessive praise
- not behave like a tutor during the interview
- remain professional and realistic

Example:

Candidate says:

```text
I would use a vector database for retrieval.
```

Bad follow-up:

```text
Great answer! Vector databases are useful because...
```

Good follow-up:

```text
What would determine your choice between HNSW and IVF for this workload?
```

## Interview Modes

Design the system so new modes can be added cleanly.

Eventually support:

```text
ml_fundamentals
deep_learning
transformers
rag_agents
ml_system_design
resume_deep_dive
behavioral
```

Do not hardcode all mode logic into a giant conditional function.

Use configuration or strategy-like abstractions where appropriate.

## Scoring

Do not return only an arbitrary score such as:

```text
7/10
```

Use multiple dimensions.

For example:

```text
technical_correctness
technical_depth
reasoning
tradeoff_awareness
communication
```

For system design interviews, possible dimensions include:

```text
requirements_clarification
architecture
scalability
reliability
tradeoff_reasoning
ml_ai_understanding
communication
```

Final reports should include:

```text
overall_score
dimension_scores
strengths
weaknesses
evidence
missed_concepts
recommended_topics
```

## API Design

Start with APIs similar to:

```text
POST /interviews
POST /interviews/{id}/start
POST /interviews/{id}/answers
GET  /interviews/{id}
POST /interviews/{id}/complete
GET  /interviews/{id}/report
```

Keep routes thin.

Business logic should live in services/orchestrators, not in FastAPI route handlers.

## Testing Requirements

Testing is mandatory.

At minimum include:

### Unit tests

- evaluator schema validation
- state transitions
- next-action logic
- score calculations
- invalid LLM responses

### Integration test

Simulate:

```text
start interview
→ receive question
→ submit answer
→ receive evaluation
→ receive adaptive follow-up
```

Use a fake/mock LLM gateway so tests are deterministic.

Do not make normal test runs depend on paid external LLM calls.

## Future Phases

Do not implement these unless the current phase requires them, but keep the design extensible.

### Phase 2 — Resume + JD

Later add:

```text
resume upload
→ parsing
→ chunking
→ embeddings
→ retrieval
```

and:

```text
job description
→ skill extraction
→ target skill map
→ interview blueprint
```

The interview should eventually generate questions based on resume claims such as:

```text
Built a RAG system using FAISS and FastAPI.
```

Possible follow-up:

```text
Why did you choose FAISS, and what would you change if the corpus grew 100x?
```

### Phase 3 — RAG

Use retrieval for:

- resume context
- job-description context
- interview knowledge
- evaluation grounding

Use pgvector initially unless there is a strong reason for a separate vector database.

### Phase 4 — Persistent Candidate Skill Profile

Track performance across interviews.

Example:

```text
Transformers        8.1
RAG                 7.3
ML System Design    5.9
MLOps               5.5
```

Use this later to personalize practice sessions.

### Phase 5 — Voice

Voice should initially be modular:

```text
Microphone
→ STT
→ existing interview engine
→ TTS
→ audio
```

Do not rewrite interview logic around voice.

The interview engine should remain modality-independent.

Later support:

- streaming STT
- streaming TTS
- WebSockets
- VAD
- interruptions/barge-in
- turn detection
- realtime latency metrics

### Phase 6 — Redis

Only add Redis when there is a concrete reason, such as:

- shared active interview state across workers
- rate limiting
- temporary realtime session data
- caching

PostgreSQL remains the source of truth.

### Phase 7 — Free Demo + BYOK

Do not add payments initially.

The product should support two usage modes:

```text
Free Demo Mode
→ app-managed provider credentials
→ strict quota / rate limits
→ frictionless onboarding

BYOK Mode
→ user provides their own API key
→ provider chosen through a common gateway
→ interview engine remains provider-agnostic
```

Implement:

- authentication if needed for quota enforcement
- free demo quotas
- rate limiting
- provider selection
- Bring Your Own Key (BYOK)
- provider-specific adapters behind one gateway
- usage metering
- graceful provider fallback where appropriate

Prefer session-only BYOK keys initially.

Do not permanently store raw user API keys unless there is a clear product need. If saved keys are introduced later, they must be encrypted at rest, never logged, and scoped to the owning user.

Track:

```text
LLM input tokens
LLM output tokens
STT minutes
TTS usage
interview duration
provider used
estimated cost per interview
free-tier quota consumption
```

A future paid tier can be added only if real usage justifies it. Do not design the MVP around Stripe or subscriptions.

## Reliability

Build toward handling:

- LLM timeout
- malformed structured output
- provider error
- database error
- duplicate requests
- user reconnect
- interrupted interview
- model fallback

Avoid silent failures.

Use useful logging around:

```text
interview_id
request_id
model
latency
token usage
evaluation failure
fallback usage
```

Do not log sensitive resume contents unnecessarily.

## Performance

Do not prematurely optimize.

But design so we can eventually measure:

```text
P50/P95 LLM latency
evaluation latency
time to next question
structured-output success rate
token usage
cost per interview
interview completion rate
```

Later for voice:

```text
speech end → response start latency
time to first audio
turn detection errors
premature interruption rate
```

## Development Rules

When implementing a feature:

1. Inspect the existing code first.
2. Explain the smallest architectural change needed.
3. Implement it.
4. Add or update tests.
5. Run the relevant tests.
6. Fix failures before moving on.
7. Summarize what changed and why.
8. Mention any technical debt introduced.

Avoid giant rewrites unless clearly justified.

Do not generate placeholder abstractions that have no current use.

Do not add dependencies without explaining why they are needed.

Prefer readable production code over clever code.

Use type hints.

Use clear domain names.

Keep functions focused.

Avoid god classes.

Avoid global mutable state.

## Important Product Philosophy

The product must feel like a realistic interviewer, not ChatGPT tutoring the candidate.

During an interview:

```text
interview first
feedback later
```

The interviewer should not constantly explain answers or encourage the user after every response.

Detailed teaching belongs in the final feedback report.

## Current Task

Start by implementing only the first milestone:

```text
Text-based adaptive interview loop
```

Required behavior:

1. User starts an AI/ML interview.
2. System generates an initial technical question.
3. Candidate submits an answer.
4. Evaluator produces validated structured evaluation.
5. Interview state updates.
6. Orchestrator chooses the next action.
7. Interviewer produces a targeted follow-up.
8. Tests verify this flow.

Before writing code, inspect the current repository and give a short implementation plan based on what already exists.

Then implement the milestone incrementally.
