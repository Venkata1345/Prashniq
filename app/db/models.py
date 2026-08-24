"""Relational schema for interviews, candidate contexts and skill observations.

Interviews and contexts are stored as documents: one JSONB column holding the
Pydantic-serialised state, plus a few extracted columns for lookup. The state
models are the domain's source of truth and already versioned by this repo;
normalising them into rows now would freeze their shape prematurely.

Skill observations are the exception -- they exist to be aggregated across
interviews, so they are real rows.

Migrations own this schema (see `migrations/`); the definitions here are the
code-side mirror the repositories query through.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

interviews = Table(
    "interviews",
    metadata,
    Column("interview_id", String(64), primary_key=True),
    Column("candidate_id", String(64), nullable=True),
    Column("interview_type", String(64), nullable=False),
    Column("status", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("state", JSONB, nullable=False),
    Index("ix_interviews_candidate_id", "candidate_id"),
)

candidate_contexts = Table(
    "candidate_contexts",
    metadata,
    Column("context_id", String(64), primary_key=True),
    Column("candidate_id", String(64), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("data", JSONB, nullable=False),
    Index("ix_candidate_contexts_candidate_id", "candidate_id"),
)

skill_observations = Table(
    "skill_observations",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("candidate_id", String(64), nullable=False),
    Column("concept", Text, nullable=False),
    # "topic" observations are coarse interview areas (what the roadmap's
    # "Transformers 8.1" is); "concept" observations are the evaluator's
    # fine-grained concept scores.
    Column("kind", String(16), nullable=False),
    Column("score", Float, nullable=False),
    Column("interview_id", String(64), nullable=False),
    Column("interview_type", String(64), nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    # Completing the same interview twice must not double-count.
    UniqueConstraint(
        "interview_id", "kind", "concept", name="uq_skill_observations_interview_concept"
    ),
    Index("ix_skill_observations_candidate_id", "candidate_id"),
)
