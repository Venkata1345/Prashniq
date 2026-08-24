"""Interviews, candidate contexts and skill observations.

Revision ID: 0001
Revises:
Create Date: 2026-08-24

The rag_chunks vector table is NOT managed here yet -- its width depends on the
configured embedding model, and PgVectorStore.create_schema() owns it for now.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interviews",
        sa.Column("interview_id", sa.String(64), primary_key=True),
        sa.Column("candidate_id", sa.String(64), nullable=True),
        sa.Column("interview_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", JSONB, nullable=False),
    )
    op.create_index("ix_interviews_candidate_id", "interviews", ["candidate_id"])

    op.create_table(
        "candidate_contexts",
        sa.Column("context_id", sa.String(64), primary_key=True),
        sa.Column("candidate_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", JSONB, nullable=False),
    )
    op.create_index(
        "ix_candidate_contexts_candidate_id", "candidate_contexts", ["candidate_id"]
    )

    op.create_table(
        "skill_observations",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("candidate_id", sa.String(64), nullable=False),
        sa.Column("concept", sa.Text, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("interview_id", sa.String(64), nullable=False),
        sa.Column("interview_type", sa.String(64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "interview_id",
            "kind",
            "concept",
            name="uq_skill_observations_interview_concept",
        ),
    )
    op.create_index(
        "ix_skill_observations_candidate_id", "skill_observations", ["candidate_id"]
    )


def downgrade() -> None:
    op.drop_table("skill_observations")
    op.drop_table("candidate_contexts")
    op.drop_table("interviews")
