"""Application settings. Provider choice lives here, not in the domain."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Prashniq"
    log_level: str = "INFO"

    llm_provider: Literal["anthropic", "openai", "groq", "fake"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_timeout_seconds: float = 60.0
    llm_structured_attempts: int = 3
    anthropic_api_key: str | None = None
    # Groq: free tier (30 req/min, 6k tokens/min) — the public-demo provider.
    groq_api_key: str | None = None

    # --- Retrieval (Phase 3) -------------------------------------------------
    # Anthropic has no embeddings endpoint, so embeddings are the one place a
    # second provider appears. `fake` uses deterministic local vectors.
    embedding_provider: Literal["openai", "fake"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 96
    openai_api_key: str | None = None

    # pgvector is the production store. With no database configured the app
    # falls back to an in-process store so a local run needs no infrastructure.
    database_url: str | None = None
    vector_store: Literal["pgvector", "memory", "auto"] = "auto"
    seed_knowledge_base: bool = True
    # Calibrated against text-embedding-3-small (live run, 2026-08-24):
    # knowledge notes match true topics at 0.46+, off-topic noise reaches 0.31;
    # short resume claims live in a lower band. `retrieval_min_score` is the
    # fallback for anything not listed per collection.
    retrieval_min_score: float = 0.15
    knowledge_min_score: float = 0.35
    claim_min_score: float = 0.25

    @field_validator("database_url")
    @classmethod
    def _coerce_asyncpg_scheme(cls, url: str | None) -> str | None:
        """Accept hosted-Postgres URLs as pasted (Neon, Supabase, Heroku-style).

        SQLAlchemy picks its driver from the scheme; a bare postgresql:// would
        select the sync psycopg2 driver it can't use here.
        """
        if not url:
            return url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://") :]
        return url

    @property
    def resolved_vector_store(self) -> Literal["pgvector", "memory"]:
        if self.vector_store == "auto":
            return "pgvector" if self.database_url else "memory"
        return self.vector_store

    @property
    def asyncpg_database_url(self) -> str | None:
        """The database URL with libpq-style params translated for asyncpg.

        Hosted Postgres (Neon, Supabase) hands out URLs with `sslmode=require`
        and `channel_binding=require` — libpq/psycopg parameters that asyncpg
        rejects. asyncpg wants `ssl=require` and has no channel_binding knob.
        The psycopg checkpointer pool keeps the original URL untouched.
        """
        if not self.database_url:
            return None
        return normalize_asyncpg_url(self.database_url)


def normalize_asyncpg_url(url: str) -> str:
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    if not parts.query:
        return url
    query = []
    for key, value in parse_qsl(parts.query):
        if key == "sslmode":
            query.append(("ssl", value))
        elif key == "channel_binding":
            continue
        else:
            query.append((key, value))
    return urlunsplit(parts._replace(query=urlencode(query)))


@lru_cache
def get_settings() -> Settings:
    return Settings()
