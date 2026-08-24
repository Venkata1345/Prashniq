"""Application settings. Provider choice lives here, not in the domain."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

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

    @property
    def resolved_vector_store(self) -> Literal["pgvector", "memory"]:
        if self.vector_store == "auto":
            return "pgvector" if self.database_url else "memory"
        return self.vector_store


@lru_cache
def get_settings() -> Settings:
    return Settings()
