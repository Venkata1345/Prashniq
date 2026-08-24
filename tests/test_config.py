"""Hosted-Postgres URL handling: what users paste from Neon/Supabase must work."""

from __future__ import annotations

from app.core.config import Settings, normalize_asyncpg_url


def _settings(url: str) -> Settings:
    return Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store="memory",
        database_url=url,
    )


class TestDatabaseUrlCoercion:
    def test_bare_postgresql_scheme_gets_the_asyncpg_driver(self) -> None:
        settings = _settings("postgresql://u:p@host/db")
        assert settings.database_url == "postgresql+asyncpg://u:p@host/db"

    def test_heroku_style_postgres_scheme_is_upgraded(self) -> None:
        settings = _settings("postgres://u:p@host/db")
        assert settings.database_url == "postgresql+asyncpg://u:p@host/db"

    def test_explicit_asyncpg_url_is_untouched(self) -> None:
        settings = _settings("postgresql+asyncpg://u:p@host/db")
        assert settings.database_url == "postgresql+asyncpg://u:p@host/db"


class TestAsyncpgParamNormalization:
    def test_neon_url_params_are_translated(self) -> None:
        url = "postgresql+asyncpg://u:p@host/db?sslmode=require&channel_binding=require"
        assert normalize_asyncpg_url(url) == "postgresql+asyncpg://u:p@host/db?ssl=require"

    def test_url_without_query_is_untouched(self) -> None:
        url = "postgresql+asyncpg://u:p@host/db"
        assert normalize_asyncpg_url(url) == url

    def test_full_paste_from_neon_resolves_through_settings(self) -> None:
        settings = _settings("postgresql://u:p@ep.neon.tech/db?sslmode=require&channel_binding=require")
        assert settings.asyncpg_database_url == "postgresql+asyncpg://u:p@ep.neon.tech/db?ssl=require"
        # The psycopg checkpointer pool keeps libpq params (it understands them).
        assert "sslmode=require" in settings.database_url.replace("+asyncpg", "")
