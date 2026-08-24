"""Alembic environment (async engine).

The database URL comes from, in order: migrate.py's programmatic override,
the DATABASE_URL environment variable, then alembic.ini's fallback.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.models import metadata as target_metadata

config = context.config
if config.config_file_name is not None:
    # keep the host app's loggers (e.g. uvicorn.error) alive when migrations
    # run inside the server's startup — otherwise startup errors vanish
    fileConfig(config.config_file_name, disable_existing_loggers=False)

if os.environ.get("DATABASE_URL") and not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    from app.core.config import normalize_asyncpg_url

    engine = create_async_engine(normalize_asyncpg_url(config.get_main_option("sqlalchemy.url")))
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
