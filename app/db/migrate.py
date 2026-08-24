"""Programmatic Alembic upgrade, run at startup when a database is configured.

Synchronous by design: the async env.py starts its own event loop, so callers
inside a running loop must use `asyncio.to_thread(upgrade_to_head, url)`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_config(database_url: str) -> Config:
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade_to_head(database_url: str) -> None:
    command.upgrade(_build_config(database_url), "head")
    logger.info("migrations_applied target=head")
