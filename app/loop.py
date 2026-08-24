"""Event-loop factory for uvicorn on Windows.

uvicorn >= 0.36 ignores the asyncio event-loop *policy* and builds its loop
from a factory; on Windows without --reload that factory is ProactorEventLoop,
which psycopg's async mode (the LangGraph checkpointer) cannot run on. Point
uvicorn here instead:

    uvicorn app.main:create_app --factory --loop app.loop:selector_loop

`serve.py` at the repo root does this for you.
"""

from __future__ import annotations

import asyncio


def selector_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop()
