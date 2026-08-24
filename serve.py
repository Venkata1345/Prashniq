"""Run the interview simulator with the right event loop on every platform.

    .venv/Scripts/python.exe serve.py            # default port 8000
    .venv/Scripts/python.exe serve.py --port 8001
    .venv/Scripts/python.exe serve.py --reload   # dev auto-reload

Exists because uvicorn's default Windows event loop (Proactor) breaks
psycopg's async mode; see app/loop.py.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI interview simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        kwargs["loop"] = "app.loop:selector_loop"

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        **kwargs,
    )


if __name__ == "__main__":
    main()
