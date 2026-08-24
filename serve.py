"""Run the interview simulator with the right event loop on every platform.

    .venv/Scripts/python.exe serve.py            # default port 8000
    .venv/Scripts/python.exe serve.py --port 8001
    .venv/Scripts/python.exe serve.py --reload   # dev auto-reload

Exists because uvicorn's default Windows event loop (Proactor) breaks
psycopg's async mode; see app/loop.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prashniq")
    # Hosts like Render inject PORT and route to whatever binds it; HOST=0.0.0.0
    # is required inside containers. Local runs keep the loopback defaults.
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
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
