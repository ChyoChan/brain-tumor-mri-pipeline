#!/usr/bin/env python3
"""Start the local MRI pipeline UI (default http://127.0.0.1:8765)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PORT = 8765


def port_number(value: str) -> int:
    """Accept only usable TCP port numbers."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    import uvicorn

    print(f"MRI reading room: http://127.0.0.1:{args.port}", flush=True)
    print("Press Ctrl+C in this terminal to stop.", flush=True)
    uvicorn.run(
        "app.server:app",
        host="127.0.0.1",
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
