#!/usr/bin/env python3
"""Start the local MRI pipeline UI at http://127.0.0.1:8000"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
