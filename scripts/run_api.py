#!/usr/bin/env python3
"""Start FastAPI server for Dia-Agent."""

from __future__ import annotations

from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.config import get_settings


def main() -> int:
    settings = get_settings()
    uvicorn.run(
        "dia_agent.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
