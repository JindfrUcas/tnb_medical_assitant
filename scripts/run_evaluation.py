#!/usr/bin/env python3
"""Wrapper script to run Dia-Agent evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.evaluation.run import main


if __name__ == "__main__":
    raise SystemExit(main())
