#!/usr/bin/env python3
"""Run one consultation from CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.pipeline import DiaAgentPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dia-Agent consultation")
    parser.add_argument("--raw-input", type=str, default="")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--rag-query", type=str, default="")
    parser.add_argument("--history", type=str, default="")
    return parser.parse_args()


def parse_payload(raw_text: str) -> str | dict:
    text = raw_text.strip()
    if not text:
        return {
            "indicators": {"eGFR": 28, "HbA1c": 9.1},
            "diseases": ["糖尿病酮症酸中毒"],
            "current_drugs": ["盐酸二甲双胍片 (普通片/缓释片)"],
        }
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return text


def main() -> int:
    args = parse_args()
    if args.input_file:
        raw_text = args.input_file.read_text(encoding="utf-8")
    else:
        raw_text = args.raw_input

    payload = parse_payload(raw_text)

    image_paths = [path for path in args.image if path.exists() and path.is_file()]
    if image_paths:
        if isinstance(payload, dict) and {"indicators", "diseases", "current_drugs"}.issubset(payload.keys()):
            consult_payload = payload
        elif isinstance(payload, dict):
            consult_payload = dict(payload)
            consult_payload["image_paths"] = [str(path) for path in image_paths]
            if args.history:
                consult_payload["history_text"] = args.history
        else:
            consult_payload = {
                "text": payload,
                "image_paths": [str(path) for path in image_paths],
                "history_text": args.history,
            }
    else:
        consult_payload = payload

    pipeline = DiaAgentPipeline()
    try:
        result = pipeline.consult(consult_payload, rag_query=args.rag_query, history_text=args.history)
    finally:
        pipeline.close()

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
