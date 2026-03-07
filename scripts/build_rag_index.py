#!/usr/bin/env python3
"""Build ChromaDB index from diabetes guideline PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.config import get_settings
from dia_agent.rag.indexer import RagIndexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG index for Dia-Agent")
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Path to guideline PDF. Repeat --pdf for multiple files.",
    )
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    pdf_paths = args.pdf or [
        Path("dataset/中国1型糖尿病诊治指南（2021版）.pdf"),
        Path("dataset/dc26sint.pdf"),
    ]
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    indexer = RagIndexer(
        persist_dir=args.persist_dir or settings.chroma_persist_dir,
        embedding_model=args.embedding_model or settings.embedding_model,
        collection_name=args.collection or settings.chroma_collection,
    )
    chunks = indexer.build(pdf_paths, reset=args.reset)
    print(f"RAG index build complete. Chunks: {chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
