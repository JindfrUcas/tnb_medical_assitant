#!/usr/bin/env python3
"""把指南 chunk 与证据连边批量导入 Neo4j。

这一步是当前项目轻量 GraphRAG 的核心建图脚本：
1. 复用现有 PDF 切分逻辑生成结构化 chunk。
2. 把文档 / 章节 / 小节 / chunk 导入到 Neo4j。
3. 用规则图中的 Drug / Disease / Indicator 名称去连到 chunk。

脚本是增量导入的，默认使用 `MERGE`，不会破坏已有 Guardrail 图结构。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.config import get_settings
from dia_agent.graph.evidence_linking import build_entity_alias_index, extract_entity_matches
from dia_agent.rag.indexer import RagIndexer


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_DATABASE = "neo4j"

CREATE_CONSTRAINTS = (
    """
    CREATE CONSTRAINT guideline_document_key_unique IF NOT EXISTS
    FOR (node:GuidelineDocument) REQUIRE node.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT guideline_chapter_key_unique IF NOT EXISTS
    FOR (node:GuidelineChapter) REQUIRE node.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT guideline_section_key_unique IF NOT EXISTS
    FOR (node:GuidelineSection) REQUIRE node.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT guideline_chunk_key_unique IF NOT EXISTS
    FOR (node:GuidelineChunk) REQUIRE node.key IS UNIQUE
    """,
)

MERGE_DOCUMENTS = """
UNWIND $rows AS row
MERGE (doc:GuidelineDocument {key: row.key})
SET doc.name = row.name,
    doc.source = row.source,
    doc.file_name = row.file_name
"""

MERGE_CHAPTERS = """
UNWIND $rows AS row
MERGE (chapter:GuidelineChapter {key: row.key})
SET chapter.name = row.name,
    chapter.source = row.source,
    chapter.document_key = row.document_key
"""

MERGE_SECTIONS = """
UNWIND $rows AS row
MERGE (section:GuidelineSection {key: row.key})
SET section.name = row.name,
    section.source = row.source,
    section.document_key = row.document_key,
    section.chapter_key = row.chapter_key
"""

MERGE_CHUNKS = """
UNWIND $rows AS row
MERGE (chunk:GuidelineChunk {key: row.key})
SET chunk.source = row.source,
    chunk.file_name = row.file_name,
    chunk.document_key = row.document_key,
    chunk.chapter = row.chapter,
    chunk.section = row.section,
    chunk.chunk_type = row.chunk_type,
    chunk.chunk_index = row.chunk_index,
    chunk.page_start = row.page_start,
    chunk.page_end = row.page_end,
    chunk.content = row.content
"""

MERGE_DOCUMENT_CHAPTERS = """
UNWIND $rows AS row
MATCH (doc:GuidelineDocument {key: row.document_key})
MATCH (chapter:GuidelineChapter {key: row.chapter_key})
MERGE (doc)-[:HAS_CHAPTER]->(chapter)
"""

MERGE_CHAPTER_SECTIONS = """
UNWIND $rows AS row
MATCH (chapter:GuidelineChapter {key: row.chapter_key})
MATCH (section:GuidelineSection {key: row.section_key})
MERGE (chapter)-[:HAS_SECTION]->(section)
"""

MERGE_SECTION_CHUNKS = """
UNWIND $rows AS row
MATCH (section:GuidelineSection {key: row.section_key})
MATCH (chunk:GuidelineChunk {key: row.chunk_key})
MERGE (section)-[:HAS_CHUNK]->(chunk)
"""

MERGE_DRUG_LINKS = """
UNWIND $rows AS row
MATCH (drug:Drug {name: row.entity_name})
MATCH (chunk:GuidelineChunk {key: row.chunk_key})
MERGE (drug)-[:MENTIONED_IN]->(chunk)
"""

MERGE_DISEASE_LINKS = """
UNWIND $rows AS row
MATCH (disease:Disease {name: row.entity_name})
MATCH (chunk:GuidelineChunk {key: row.chunk_key})
MERGE (disease)-[:MENTIONED_IN]->(chunk)
"""

MERGE_INDICATOR_LINKS = """
UNWIND $rows AS row
MATCH (indicator:Indicator {name: row.entity_name})
MATCH (chunk:GuidelineChunk {key: row.chunk_key})
MERGE (indicator)-[:MENTIONED_IN]->(chunk)
"""


@dataclass
class GuidelineRows:
    """保存本次指南入图需要写入的全部节点和边。"""

    documents: list[dict[str, Any]]
    chapters: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    document_chapters: list[dict[str, Any]]
    chapter_sections: list[dict[str, Any]]
    section_chunks: list[dict[str, Any]]
    drug_links: list[dict[str, Any]]
    disease_links: list[dict[str, Any]]
    indicator_links: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Import guideline chunks and entity links into Neo4j.")
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Path to guideline PDF. Repeat --pdf for multiple files.",
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=settings.graph_json_path,
        help="Path to graph.json used for entity alias construction.",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", settings.neo4j_uri or DEFAULT_URI),
        help="Neo4j URI.",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", settings.neo4j_user or DEFAULT_USER)),
        help="Neo4j username.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", settings.neo4j_password or ""),
        help="Neo4j password.",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", settings.neo4j_database or DEFAULT_DATABASE),
        help="Neo4j database name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of rows per UNWIND batch.",
    )
    parser.add_argument(
        "--skip-constraints",
        action="store_true",
        help="Skip CREATE CONSTRAINT IF NOT EXISTS statements.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse PDF and build row summary without writing Neo4j.",
    )
    return parser.parse_args()


def load_graph_entities(graph_json_path: Path) -> tuple[set[str], set[str], set[str]]:
    """从 graph.json 中提取药物 / 疾病 / 指标名称集合。"""
    payload = json.loads(graph_json_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("graph.json must be a list")

    drug_names: set[str] = set()
    disease_names: set[str] = set()
    indicator_names: set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue
        drug_name = str(item.get("drug_name", "")).strip()
        if drug_name:
            drug_names.add(drug_name)

        for disease in item.get("disease_excludes") or []:
            disease_name = str(disease).strip()
            if disease_name:
                disease_names.add(disease_name)

        for contraindication in item.get("contraindications") or []:
            if not isinstance(contraindication, dict):
                continue
            indicator_name = str(contraindication.get("indicator", "")).strip()
            if indicator_name:
                indicator_names.add(indicator_name)

    return drug_names, disease_names, indicator_names


def build_guideline_rows(
    documents: list[Any],
    drug_names: set[str],
    disease_names: set[str],
    indicator_names: set[str],
) -> GuidelineRows:
    """把 LangChain Document 列表转换成适合 Neo4j 批量导入的结构。"""
    document_rows: dict[str, dict[str, Any]] = {}
    chapter_rows: dict[str, dict[str, Any]] = {}
    section_rows: dict[str, dict[str, Any]] = {}
    chunk_rows: list[dict[str, Any]] = []
    document_chapter_rows: set[tuple[str, str]] = set()
    chapter_section_rows: set[tuple[str, str]] = set()
    section_chunk_rows: list[dict[str, Any]] = []

    drug_link_rows: set[tuple[str, str]] = set()
    disease_link_rows: set[tuple[str, str]] = set()
    indicator_link_rows: set[tuple[str, str]] = set()

    alias_index = build_entity_alias_index(
        drug_names=drug_names,
        disease_names=disease_names,
        indicator_names=indicator_names,
    )

    for document in documents:
        metadata = dict(document.metadata)
        source = str(metadata.get("source", "UNKNOWN")).strip() or "UNKNOWN"
        file_name = str(metadata.get("file_name", source)).strip() or source
        document_key = file_name

        chapter_name = str(metadata.get("chapter", "未分类章节")).strip() or "未分类章节"
        section_name = str(metadata.get("section", chapter_name)).strip() or chapter_name
        chunk_index = int(metadata.get("chunk_index", 0))
        chunk_key = f"{document_key}::{chunk_index}"
        chapter_key = f"{document_key}::{chapter_name}"
        section_key = f"{chapter_key}::{section_name}"

        document_rows[document_key] = {
            "key": document_key,
            "name": source,
            "source": source,
            "file_name": file_name,
        }
        chapter_rows[chapter_key] = {
            "key": chapter_key,
            "name": chapter_name,
            "source": source,
            "document_key": document_key,
        }
        section_rows[section_key] = {
            "key": section_key,
            "name": section_name,
            "source": source,
            "document_key": document_key,
            "chapter_key": chapter_key,
        }
        chunk_rows.append(
            {
                "key": chunk_key,
                "source": source,
                "file_name": file_name,
                "document_key": document_key,
                "chapter": chapter_name,
                "section": section_name,
                "chunk_type": str(metadata.get("chunk_type", "detail")).strip() or "detail",
                "chunk_index": chunk_index,
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "content": document.page_content,
            }
        )
        document_chapter_rows.add((document_key, chapter_key))
        chapter_section_rows.add((chapter_key, section_key))
        section_chunk_rows.append({"section_key": section_key, "chunk_key": chunk_key})

        matches = extract_entity_matches(document.page_content, alias_index)
        for entity_name in matches["Drug"]:
            drug_link_rows.add((entity_name, chunk_key))
        for entity_name in matches["Disease"]:
            disease_link_rows.add((entity_name, chunk_key))
        for entity_name in matches["Indicator"]:
            indicator_link_rows.add((entity_name, chunk_key))

    return GuidelineRows(
        documents=[document_rows[key] for key in sorted(document_rows)],
        chapters=[chapter_rows[key] for key in sorted(chapter_rows)],
        sections=[section_rows[key] for key in sorted(section_rows)],
        chunks=sorted(chunk_rows, key=lambda row: (row["file_name"], row["chunk_index"])),
        document_chapters=[
            {"document_key": document_key, "chapter_key": chapter_key}
            for document_key, chapter_key in sorted(document_chapter_rows)
        ],
        chapter_sections=[
            {"chapter_key": chapter_key, "section_key": section_key}
            for chapter_key, section_key in sorted(chapter_section_rows)
        ],
        section_chunks=section_chunk_rows,
        drug_links=[
            {"entity_name": entity_name, "chunk_key": chunk_key}
            for entity_name, chunk_key in sorted(drug_link_rows)
        ],
        disease_links=[
            {"entity_name": entity_name, "chunk_key": chunk_key}
            for entity_name, chunk_key in sorted(disease_link_rows)
        ],
        indicator_links=[
            {"entity_name": entity_name, "chunk_key": chunk_key}
            for entity_name, chunk_key in sorted(indicator_link_rows)
        ],
    )


def batched(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """把批量写入数据按固定大小切块。"""
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def execute_batches(session: Any, query: str, rows: list[dict[str, Any]], batch_size: int, label: str) -> None:
    """统一执行一组 Cypher 批量导入语句。"""
    total = len(rows)
    if total == 0:
        print(f"{label}: 0 rows")
        return

    for batch_number, batch in enumerate(batched(rows, batch_size), start=1):
        session.run(query, rows=batch).consume()
        print(f"{label}: imported batch {batch_number} ({min(batch_number * batch_size, total)}/{total})")


def create_constraints(session: Any) -> None:
    """确保指南图谱需要的唯一约束已存在。"""
    for statement in CREATE_CONSTRAINTS:
        session.run(statement).consume()


def print_summary(rows: GuidelineRows) -> None:
    """打印本次导入的规模摘要。"""
    print("Guideline graph summary")
    print(f"  GuidelineDocument nodes: {len(rows.documents)}")
    print(f"  GuidelineChapter nodes: {len(rows.chapters)}")
    print(f"  GuidelineSection nodes: {len(rows.sections)}")
    print(f"  GuidelineChunk nodes: {len(rows.chunks)}")
    print(f"  HAS_CHAPTER edges: {len(rows.document_chapters)}")
    print(f"  HAS_SECTION edges: {len(rows.chapter_sections)}")
    print(f"  HAS_CHUNK edges: {len(rows.section_chunks)}")
    print(f"  Drug->MENTIONED_IN edges: {len(rows.drug_links)}")
    print(f"  Disease->MENTIONED_IN edges: {len(rows.disease_links)}")
    print(f"  Indicator->MENTIONED_IN edges: {len(rows.indicator_links)}")


def main() -> int:
    """执行指南 GraphRAG 入图流程。"""
    args = parse_args()
    settings = get_settings()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if not args.graph_json.exists():
        raise FileNotFoundError(f"graph.json not found: {args.graph_json}")

    pdf_paths = args.pdf or settings.guideline_pdf_paths
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    drug_names, disease_names, indicator_names = load_graph_entities(args.graph_json)
    indexer = RagIndexer(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        collection_name=settings.chroma_collection,
        embedding_device=settings.embedding_device,
    )
    documents = indexer.extract_documents(pdf_paths)
    rows = build_guideline_rows(documents, drug_names, disease_names, indicator_names)
    print_summary(rows)

    if args.dry_run:
        print("Dry run complete. No Neo4j writes executed.")
        return 0

    if not args.password:
        raise ValueError("Neo4j password is required. Pass --password or set NEO4J_PASSWORD.")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session(database=args.database) as session:
            if not args.skip_constraints:
                create_constraints(session)
                print("Constraints ensured.")
            execute_batches(session, MERGE_DOCUMENTS, rows.documents, args.batch_size, "GuidelineDocument")
            execute_batches(session, MERGE_CHAPTERS, rows.chapters, args.batch_size, "GuidelineChapter")
            execute_batches(session, MERGE_SECTIONS, rows.sections, args.batch_size, "GuidelineSection")
            execute_batches(session, MERGE_CHUNKS, rows.chunks, args.batch_size, "GuidelineChunk")
            execute_batches(session, MERGE_DOCUMENT_CHAPTERS, rows.document_chapters, args.batch_size, "HAS_CHAPTER")
            execute_batches(session, MERGE_CHAPTER_SECTIONS, rows.chapter_sections, args.batch_size, "HAS_SECTION")
            execute_batches(session, MERGE_SECTION_CHUNKS, rows.section_chunks, args.batch_size, "HAS_CHUNK")
            execute_batches(session, MERGE_DRUG_LINKS, rows.drug_links, args.batch_size, "Drug->MENTIONED_IN")
            execute_batches(
                session,
                MERGE_DISEASE_LINKS,
                rows.disease_links,
                args.batch_size,
                "Disease->MENTIONED_IN",
            )
            execute_batches(
                session,
                MERGE_INDICATOR_LINKS,
                rows.indicator_links,
                args.batch_size,
                "Indicator->MENTIONED_IN",
            )
    finally:
        driver.close()

    print("Guideline graph import complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
