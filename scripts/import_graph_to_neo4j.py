#!/usr/bin/env python3
"""把 Dia-Agent 的图谱数据批量导入 Neo4j。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase


DEFAULT_INPUT = Path("dataset/graph.json")
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_DATABASE = "neo4j"


CREATE_CONSTRAINTS = (
    """
    CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
    FOR (node:Drug) REQUIRE node.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
    FOR (node:Disease) REQUIRE node.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT indicator_key_unique IF NOT EXISTS
    FOR (node:Indicator) REQUIRE node.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT dosage_strategy_key_unique IF NOT EXISTS
    FOR (node:DosageStrategy) REQUIRE node.key IS UNIQUE
    """,
)


MERGE_DRUGS = """
UNWIND $rows AS row
MERGE (:Drug {name: row.name})
"""

MERGE_DISEASES = """
UNWIND $rows AS row
MERGE (:Disease {name: row.name})
"""

MERGE_INDICATORS = """
UNWIND $rows AS row
MERGE (indicator:Indicator {key: row.key})
SET indicator.name = row.name,
    indicator.unit = row.unit
"""

MERGE_STRATEGIES = """
UNWIND $rows AS row
MERGE (strategy:DosageStrategy {key: row.key})
SET strategy.description = row.description,
    strategy.action = row.action,
    strategy.note = row.note
"""

MERGE_CONTRAINDICATIONS = """
UNWIND $rows AS row
MATCH (drug:Drug {name: row.drug_name})
MATCH (indicator:Indicator {key: row.indicator_key})
MERGE (drug)-[:CONTRAINDICATED_BY {
    operator: row.operator,
    threshold: row.threshold,
    reason: row.reason
}]->(indicator)
"""

MERGE_EXCLUDES = """
UNWIND $rows AS row
MATCH (disease:Disease {name: row.disease_name})
MATCH (drug:Drug {name: row.drug_name})
MERGE (disease)-[:EXCLUDES]->(drug)
"""

MERGE_ADJUSTMENTS = """
UNWIND $rows AS row
MATCH (drug:Drug {name: row.drug_name})
MATCH (strategy:DosageStrategy {key: row.strategy_key})
MERGE (drug)-[:ADJUST_DOSAGE {condition: row.condition}]->(strategy)
"""


@dataclass
class GraphRows:
    """保存导入前整理好的各类节点与边数据。"""

    drugs: list[dict[str, Any]]
    indicators: list[dict[str, Any]]
    diseases: list[dict[str, Any]]
    strategies: list[dict[str, Any]]
    contraindications: list[dict[str, Any]]
    excludes: list[dict[str, Any]]
    adjustments: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """解析 Neo4j 导入脚本参数。"""
    parser = argparse.ArgumentParser(
        description="Import Dia-Agent graph.json into Neo4j with schema-safe MERGE operations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to graph.json. Defaults to dataset/graph.json.",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", DEFAULT_URI),
        help="Neo4j URI. Defaults to $NEO4J_URI or bolt://localhost:7687.",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", DEFAULT_USER)),
        help="Neo4j username. Defaults to $NEO4J_USERNAME or neo4j.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password. Defaults to $NEO4J_PASSWORD.",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE),
        help="Neo4j database name. Defaults to $NEO4J_DATABASE or neo4j.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of rows per UNWIND batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate graph.json without connecting to Neo4j.",
    )
    parser.add_argument(
        "--skip-constraints",
        action="store_true",
        help="Skip CREATE CONSTRAINT IF NOT EXISTS statements.",
    )
    return parser.parse_args()


def load_json_documents(path: Path) -> list[dict[str, Any]]:
    """读取一个或多个连续 JSON 文档，并统一展开成对象列表。"""
    text = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    offset = 0
    documents: list[dict[str, Any]] = []

    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break

        payload, offset = decoder.raw_decode(text, offset)
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("Top-level array entries must be objects.")
                documents.append(item)
        elif isinstance(payload, dict):
            documents.append(payload)
        else:
            raise ValueError("Top-level JSON values must be objects or arrays of objects.")

    return documents


def normalize_text(value: Any, field_name: str, allow_empty: bool = False) -> str:
    """把字段标准化成字符串，并校验必填约束。"""
    if value is None:
        if allow_empty:
            return ""
        raise ValueError(f"Missing required field: {field_name}")
    text = str(value).strip()
    if text or allow_empty:
        return text
    raise ValueError(f"Empty required field: {field_name}")


def ensure_list(value: Any, field_name: str) -> list[Any]:
    """确保某个字段为列表，不存在时返回空列表。"""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field {field_name} must be a list.")
    return value


def normalize_threshold(value: Any) -> int | float | str:
    """把阈值字段标准化成数值或原始字符串。"""
    if isinstance(value, (int, float)):
        return value
    text = normalize_text(value, "threshold")
    try:
        parsed = float(text)
    except ValueError:
        return text
    return int(parsed) if parsed.is_integer() else parsed


def indicator_key(name: str, unit: str) -> str:
    """生成指标节点的稳定唯一键。"""
    return f"{name}::{unit}"


def strategy_description(action: str, note: str) -> str:
    """把动作与备注拼成剂量策略描述。"""
    return action if not note else f"{action}；{note}"


def build_rows(records: list[dict[str, Any]]) -> GraphRows:
    """把原始 graph.json 记录整理成适合批量导入的行数据。"""
    drug_names: set[str] = set()
    disease_names: set[str] = set()
    indicator_rows: dict[str, dict[str, Any]] = {}
    strategy_rows: dict[str, dict[str, Any]] = {}
    contraindication_rows: set[tuple[Any, ...]] = set()
    exclude_rows: set[tuple[str, str]] = set()
    adjustment_rows: set[tuple[str, str, str]] = set()

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record #{index} must be an object.")

        drug_name = normalize_text(record.get("drug_name"), f"drug_name at record #{index}")
        drug_names.add(drug_name)

        for item in ensure_list(record.get("contraindications"), f"contraindications at record #{index}"):
            if not isinstance(item, dict):
                raise ValueError(f"Contraindication at record #{index} must be an object.")
            name = normalize_text(item.get("indicator"), f"indicator at record #{index}")
            unit = normalize_text(item.get("unit"), f"unit at record #{index}", allow_empty=True)
            key = indicator_key(name, unit)
            indicator_rows[key] = {"key": key, "name": name, "unit": unit}
            contraindication_rows.add(
                (
                    drug_name,
                    key,
                    normalize_text(item.get("operator"), f"operator at record #{index}"),
                    normalize_threshold(item.get("threshold")),
                    normalize_text(item.get("reason"), f"reason at record #{index}"),
                )
            )

        for disease in ensure_list(record.get("disease_excludes"), f"disease_excludes at record #{index}"):
            disease_name = normalize_text(disease, f"disease_excludes at record #{index}")
            disease_names.add(disease_name)
            exclude_rows.add((disease_name, drug_name))

        for item in ensure_list(record.get("dosage_adjust"), f"dosage_adjust at record #{index}"):
            if not isinstance(item, dict):
                raise ValueError(f"Dosage adjustment at record #{index} must be an object.")
            condition = normalize_text(item.get("condition"), f"condition at record #{index}")
            action = normalize_text(item.get("action"), f"action at record #{index}")
            note = normalize_text(item.get("note"), f"note at record #{index}", allow_empty=True)
            description = strategy_description(action, note)
            strategy_key = description
            strategy_rows[strategy_key] = {
                "key": strategy_key,
                "description": description,
                "action": action,
                "note": note,
            }
            adjustment_rows.add((drug_name, strategy_key, condition))

    return GraphRows(
        drugs=[{"name": name} for name in sorted(drug_names)],
        indicators=[indicator_rows[key] for key in sorted(indicator_rows)],
        diseases=[{"name": name} for name in sorted(disease_names)],
        strategies=[strategy_rows[key] for key in sorted(strategy_rows)],
        contraindications=[
            {
                "drug_name": drug_name,
                "indicator_key": indicator_key_value,
                "operator": operator,
                "threshold": threshold,
                "reason": reason,
            }
            for drug_name, indicator_key_value, operator, threshold, reason in sorted(
                contraindication_rows,
                key=lambda row: (row[0], row[1], row[2], str(row[3]), row[4]),
            )
        ],
        excludes=[
            {"disease_name": disease_name, "drug_name": drug_name}
            for disease_name, drug_name in sorted(exclude_rows)
        ],
        adjustments=[
            {"drug_name": drug_name, "strategy_key": strategy_key_value, "condition": condition}
            for drug_name, strategy_key_value, condition in sorted(adjustment_rows)
        ],
    )


def batched(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """按批大小切分待导入数据。"""
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def execute_batches(session: Any, query: str, rows: list[dict[str, Any]], batch_size: int, label: str) -> None:
    """按批执行 Cypher 导入语句，并打印进度。"""
    total = len(rows)
    if total == 0:
        print(f"{label}: 0 rows")
        return

    for batch_number, batch in enumerate(batched(rows, batch_size), start=1):
        session.run(query, rows=batch).consume()
        print(f"{label}: imported batch {batch_number} ({len(batch)}/{total})")


def create_constraints(session: Any) -> None:
    """确保导入所需的唯一约束已存在。"""
    for statement in CREATE_CONSTRAINTS:
        session.run(statement).consume()


def print_summary(rows: GraphRows) -> None:
    """打印本次导入的数据规模摘要。"""
    print("Schema summary")
    print(f"  Drug nodes: {len(rows.drugs)}")
    print(f"  Indicator nodes: {len(rows.indicators)}")
    print(f"  Disease nodes: {len(rows.diseases)}")
    print(f"  DosageStrategy nodes: {len(rows.strategies)}")
    print(f"  CONTRAINDICATED_BY edges: {len(rows.contraindications)}")
    print(f"  EXCLUDES edges: {len(rows.excludes)}")
    print(f"  ADJUST_DOSAGE edges: {len(rows.adjustments)}")


def main() -> int:
    """执行导入流程，必要时连接 Neo4j 并批量写入。"""
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = load_json_documents(args.input)
    rows = build_rows(records)
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
            execute_batches(session, MERGE_DRUGS, rows.drugs, args.batch_size, "Drug")
            execute_batches(session, MERGE_DISEASES, rows.diseases, args.batch_size, "Disease")
            execute_batches(session, MERGE_INDICATORS, rows.indicators, args.batch_size, "Indicator")
            execute_batches(session, MERGE_STRATEGIES, rows.strategies, args.batch_size, "DosageStrategy")
            execute_batches(
                session,
                MERGE_CONTRAINDICATIONS,
                rows.contraindications,
                args.batch_size,
                "CONTRAINDICATED_BY",
            )
            execute_batches(session, MERGE_EXCLUDES, rows.excludes, args.batch_size, "EXCLUDES")
            execute_batches(
                session,
                MERGE_ADJUSTMENTS,
                rows.adjustments,
                args.batch_size,
                "ADJUST_DOSAGE",
            )
    finally:
        driver.close()

    print("Neo4j import complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
