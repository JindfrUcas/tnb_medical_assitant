"""Guardrail 数据仓库实现层。

这里把上层节点依赖的查询能力分别映射到：
- Neo4j 图数据库
- 本地 `graph.json`

同时也补充了轻量 GraphRAG 运行时需要的指南图谱查询能力。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


class Neo4jGuardrailRepository:
    """基于 Neo4j 的红线规则仓库。"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """创建 Neo4j 驱动，并保存目标数据库名。"""
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """关闭数据库连接。"""
        self._driver.close()

    def contraindications_by_indicator(self, indicator_name: str) -> list[dict[str, Any]]:
        """查询某个指标相关的禁忌药物。"""
        query = """
        MATCH (d:Drug)-[r:CONTRAINDICATED_BY]->(i:Indicator)
        WHERE toLower(i.name) = toLower($indicator_name)
        RETURN d.name AS drug_name,
               i.name AS indicator_name,
               i.unit AS indicator_unit,
               r.operator AS operator,
               r.threshold AS threshold,
               r.reason AS reason
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(query, indicator_name=indicator_name)
            return [dict(item) for item in records]

    def excludes_by_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """查询某个疾病对应的排除药物。"""
        query = """
        MATCH (ds:Disease)-[:EXCLUDES]->(d:Drug)
        WHERE toLower(ds.name) = toLower($disease_name)
        RETURN ds.name AS disease_name, d.name AS drug_name
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(query, disease_name=disease_name)
            return [dict(item) for item in records]

    def adjustments_by_drug(self, drug_name: str) -> list[dict[str, Any]]:
        """查询某个药物的剂量调整策略。"""
        query = """
        MATCH (d:Drug)-[r:ADJUST_DOSAGE]->(s:DosageStrategy)
        WHERE toLower(d.name) = toLower($drug_name)
        RETURN d.name AS drug_name,
               r.condition AS condition,
               s.action AS action,
               s.note AS note
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(query, drug_name=drug_name)
            return [dict(item) for item in records]


class Neo4jGuidelineRepository:
    """基于 Neo4j 的指南证据图查询仓库。"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """创建独立的 Neo4j 驱动，用于查询 chunk 图谱与实体连边。"""
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """关闭数据库连接。"""
        self._driver.close()

    def list_entity_names(self) -> dict[str, list[str]]:
        """读取当前图中已有的药物 / 疾病 / 指标名称。

        运行时图检索不会让模型自由写 Cypher，而是先在 Python 侧做实体识别，
        再把识别结果作为受控参数传给固定查询模板。
        """
        queries = {
            "Drug": "MATCH (node:Drug) RETURN node.name AS name ORDER BY node.name",
            "Disease": "MATCH (node:Disease) RETURN node.name AS name ORDER BY node.name",
            "Indicator": "MATCH (node:Indicator) RETURN node.name AS name ORDER BY node.name",
        }
        results: dict[str, list[str]] = {"Drug": [], "Disease": [], "Indicator": []}

        with self._driver.session(database=self._database) as session:
            for label, query in queries.items():
                rows = session.run(query)
                results[label] = [str(item["name"]).strip() for item in rows if str(item["name"]).strip()]
        return results

    def fetch_linked_chunks(
        self,
        drug_names: list[str],
        disease_names: list[str],
        indicator_names: list[str],
        top_k: int = 4,
        source: str | None = None,
        min_hits: int = 1,
        exclude_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按实体集合查询已连边的指南 chunk。

        这一步仍然是固定模板查询，不存在“让大模型自己生成 Cypher”的过程。
        """
        query = """
        MATCH (chunk:GuidelineChunk)
        WHERE ($source IS NULL OR chunk.source = $source)
          AND (size($exclude_keys) = 0 OR NOT chunk.key IN $exclude_keys)
        OPTIONAL MATCH (drug:Drug)-[:MENTIONED_IN]->(chunk)
        WITH chunk, count(DISTINCT CASE WHEN drug.name IN $drug_names THEN drug END) AS drug_hits
        OPTIONAL MATCH (disease:Disease)-[:MENTIONED_IN]->(chunk)
        WITH chunk, drug_hits, count(DISTINCT CASE WHEN disease.name IN $disease_names THEN disease END) AS disease_hits
        OPTIONAL MATCH (indicator:Indicator)-[:MENTIONED_IN]->(chunk)
        WITH chunk, drug_hits, disease_hits,
             count(DISTINCT CASE WHEN indicator.name IN $indicator_names THEN indicator END) AS indicator_hits
        WITH chunk, drug_hits, disease_hits, indicator_hits,
             drug_hits + disease_hits + indicator_hits AS total_hits
        WHERE total_hits >= $min_hits
        RETURN chunk.key AS chunk_key,
               chunk.source AS source,
               chunk.content AS content,
               chunk.chapter AS chapter,
               chunk.section AS section,
               chunk.page_start AS page_start,
               chunk.page_end AS page_end,
               total_hits AS score
        ORDER BY total_hits DESC, coalesce(chunk.page_start, 0) ASC, chunk.chunk_index ASC
        LIMIT $top_k
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(
                query,
                drug_names=drug_names,
                disease_names=disease_names,
                indicator_names=indicator_names,
                top_k=max(1, top_k),
                source=source,
                min_hits=max(1, min_hits),
                exclude_keys=exclude_keys or [],
            )
            return [dict(item) for item in records]


class JsonGuardrailRepository:
    """基于本地 JSON 文件的红线规则仓库。"""

    def __init__(self, graph_json_path: Path):
        """读取本地 graph.json，并缓存为内存中的记录列表。"""
        payload = json.loads(graph_json_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError("graph.json must be a list")
        self._records: list[dict[str, Any]] = [item for item in payload if isinstance(item, dict)]

    def close(self) -> None:
        """与 Neo4j 版本保持相同接口；本地文件无需释放资源。"""
        return None

    def contraindications_by_indicator(self, indicator_name: str) -> list[dict[str, Any]]:
        """从本地 JSON 中查询某个指标相关禁忌。"""
        normalized = indicator_name.strip().lower()
        rows: list[dict[str, Any]] = []
        for record in self._records:
            drug_name = str(record.get("drug_name", "")).strip()
            for item in record.get("contraindications") or []:
                if not isinstance(item, dict):
                    continue
                current_name = str(item.get("indicator", "")).strip()
                if current_name.lower() != normalized:
                    continue
                rows.append(
                    {
                        "drug_name": drug_name,
                        "indicator_name": current_name,
                        "indicator_unit": str(item.get("unit", "")).strip(),
                        "operator": str(item.get("operator", "")).strip(),
                        "threshold": item.get("threshold"),
                        "reason": str(item.get("reason", "")).strip(),
                    }
                )
        return rows

    def excludes_by_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """从本地 JSON 中查询某个疾病的排除药物。"""
        normalized = disease_name.strip().lower()
        rows: list[dict[str, Any]] = []
        for record in self._records:
            drug_name = str(record.get("drug_name", "")).strip()
            for item in record.get("disease_excludes") or []:
                current_name = str(item).strip()
                if current_name.lower() != normalized:
                    continue
                rows.append({"disease_name": current_name, "drug_name": drug_name})
        return rows

    def adjustments_by_drug(self, drug_name: str) -> list[dict[str, Any]]:
        """从本地 JSON 中查询某个药物的剂量调整规则。"""
        normalized = drug_name.strip().lower()
        rows: list[dict[str, Any]] = []
        for record in self._records:
            current_drug = str(record.get("drug_name", "")).strip()
            if current_drug.lower() != normalized:
                continue
            for item in record.get("dosage_adjust") or []:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "drug_name": current_drug,
                        "condition": str(item.get("condition", "")).strip(),
                        "action": str(item.get("action", "")).strip(),
                        "note": str(item.get("note", "")).strip(),
                    }
                )
        return rows
