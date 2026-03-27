"""Guardrail 数据仓库实现层。

这里把上层节点依赖的查询能力分别映射到：
- Neo4j 图数据库
- 本地 `graph.json`
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
