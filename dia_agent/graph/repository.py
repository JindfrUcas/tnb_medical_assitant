"""Repository layer for structured guardrail knowledge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


class Neo4jGuardrailRepository:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def contraindications_by_indicator(self, indicator_name: str) -> list[dict[str, Any]]:
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
        query = """
        MATCH (ds:Disease)-[:EXCLUDES]->(d:Drug)
        WHERE toLower(ds.name) = toLower($disease_name)
        RETURN ds.name AS disease_name, d.name AS drug_name
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(query, disease_name=disease_name)
            return [dict(item) for item in records]

    def adjustments_by_drug(self, drug_name: str) -> list[dict[str, Any]]:
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
    def __init__(self, graph_json_path: Path):
        payload = json.loads(graph_json_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError("graph.json must be a list")
        self._records: list[dict[str, Any]] = [item for item in payload if isinstance(item, dict)]

    def close(self) -> None:
        return None

    def contraindications_by_indicator(self, indicator_name: str) -> list[dict[str, Any]]:
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
