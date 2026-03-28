"""Guardrail 数据仓库实现层。

这里把上层节点依赖的查询能力分别映射到：
- Neo4j 图数据库
- 本地 `graph.json`

同时也补充了轻量 GraphRAG 运行时需要的指南图谱查询能力。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


class Neo4jGuardrailRepository:
    """基于 Neo4j 的红线规则仓库。"""

    def __init__(self, uri: str = "", user: str = "", password: str = "", database: str = "neo4j", *, driver=None):
        """创建或复用 Neo4j 驱动。传入 driver 时复用外部连接池。"""
        self._database = database
        self._driver = driver or GraphDatabase.driver(uri, auth=(user, password))
        self._owns_driver = driver is None

    def close(self) -> None:
        """仅在自己创建 driver 时关闭连接。"""
        if self._owns_driver:
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

    def __init__(self, uri: str = "", user: str = "", password: str = "", database: str = "neo4j", *, driver=None):
        """创建或复用 Neo4j 驱动。传入 driver 时复用外部连接池。"""
        self._database = database
        self._driver = driver or GraphDatabase.driver(uri, auth=(user, password))
        self._owns_driver = driver is None

    def close(self) -> None:
        """仅在自己创建 driver 时关闭连接。"""
        if self._owns_driver:
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

    def traverse_from_entity(
        self,
        entity_name: str,
        entity_type: str = "Drug",
        max_hops: int = 2,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """从一个实体出发做 1-2 跳图遍历。"""
        result: dict[str, Any] = {"chunks": [], "siblings": [], "related_drugs": []}

        query_chunks = """
        MATCH (e)-[:MENTIONED_IN]->(chunk:GuidelineChunk)
        WHERE e.name = $entity_name AND $entity_type IN labels(e)
        RETURN chunk.key AS chunk_key,
               chunk.source AS source,
               chunk.content AS content,
               chunk.chapter AS chapter,
               chunk.section AS section,
               chunk.chunk_index AS chunk_index
        ORDER BY chunk.page_start ASC
        LIMIT $top_k
        """
        with self._driver.session(database=self._database) as session:
            rows = session.run(query_chunks, entity_name=entity_name, entity_type=entity_type, top_k=top_k)
            result["chunks"] = [dict(r) for r in rows]

        if max_hops < 2 or not result["chunks"]:
            return result

        first_chunk_key = result["chunks"][0]["chunk_key"]
        query_siblings = """
        MATCH (chunk:GuidelineChunk {key: $chunk_key})
              <-[:HAS_CHUNK]-(section:GuidelineSection)
              -[:HAS_CHUNK]->(sibling:GuidelineChunk)
        WHERE sibling.key <> $chunk_key
        RETURN sibling.key AS chunk_key,
               sibling.source AS source,
               sibling.content AS content,
               sibling.chapter AS chapter,
               sibling.section AS section,
               sibling.chunk_index AS chunk_index
        ORDER BY sibling.chunk_index ASC
        LIMIT $top_k
        """
        with self._driver.session(database=self._database) as session:
            rows = session.run(query_siblings, chunk_key=first_chunk_key, top_k=top_k)
            result["siblings"] = [dict(r) for r in rows]

        if entity_type == "Drug":
            query_related = """
            MATCH (d:Drug {name: $entity_name})-[r:CONTRAINDICATED_BY]->(i:Indicator)
                  <-[r2:CONTRAINDICATED_BY]-(other:Drug)
            WHERE other.name <> $entity_name
            RETURN d.name AS source_drug,
                   i.name AS shared_indicator,
                   r.operator AS operator,
                   r.threshold AS threshold,
                   r.reason AS reason,
                   collect(DISTINCT other.name) AS related_drugs
            """
            with self._driver.session(database=self._database) as session:
                rows = session.run(query_related, entity_name=entity_name)
                result["related_drugs"] = [dict(r) for r in rows]

        return result

    def expand_subgraph(
        self,
        entity_names: list[str],
        min_overlap: int = 2,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """给定一组实体，提取它们共同关联的 chunk 和实体间关系路径。"""
        result: dict[str, Any] = {"shared_chunks": [], "relations": []}

        query_shared = """
        UNWIND $entity_names AS ename
        MATCH (e)-[:MENTIONED_IN]->(chunk:GuidelineChunk)
        WHERE e.name = ename
        WITH chunk, collect(DISTINCT e.name) AS mentioned_entities,
             count(DISTINCT e) AS entity_count
        WHERE entity_count >= $min_overlap
        RETURN chunk.key AS chunk_key,
               chunk.source AS source,
               chunk.content AS content,
               chunk.chapter AS chapter,
               chunk.section AS section,
               mentioned_entities,
               entity_count
        ORDER BY entity_count DESC, chunk.page_start ASC
        LIMIT $top_k
        """
        with self._driver.session(database=self._database) as session:
            rows = session.run(query_shared, entity_names=entity_names, min_overlap=min_overlap, top_k=top_k)
            result["shared_chunks"] = [dict(r) for r in rows]

        query_relations = """
        MATCH (a)-[r]-(b)
        WHERE a.name IN $entity_names AND b.name IN $entity_names
          AND a.name < b.name
        RETURN a.name AS from_entity,
               type(r) AS relation,
               b.name AS to_entity,
               properties(r) AS properties
        LIMIT 20
        """
        with self._driver.session(database=self._database) as session:
            rows = session.run(query_relations, entity_names=entity_names)
            result["relations"] = [dict(r) for r in rows]

        return result

    def get_chapter_context(
        self,
        chunk_key: str,
        context_window: int = 2,
    ) -> dict[str, Any]:
        """沿层级结构回溯，获取 chunk 所在章节标题和相邻 chunk。"""
        query = """
        MATCH (chunk:GuidelineChunk {key: $chunk_key})
              <-[:HAS_CHUNK]-(section:GuidelineSection)
              <-[:HAS_SECTION]-(chapter:GuidelineChapter)
              <-[:HAS_CHAPTER]-(doc:GuidelineDocument)
        OPTIONAL MATCH (section)-[:HAS_CHUNK]->(neighbor:GuidelineChunk)
        WHERE abs(neighbor.chunk_index - chunk.chunk_index) <= $context_window
          AND neighbor.key <> $chunk_key
        WITH doc, chapter, section, chunk,
             neighbor ORDER BY neighbor.chunk_index ASC
        RETURN doc.title AS document,
               chapter.title AS chapter,
               section.title AS section,
               chunk.content AS target_content,
               collect({key: neighbor.key, content: neighbor.content}) AS neighbors
        """
        with self._driver.session(database=self._database) as session:
            rows = list(session.run(query, chunk_key=chunk_key, context_window=context_window))
            if not rows:
                return {"document": "", "chapter": "", "section": "", "target_content": "", "neighbors": []}
            return dict(rows[0])


class JsonGuardrailRepository:
    """基于本地 JSON 文件的红线规则仓库。"""

    def __init__(self, graph_json_path: Path):
        """读取本地 graph.json，并构建倒排索引加速查询。"""
        payload = json.loads(graph_json_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError("graph.json must be a list")
        self._records: list[dict[str, Any]] = [item for item in payload if isinstance(item, dict)]
        self._indicator_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._disease_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._drug_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """在初始化时构建倒排索引，避免每次查询全表扫描。"""
        for record in self._records:
            drug_name = str(record.get("drug_name", "")).strip()
            for item in record.get("contraindications") or []:
                if not isinstance(item, dict):
                    continue
                indicator_name = str(item.get("indicator", "")).strip()
                key = indicator_name.lower()
                self._indicator_index[key].append(
                    {
                        "drug_name": drug_name,
                        "indicator_name": indicator_name,
                        "indicator_unit": str(item.get("unit", "")).strip(),
                        "operator": str(item.get("operator", "")).strip(),
                        "threshold": item.get("threshold"),
                        "reason": str(item.get("reason", "")).strip(),
                    }
                )
            for item in record.get("disease_excludes") or []:
                disease_name = str(item).strip()
                key = disease_name.lower()
                self._disease_index[key].append({"disease_name": disease_name, "drug_name": drug_name})
            drug_key = drug_name.lower()
            for item in record.get("dosage_adjust") or []:
                if not isinstance(item, dict):
                    continue
                self._drug_index[drug_key].append(
                    {
                        "drug_name": drug_name,
                        "condition": str(item.get("condition", "")).strip(),
                        "action": str(item.get("action", "")).strip(),
                        "note": str(item.get("note", "")).strip(),
                    }
                )

    def close(self) -> None:
        """与 Neo4j 版本保持相同接口；本地文件无需释放资源。"""
        return None

    def contraindications_by_indicator(self, indicator_name: str) -> list[dict[str, Any]]:
        """从倒排索引中查询某个指标相关禁忌。"""
        return list(self._indicator_index.get(indicator_name.strip().lower(), []))

    def excludes_by_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """从倒排索引中查询某个疾病的排除药物。"""
        return list(self._disease_index.get(disease_name.strip().lower(), []))

    def adjustments_by_drug(self, drug_name: str) -> list[dict[str, Any]]:
        """从倒排索引中查询某个药物的剂量调整规则。"""
        return list(self._drug_index.get(drug_name.strip().lower(), []))
