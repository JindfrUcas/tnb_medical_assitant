"""RAG 检索模块。

负责从已构建好的向量索引和 Neo4j 指南证据图中，
取回与当前问题最相关的指南片段。
"""

from __future__ import annotations

from pathlib import Path

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-not-found]
    from langchain_chroma import Chroma  # type: ignore[import-not-found]
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

from dia_agent.graph.evidence_linking import EntityAlias, build_entity_alias_index, extract_entity_matches
from dia_agent.graph.repository import Neo4jGuidelineRepository
from dia_agent.schemas import RagSnippet


class Neo4jGuidelineGraphRetriever:
    """基于 Neo4j 实体连边的轻量图证据检索器。"""

    def __init__(self, repository: Neo4jGuidelineRepository):
        """初始化图检索器，并缓存实体别名索引。"""
        self._repository = repository
        self._alias_index: list[EntityAlias] | None = None

    def retrieve(self, query: str, top_k: int = 4, source: str | None = None) -> list[RagSnippet]:
        """根据查询里命中的实体，优先取回图上直接可解释的 chunk。"""
        query = query.strip()
        if not query:
            return []

        alias_index = self._get_alias_index()
        matches = extract_entity_matches(query, alias_index)
        total_entities = sum(len(items) for items in matches.values())
        if total_entities == 0:
            return []

        primary_min_hits = 2 if total_entities >= 2 else 1
        rows = self._repository.fetch_linked_chunks(
            drug_names=matches["Drug"],
            disease_names=matches["Disease"],
            indicator_names=matches["Indicator"],
            top_k=top_k,
            source=source,
            min_hits=primary_min_hits,
        )
        if len(rows) < top_k and primary_min_hits > 1:
            rows.extend(
                self._repository.fetch_linked_chunks(
                    drug_names=matches["Drug"],
                    disease_names=matches["Disease"],
                    indicator_names=matches["Indicator"],
                    top_k=top_k - len(rows),
                    source=source,
                    min_hits=1,
                    exclude_keys=[str(row.get("chunk_key", "")) for row in rows],
                )
            )
        snippets: list[RagSnippet] = []
        seen_keys: set[str] = set()
        for row in rows:
            chunk_key = str(row.get("chunk_key", "")).strip()
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            if chunk_key and chunk_key in seen_keys:
                continue
            if chunk_key:
                seen_keys.add(chunk_key)
            snippets.append(
                RagSnippet(
                    source=str(row.get("source", "UNKNOWN")),
                    content=content,
                    score=float(row["score"]) if row.get("score") is not None else None,
                )
            )
            if len(snippets) >= top_k:
                break
        return snippets

    def _get_alias_index(self) -> list[EntityAlias]:
        """延迟加载图中实体名称，并构建可复用的别名索引。"""
        if self._alias_index is None:
            entities = self._repository.list_entity_names()
            self._alias_index = build_entity_alias_index(
                drug_names=entities["Drug"],
                disease_names=entities["Disease"],
                indicator_names=entities["Indicator"],
            )
        return self._alias_index


class GuidelineRetriever:
    """指南检索器。"""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model: str,
        collection_name: str,
        embedding_device: str = "cpu",
        graph_retriever: Neo4jGuidelineGraphRetriever | None = None,
    ):
        """初始化检索器，但延迟创建向量库连接。"""
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device.strip() or "cpu"
        self._collection_name = collection_name
        self._graph_retriever = graph_retriever
        self._vector_store: Chroma | None = None

    def retrieve(self, query: str, top_k: int = 4, source: str | None = None) -> list[RagSnippet]:
        """执行混合检索并返回统一结构。

        优先尝试图证据检索；如果不够，再用向量相似度补齐。
        """
        query = query.strip()
        if not query:
            return []
        merged: list[RagSnippet] = []
        seen: set[tuple[str, str]] = set()

        if self._graph_retriever is not None:
            try:
                graph_snippets = self._graph_retriever.retrieve(query=query, top_k=top_k, source=source)
            except Exception:
                graph_snippets = []
            self._extend_unique(merged, seen, graph_snippets, top_k=top_k)

        remaining = top_k - len(merged)
        if remaining > 0:
            self._extend_unique(merged, seen, self._retrieve_from_vector_store(query, remaining, source), top_k=top_k)
        return merged

    def retrieve_graph(self, query: str, top_k: int = 4, source: str | None = None) -> list[RagSnippet]:
        """只取图证据结果。

        供 Evidence 节点显式区分“图证据”和“向量证据”时使用。
        """
        if self._graph_retriever is None:
            return []
        try:
            return self._graph_retriever.retrieve(query=query, top_k=top_k, source=source)
        except Exception:
            return []

    def retrieve_vector(self, query: str, top_k: int = 4, source: str | None = None) -> list[RagSnippet]:
        """只取向量检索结果。"""
        return self._retrieve_from_vector_store(query=query, top_k=top_k, source=source)

    def _get_vector_store(self) -> Chroma:
        """按需初始化 Chroma 向量库连接。"""
        if self._vector_store is None:
            embedding = HuggingFaceEmbeddings(
                model_name=self._embedding_model,
                model_kwargs={"device": self._embedding_device},
            )
            self._vector_store = Chroma(
                collection_name=self._collection_name,
                persist_directory=str(self._persist_dir),
                embedding_function=embedding,
            )
        return self._vector_store

    def _retrieve_from_vector_store(self, query: str, top_k: int, source: str | None) -> list[RagSnippet]:
        """从 Chroma 向量库中取回补充证据。"""
        if top_k <= 0 or not self._persist_dir.exists():
            return []

        vector_store = self._get_vector_store()
        metadata_filter = {"source": source} if source else None
        docs = vector_store.similarity_search(query, k=top_k, filter=metadata_filter)
        return [
            RagSnippet(
                source=str(doc.metadata.get("source", "UNKNOWN")),
                content=doc.page_content,
                score=None,
            )
            for doc in docs
        ]

    def _extend_unique(
        self,
        target: list[RagSnippet],
        seen: set[tuple[str, str]],
        snippets: list[RagSnippet],
        top_k: int,
    ) -> None:
        """按 `source + content` 去重，把片段合并到最终结果里。"""
        for snippet in snippets:
            if len(target) >= top_k:
                return
            key = (snippet.source, snippet.content)
            if key in seen:
                continue
            seen.add(key)
            target.append(snippet)
