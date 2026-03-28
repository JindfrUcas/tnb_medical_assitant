"""项目总装配入口。

`DiaAgentPipeline` 负责把配置、图谱仓库、LLM 客户端、RAG 检索器
和 ReAct 主控工作流装配在一起。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from neo4j import GraphDatabase

from dia_agent.config import Settings, get_settings
from dia_agent.graph.repository import JsonGuardrailRepository, Neo4jGuardrailRepository, Neo4jGuidelineRepository
from dia_agent.llm import LLMConfig, build_chat_model
from dia_agent.nodes.guardrail import GuardrailNode
from dia_agent.nodes.perception import PerceptionNode
from dia_agent.nodes.react_controller import ReactControllerNode
from dia_agent.rag.retriever import GuidelineRetriever, Neo4jGuidelineGraphRetriever
from dia_agent.schemas import ConsultationOutput
from dia_agent.workflow.graph import DiaAgentWorkflow


class DiaAgentPipeline:
    """Dia-Agent 的顶层调用入口。"""

    def __init__(self, settings: Settings | None = None):
        """根据配置组装整套问诊链路。"""
        self._settings = settings or get_settings()
        self._neo4j_driver = self._build_neo4j_driver()
        self._repository = self._build_repository()
        self._guideline_repository = self._build_guideline_repository()

        llm_client = self._build_llm_client()
        vision_client = self._build_vision_client() or llm_client

        perception = PerceptionNode(llm_client=vision_client)
        guardrail = GuardrailNode(self._repository)

        graph_retriever = (
            Neo4jGuidelineGraphRetriever(self._guideline_repository)
            if self._guideline_repository is not None
            else None
        )
        retriever = GuidelineRetriever(
            persist_dir=self._settings.chroma_persist_dir,
            embedding_model=self._settings.embedding_model,
            collection_name=self._settings.chroma_collection,
            embedding_device=self._settings.embedding_device,
            graph_retriever=graph_retriever,
        )

        react_controller = ReactControllerNode(
            llm_client=llm_client,
            guardrail_node=guardrail,
            guardrail_repo=self._repository,
            guideline_repo=self._guideline_repository,
            retriever=retriever,
            max_steps=self._settings.react_max_steps,
            max_audit_retries=self._settings.react_max_audit_retries,
            retrieval_k=self._settings.retrieval_k,
        )

        self._workflow = DiaAgentWorkflow(
            perception_node=perception,
            react_controller=react_controller,
        )

    def consult(self, raw_input: str | dict, rag_query: str = "", history_text: str = "") -> ConsultationOutput:
        """对外暴露的统一问诊方法。"""
        return self._workflow.invoke(raw_input=raw_input, rag_query=rag_query, history_text=history_text)

    def close(self) -> None:
        """释放底层资源。"""
        for resource in [self._repository, self._guideline_repository]:
            close_method = getattr(resource, "close", None)
            if callable(close_method):
                close_method()
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()
            self._neo4j_driver = None

    def _build_neo4j_driver(self):
        """构建共享的 Neo4j driver。"""
        if not self._settings.neo4j_password:
            return None
        try:
            return GraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            )
        except Exception:
            return None

    def _build_repository(self):
        """构建结构化红线数据源。"""
        if self._neo4j_driver is not None:
            try:
                return Neo4jGuardrailRepository(
                    database=self._settings.neo4j_database,
                    driver=self._neo4j_driver,
                )
            except Exception:
                pass
        graph_json_path = Path(self._settings.graph_json_path)
        if not graph_json_path.exists():
            raise FileNotFoundError(f"Missing graph.json at {graph_json_path}")
        return JsonGuardrailRepository(graph_json_path)

    def _build_guideline_repository(self) -> Neo4jGuidelineRepository | None:
        """构建指南证据图仓库。"""
        if self._neo4j_driver is None:
            return None
        try:
            return Neo4jGuidelineRepository(
                database=self._settings.neo4j_database,
                driver=self._neo4j_driver,
            )
        except Exception:
            return None

    def _build_llm_client(self) -> BaseChatModel | None:
        """构建文本推理模型客户端。"""
        if not self._settings.llm_base_url or not self._settings.llm_api_key:
            return None
        config = LLMConfig(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            model=self._settings.llm_model,
            wire_api=self._settings.llm_wire_api,
            temperature=self._settings.llm_temperature,
            timeout_sec=self._settings.llm_timeout_sec,
        )
        return build_chat_model(config)

    def _build_vision_client(self) -> BaseChatModel | None:
        """构建视觉模型客户端。"""
        base_url = self._settings.vision_base_url or self._settings.llm_base_url
        api_key = self._settings.vision_api_key or self._settings.llm_api_key
        if not base_url or not api_key:
            return None
        config = LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=self._settings.vision_model,
            wire_api=self._settings.vision_wire_api or self._settings.llm_wire_api,
            temperature=0.0,
            timeout_sec=self._settings.llm_timeout_sec,
        )
        return build_chat_model(config)
