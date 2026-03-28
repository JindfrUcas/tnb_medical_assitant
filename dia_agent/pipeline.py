"""项目总装配入口。

`DiaAgentPipeline` 负责把配置、图谱仓库、LLM 客户端、RAG 检索器和 LangGraph 工作流装配在一起。
你可以把它理解成整个项目的“总控制台”。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from dia_agent.config import Settings, get_settings
from dia_agent.graph.repository import JsonGuardrailRepository, Neo4jGuardrailRepository, Neo4jGuidelineRepository
from dia_agent.llm import LLMConfig, build_chat_model
from dia_agent.nodes.auditor import AuditorNode
from dia_agent.nodes.evidence import EvidenceAssemblyNode
from dia_agent.nodes.guardrail import GuardrailNode
from dia_agent.nodes.perception import PerceptionNode
from dia_agent.nodes.reasoner import ReActReasonerNode
from dia_agent.rag.retriever import GuidelineRetriever, Neo4jGuidelineGraphRetriever
from dia_agent.schemas import ConsultationOutput
from dia_agent.workflow.graph import DiaAgentWorkflow


class DiaAgentPipeline:
    """Dia-Agent 的顶层调用入口。"""

    def __init__(self, settings: Settings | None = None):
        """根据配置组装整套问诊链路。"""
        self._settings = settings or get_settings()
        self._repository = self._build_repository()
        self._guideline_repository = self._build_guideline_repository()

        # 语言模型负责文本推理；视觉模型主要用于多模态结构化抽取。
        # 如果没有单独配置视觉模型，就退回复用文本模型。
        llm_client = self._build_llm_client()
        vision_client = self._build_vision_client() or llm_client

        # 这里把“感知、红线、推理、审计、RAG 检索”几个核心部件拼成一条工作流。
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
        reasoner = ReActReasonerNode(
            llm_client=llm_client,
            repository=self._repository,
            retriever=retriever,
            max_steps=self._settings.react_max_steps,
            retrieval_k=self._settings.retrieval_k,
        )
        evidence = EvidenceAssemblyNode(retriever=retriever, retrieval_k=self._settings.retrieval_k)
        auditor = AuditorNode()

        self._workflow = DiaAgentWorkflow(
            perception_node=perception,
            guardrail_node=guardrail,
            evidence_node=evidence,
            reasoner_node=reasoner,
            auditor_node=auditor,
            retriever=retriever,
            max_retries=self._settings.max_reasoner_retries,
        )

    def consult(self, raw_input: str | dict, rag_query: str = "", history_text: str = "") -> ConsultationOutput:
        """对外暴露的统一问诊方法。"""
        return self._workflow.invoke(raw_input=raw_input, rag_query=rag_query, history_text=history_text)

    def close(self) -> None:
        """释放底层资源，比如 Neo4j 连接。"""
        for resource in [self._repository, self._guideline_repository]:
            close_method = getattr(resource, "close", None)
            if callable(close_method):
                close_method()

    def _build_repository(self):
        """构建结构化红线数据源。

        优先使用 Neo4j；如果没有配置或连接失败，则回退到本地 `graph.json`。
        """
        if self._settings.neo4j_password:
            try:
                return Neo4jGuardrailRepository(
                    uri=self._settings.neo4j_uri,
                    user=self._settings.neo4j_user,
                    password=self._settings.neo4j_password,
                    database=self._settings.neo4j_database,
                )
            except Exception:
                pass

        graph_json_path = Path(self._settings.graph_json_path)
        if not graph_json_path.exists():
            raise FileNotFoundError(f"Missing graph.json at {graph_json_path}")
        return JsonGuardrailRepository(graph_json_path)

    def _build_guideline_repository(self) -> Neo4jGuidelineRepository | None:
        """构建指南证据图仓库。

        这里是一个纯增量能力：
        - 没有 Neo4j 凭据时直接关闭。
        - 有 Neo4j 凭据但库里还没导入 chunk 时，运行时会自动回退到向量检索。
        """
        if not self._settings.neo4j_password:
            return None
        try:
            return Neo4jGuidelineRepository(
                uri=self._settings.neo4j_uri,
                user=self._settings.neo4j_user,
                password=self._settings.neo4j_password,
                database=self._settings.neo4j_database,
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
        """构建视觉模型客户端。

        当前项目的图片能力主要用在 Perception 节点，用于把化验单等图片转成结构化状态。
        """
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
