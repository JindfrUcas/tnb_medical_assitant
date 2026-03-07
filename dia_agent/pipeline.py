"""Top-level pipeline entry for consultation flow."""

from __future__ import annotations

from pathlib import Path

from dia_agent.config import Settings, get_settings
from dia_agent.graph.repository import JsonGuardrailRepository, Neo4jGuardrailRepository
from dia_agent.llm import LLMConfig, OpenAICompatibleChatClient
from dia_agent.nodes.auditor import AuditorNode
from dia_agent.nodes.guardrail import GuardrailNode
from dia_agent.nodes.perception import PerceptionNode
from dia_agent.nodes.reasoner import ReasonerNode
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import ConsultationOutput
from dia_agent.workflow.graph import DiaAgentWorkflow


class DiaAgentPipeline:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._repository = self._build_repository()

        llm_client = self._build_llm_client()
        vision_client = self._build_vision_client() or llm_client

        perception = PerceptionNode(llm_client=vision_client)
        guardrail = GuardrailNode(self._repository)
        reasoner = ReasonerNode(llm_client=llm_client)
        auditor = AuditorNode()
        retriever = GuidelineRetriever(
            persist_dir=self._settings.chroma_persist_dir,
            embedding_model=self._settings.embedding_model,
            collection_name=self._settings.chroma_collection,
        )

        self._workflow = DiaAgentWorkflow(
            perception_node=perception,
            guardrail_node=guardrail,
            reasoner_node=reasoner,
            auditor_node=auditor,
            retriever=retriever,
            max_retries=self._settings.max_reasoner_retries,
        )

    def consult(self, raw_input: str | dict, rag_query: str = "", history_text: str = "") -> ConsultationOutput:
        return self._workflow.invoke(raw_input=raw_input, rag_query=rag_query, history_text=history_text)

    def close(self) -> None:
        close_method = getattr(self._repository, "close", None)
        if callable(close_method):
            close_method()

    def _build_repository(self):
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

    def _build_llm_client(self) -> OpenAICompatibleChatClient | None:
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
        return OpenAICompatibleChatClient(config)

    def _build_vision_client(self) -> OpenAICompatibleChatClient | None:
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
        return OpenAICompatibleChatClient(config)
