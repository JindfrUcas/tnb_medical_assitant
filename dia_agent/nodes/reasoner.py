"""Reasoner 节点：生成最终诊疗建议。

这一层会综合患者状态、红线约束和 RAG 检索片段来形成最终输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from dia_agent.agent_tools import AgentToolbox, build_langchain_tools
from dia_agent.graph.repository import JsonGuardrailRepository, Neo4jGuardrailRepository
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import GuardrailReport, PatientState, RagSnippet, ReasonerResult


@dataclass
class ReasonerRunOutput:
    """Reasoner 节点的结构化执行结果。"""

    reasoner_result: ReasonerResult
    rag_snippets: list[RagSnippet] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_scratchpad: list[str] = field(default_factory=list)


class ReasonerStructuredResponse(BaseModel):
    """LangGraph ReAct agent 的最终结构化输出。"""

    final_recommendation: str = Field(description="最终诊疗建议正文。")
    recommended_drugs: list[str] = Field(default_factory=list, description="最终推荐保留或考虑的药物方案。")
    thought_summary: str = Field(default="", description="对本轮决策过程的简要总结。")


def _dedupe(items: Iterable[str]) -> list[str]:
    """按顺序去重，保证输出稳定。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        token = item.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(token)
    return ordered


def _forbidden_drug_set(report: GuardrailReport) -> set[str]:
    """把禁用药物整理成集合，便于快速判断。"""
    return {item.drug_name.lower() for item in report.contraindications}


def baseline_llm_like_recommendation(patient_state: PatientState) -> str:
    """生成一个未受红线强约束的基线建议，用于对照展示。"""
    indicators = patient_state.indicators
    hba1c = indicators.get("HbA1c") or indicators.get("hba1c")
    egfr = indicators.get("eGFR") or indicators.get("egfr")
    lines = ["标准LLM模拟方案（未强制红线）"]
    if hba1c is not None and hba1c >= 8.0:
        lines.append("- 建议首选二甲双胍并联合SGLT2抑制剂。")
    else:
        lines.append("- 建议口服降糖药单药起始。")
    if egfr is not None and egfr < 45:
        lines.append("- 肾功能偏低，建议加强监测。")
    lines.append("- 同步进行饮食运动管理。")
    return "\n".join(lines)


class ReasonerNode:
    """诊疗建议生成节点。"""

    def __init__(self, llm_client: BaseChatModel | None = None):
        """初始化基础推理节点，可选挂载文本模型客户端。"""
        self._llm_client = llm_client

    def run(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        rag_snippets: list[RagSnippet] | None = None,
        feedback: str = "",
        default_query: str = "",
    ) -> ReasonerRunOutput:
        """生成建议主入口。"""
        snippets = list(rag_snippets or [])
        if self._llm_client:
            recommendation = self._llm_generate(patient_state, guardrail_report, snippets, feedback)
        else:
            recommendation = self._template_generate(patient_state, guardrail_report, snippets, feedback)

        recommended_drugs = self._extract_recommended_drugs(patient_state, guardrail_report)
        return ReasonerRunOutput(
            reasoner_result=ReasonerResult(
                recommendation=recommendation,
                recommended_drugs=recommended_drugs,
                references=snippets,
            ),
            rag_snippets=snippets,
        )

    def _llm_generate(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        rag_snippets: list[RagSnippet],
        feedback: str,
    ) -> str:
        """使用 LangChain 标准 Prompt 链生成建议文本。"""
        guardrail_lines = []
        for item in guardrail_report.contraindications:
            guardrail_lines.append(f"- 禁用 {item.drug_name}: {item.reason}")
        for item in guardrail_report.dosage_adjustments:
            guardrail_lines.append(f"- 调整 {item.drug_name}: {item.condition} -> {item.action} {item.note}")
        rag_lines = [f"[{snippet.source}] {snippet.content[:320]}" for snippet in rag_snippets]

        llm_client = self._llm_client
        if llm_client is None:
            return self._template_generate(patient_state, guardrail_report, rag_snippets, feedback)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是内分泌专病助手。必须绝对遵守红线约束；若指南与红线冲突，红线优先。"
                    "输出结构包括：风险判断、治疗建议、监测计划。",
                ),
                (
                    "human",
                    "患者状态:\n{patient_json}\n"
                    "红线约束:\n{guardrail_text}\n"
                    "指南片段:\n{rag_text}\n"
                    "审计反馈: {feedback}",
                ),
            ]
        )
        chain = prompt | llm_client | StrOutputParser()
        content = chain.invoke(
            {
                "patient_json": json.dumps(patient_state.model_dump(), ensure_ascii=False, indent=2),
                "guardrail_text": "\n".join(guardrail_lines) or "- 无",
                "rag_text": "\n".join(rag_lines) or "- 无",
                "feedback": feedback or "无",
            }
        )
        return content or self._template_generate(patient_state, guardrail_report, rag_snippets, feedback)

    def _template_generate(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        rag_snippets: list[RagSnippet],
        feedback: str,
    ) -> str:
        """在没有模型可用时，用模板策略生成保守输出。"""
        forbidden = _forbidden_drug_set(guardrail_report)
        safe_current_drugs = [
            drug for drug in patient_state.current_drugs if drug.strip().lower() not in forbidden
        ]

        lines: list[str] = ["Dia-Agent方案（红线优先）"]
        lines.append("1) 风险判断")
        if guardrail_report.contraindications:
            for item in guardrail_report.contraindications:
                lines.append(f"- 禁用: {item.drug_name}，触发项: {item.trigger_name}，原因: {item.reason}")
        else:
            lines.append("- 未命中结构化禁忌，按指南路径继续评估。")

        lines.append("2) 治疗建议")
        if safe_current_drugs:
            lines.append(f"- 当前可继续药物: {', '.join(_dedupe(safe_current_drugs))}。")
        else:
            lines.append("- 当前药物需全面复核后再决策。")

        indicators = patient_state.indicators
        hba1c = indicators.get("HbA1c") or indicators.get("hba1c")
        if hba1c is not None and hba1c >= 9.0:
            lines.append("- 血糖控制欠佳，优先考虑胰岛素强化或基础-餐时个体化方案。")
        elif hba1c is not None and hba1c >= 7.0:
            lines.append("- 建议在安全前提下进行双药或分层强化。")
        else:
            lines.append("- 优先维持既有方案并加强生活方式干预。")

        if guardrail_report.dosage_adjustments:
            for item in guardrail_report.dosage_adjustments:
                suffix = f"（{item.note}）" if item.note else ""
                lines.append(f"- 剂量调整: {item.drug_name} 在 {item.condition} 下 {item.action}{suffix}")

        lines.append("3) 监测计划")
        lines.append("- 每2-4周复查血糖与关键肝肾功能指标，必要时提前复评。")
        lines.append("- 出现低血糖、脱水、感染等高危信号时立即就医。")

        if rag_snippets:
            lines.append("4) 指南依据")
            for snippet in rag_snippets[:3]:
                preview = snippet.content.replace("\n", " ").strip()
                lines.append(f"- [{snippet.source}] {preview[:180]}")

        if feedback:
            lines.append("5) 审计修正")
            lines.append(f"- 已根据审计反馈修正: {feedback}")

        return "\n".join(lines)

    def _extract_recommended_drugs(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
    ) -> list[str]:
        """从当前药物和基础规则中归纳候选推荐药物。"""
        forbidden = _forbidden_drug_set(guardrail_report)
        candidates: list[str] = []
        for drug in patient_state.current_drugs:
            if drug.strip().lower() in forbidden:
                continue
            candidates.append(drug)

        hba1c = patient_state.indicators.get("HbA1c") or patient_state.indicators.get("hba1c")
        if hba1c is not None and hba1c >= 9.0:
            candidates.append("个体化胰岛素方案")

        return _dedupe(candidates)


class ReActReasonerNode(ReasonerNode):
    """基于 LangGraph 预置 ReAct agent 的推理节点。"""

    def __init__(
        self,
        llm_client: BaseChatModel | None,
        repository: JsonGuardrailRepository | Neo4jGuardrailRepository,
        retriever: GuidelineRetriever,
        max_steps: int = 4,
        retrieval_k: int = 4,
    ):
        """初始化受控 ReAct 推理器，绑定工具所需的数据源与检索器。"""
        super().__init__(llm_client=llm_client)
        self._repository = repository
        self._retriever = retriever
        self._max_steps = max(1, max_steps)
        self._retrieval_k = max(1, retrieval_k)

    def run(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        rag_snippets: list[RagSnippet] | None = None,
        feedback: str = "",
        default_query: str = "",
    ) -> ReasonerRunOutput:
        """执行 LangGraph ReAct 推理；失败时回退到保守模板策略。"""
        query_hint = default_query.strip() or self._build_default_query(patient_state)
        initial_snippets = list(rag_snippets or self._retriever.retrieve(query_hint, top_k=self._retrieval_k))
        if self._llm_client is None:
            return super().run(
                patient_state=patient_state,
                guardrail_report=guardrail_report,
                rag_snippets=initial_snippets,
                feedback=feedback,
                default_query=query_hint,
            )

        toolbox = AgentToolbox(
            patient_state=patient_state,
            guardrail_report=guardrail_report,
            repository=self._repository,
            retriever=self._retriever,
            default_query=query_hint,
            default_top_k=self._retrieval_k,
        )
        toolbox.seed_snippets(initial_snippets)

        try:
            agent = create_react_agent(
                model=self._llm_client,
                tools=build_langchain_tools(toolbox),
                prompt=self._build_system_prompt(),
                response_format=ReasonerStructuredResponse,
                version="v2",
                name="dia_agent_reasoner",
            )
            result = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=self._build_user_prompt(
                                patient_state=patient_state,
                                guardrail_report=guardrail_report,
                                feedback=feedback,
                                query_hint=query_hint,
                                toolbox=toolbox,
                            )
                        )
                    ]
                },
                config={"recursion_limit": max(8, self._max_steps * 4)},
            )
        except Exception:
            fallback_snippets = toolbox.used_rag_snippets or self._retriever.retrieve(query_hint, top_k=self._retrieval_k)
            fallback_feedback = feedback or "LangGraph ReAct 执行失败，已回退到保守生成。"
            fallback = super().run(
                patient_state=patient_state,
                guardrail_report=guardrail_report,
                rag_snippets=fallback_snippets,
                feedback=fallback_feedback,
                default_query=query_hint,
            )
            return ReasonerRunOutput(
                reasoner_result=fallback.reasoner_result,
                rag_snippets=fallback.rag_snippets,
                tool_calls=toolbox.tool_calls,
                agent_scratchpad=[],
            )

        structured = result.get("structured_response")
        scratchpad = self._build_scratchpad(result.get("messages", []))
        recommendation = self._extract_structured_recommendation(structured)
        if not recommendation:
            recommendation = self._extract_message_text(result.get("messages", []))
        if not recommendation:
            recommendation = self._template_generate(
                patient_state=patient_state,
                guardrail_report=guardrail_report,
                rag_snippets=toolbox.used_rag_snippets,
                feedback=feedback,
            )
        recommended_drugs = self._extract_structured_recommended_drugs(structured)
        if not recommended_drugs:
            recommended_drugs = self._extract_recommended_drugs(patient_state, guardrail_report)

        if toolbox.used_rag_snippets:
            snippets = toolbox.used_rag_snippets
        else:
            snippets = list(rag_snippets or self._retriever.retrieve(query_hint, top_k=self._retrieval_k))
        return ReasonerRunOutput(
            reasoner_result=ReasonerResult(
                recommendation=recommendation,
                recommended_drugs=recommended_drugs,
                references=snippets,
            ),
            rag_snippets=snippets,
            tool_calls=toolbox.tool_calls,
            agent_scratchpad=scratchpad,
        )

    def _build_system_prompt(self) -> str:
        """构造 LangGraph ReAct agent 的系统提示词。"""
        return (
            "你是 Guardrail-First 的糖尿病专病助手。"
            "你必须绝对遵守红线约束，不能推荐已命中禁忌的药物。"
            "你可以调用受控工具补充证据，并根据工具观察结果逐步决策。"
            "你不能编造数据库查询语句，也不能跳过 Guardrail。"
            "当证据足够时，请直接输出最终结构化结果，不要继续调用无关工具。"
        )

    def _build_user_prompt(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        feedback: str,
        query_hint: str,
        toolbox: AgentToolbox,
    ) -> str:
        """构造 ReAct 阶段的用户提示词，注入患者状态与可用工具。"""
        return (
            "你当前负责生成安全诊疗建议。\n"
            f"患者状态:\n{json.dumps(patient_state.model_dump(), ensure_ascii=False, indent=2)}\n"
            "已命中的红线摘要:\n"
            f"{guardrail_report.whitepaper}\n"
            f"建议的默认指南检索词: {query_hint or '无'}\n"
            f"当前已准备的初始指南片段数: {len(toolbox.used_rag_snippets)}\n"
            f"审计反馈: {feedback or '无'}\n"
            "可用工具:\n"
            f"{toolbox.describe_tools()}\n"
            "请先判断是否还需要工具补充证据。"
            "如果还需要证据，就调用最合适的工具；如果证据已经足够，就直接给出最终诊疗建议。"
        )

    def _extract_structured_recommendation(self, payload: Any) -> str:
        """从结构化结果里提取最终建议文本。"""
        if isinstance(payload, ReasonerStructuredResponse):
            return payload.final_recommendation.strip()
        if isinstance(payload, dict):
            value = payload.get("final_recommendation")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_structured_recommended_drugs(self, payload: Any) -> list[str]:
        """从结构化结果里提取推荐药物列表。"""
        if isinstance(payload, ReasonerStructuredResponse):
            return _dedupe(payload.recommended_drugs)
        if isinstance(payload, dict):
            value = payload.get("recommended_drugs")
            if isinstance(value, list):
                return _dedupe([str(item) for item in value])
        return []

    def _extract_message_text(self, messages: list[Any]) -> str:
        """从 LangGraph 返回的消息列表中提取最后一条 AI 文本。"""
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                continue
            content = getattr(message, "content", "")
            text = self._normalize_message_content(content)
            if text:
                return text
        return ""

    def _build_scratchpad(self, messages: list[Any]) -> list[str]:
        """把 LangGraph 消息历史整理成便于调试的文本轨迹。"""
        scratchpad: list[str] = []
        for message in messages:
            role = getattr(message, "type", message.__class__.__name__)
            content = self._normalize_message_content(getattr(message, "content", ""))
            if content:
                scratchpad.append(f"{role}: {content}")
        return scratchpad

    def _normalize_message_content(self, content: Any) -> str:
        """把 LangChain 消息内容压平成普通文本。"""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()

    def _build_default_query(self, patient_state: PatientState) -> str:
        """当外部未传入检索词时，根据患者状态拼默认 RAG 查询。"""
        indicator_tokens = [f"{name}:{value}" for name, value in patient_state.indicators.items()]
        disease_tokens = patient_state.diseases
        return " ".join(indicator_tokens + disease_tokens) or "1型糖尿病 指南 用药"
