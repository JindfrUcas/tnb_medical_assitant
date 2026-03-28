"""ReAct 主控节点：整个问诊流程的核心控制器。

用 create_react_agent 驱动，通过 7 个受控工具自主完成
红线检查、图遍历、证据检索、建议生成和自审计。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from dia_agent.graph.graph_tools import _ToolContext, build_react_tools
from dia_agent.graph.repository import Neo4jGuidelineRepository
from dia_agent.nodes.guardrail import GuardrailNode, GuardrailRepository
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import (
    AuditResult,
    GuardrailReport,
    PatientState,
    RagSnippet,
    ReasonerResult,
)
from dia_agent.utils import normalize_message_content


class ReactStructuredResponse(BaseModel):
    """ReAct agent 的最终结构化输出。"""

    final_recommendation: str = Field(description="最终诊疗建议正文。")
    recommended_drugs: list[str] = Field(default_factory=list, description="最终推荐的药物方案。")
    thought_summary: str = Field(default="", description="决策过程简要总结。")


class ReactControllerOutput(BaseModel):
    """ReAct 主控节点的完整输出。"""

    guardrail_report: GuardrailReport
    reasoner_result: ReasonerResult
    audit_result: AuditResult
    trace: list[str] = Field(default_factory=list)
    retries: int = 0


SYSTEM_PROMPT = (
    "你是 Guardrail-First 糖尿病专病助手。你通过调用工具自主完成问诊推理。\n\n"
    "【强制流程】\n"
    "1. 必须首先调用 guardrail_check 获取红线报告\n"
    "2. 根据患者药物/疾病/指标，调用 graph_traverse 或 graph_expand 获取图证据\n"
    "3. 图证据不足时，调用 vector_search 补充\n"
    "4. 需要了解剂量调整时，调用 get_dosage_info\n"
    "5. 需要 chunk 上下文时，调用 get_chapter_context\n"
    "6. 生成建议后，必须调用 audit_self 验证推荐药物列表\n"
    "7. audit_self 未通过则根据 feedback 修正后再次 audit_self（最多3次）\n\n"
    "【硬性约束】\n"
    "- 禁用药物绝对不能出现在 recommended_drugs 中\n"
    "- 红线与指南冲突时，红线优先\n"
    "- audit_self 通过后输出最终结构化结果，停止调用工具\n"
)


class ReactControllerNode:
    """ReAct 主控节点，替代原来的线性 workflow。"""

    def __init__(
        self,
        llm_client: BaseChatModel | None,
        guardrail_node: GuardrailNode,
        guardrail_repo: GuardrailRepository,
        guideline_repo: Neo4jGuidelineRepository | None,
        retriever: GuidelineRetriever,
        max_steps: int = 8,
        max_audit_retries: int = 3,
        retrieval_k: int = 4,
    ):
        self._llm_client = llm_client
        self._guardrail_node = guardrail_node
        self._guardrail_repo = guardrail_repo
        self._guideline_repo = guideline_repo
        self._retriever = retriever
        self._max_steps = max_steps
        self._max_audit_retries = max_audit_retries
        self._retrieval_k = retrieval_k

    def run(self, patient_state: PatientState, rag_query: str = "") -> ReactControllerOutput:
        """执行 ReAct 主控循环。"""
        if self._llm_client is None:
            return self._fallback_run(patient_state)

        ctx = _ToolContext(
            patient_state=patient_state,
            guardrail_node=self._guardrail_node,
            guardrail_repo=self._guardrail_repo,
            guideline_repo=self._guideline_repo,
            retriever=self._retriever,
            max_audit_retries=self._max_audit_retries,
        )
        tools = build_react_tools(ctx)

        user_prompt = self._build_user_prompt(patient_state, rag_query)

        try:
            agent = create_react_agent(
                model=self._llm_client,
                tools=tools,
                prompt=SYSTEM_PROMPT,
                response_format=ReactStructuredResponse,
                version="v2",
                name="dia_react_agent",
            )
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_prompt)]},
                config={"recursion_limit": max(16, self._max_steps * 4)},
            )
        except Exception as exc:
            return self._fallback_run(patient_state, error=str(exc))

        return self._parse_agent_result(result, ctx)

    def _build_user_prompt(self, patient_state: PatientState, rag_query: str) -> str:
        """构造 ReAct 的用户提示词。"""
        state_json = json.dumps(patient_state.model_dump(), ensure_ascii=False, indent=2)
        parts = [
            "请为以下患者生成安全诊疗建议。",
            f"患者状态:\n{state_json}",
        ]
        if rag_query.strip():
            parts.append(f"建议的补充检索词: {rag_query}")
        parts.append(
            "请按照强制流程执行：先 guardrail_check，再图遍历获取证据，最后 audit_self 验证。"
        )
        return "\n".join(parts)

    def _parse_agent_result(self, result: dict[str, Any], ctx: _ToolContext) -> ReactControllerOutput:
        """从 agent 执行结果中提取结构化输出。"""
        # 提取结构化响应
        structured = result.get("structured_response")
        recommendation = ""
        recommended_drugs: list[str] = []

        if isinstance(structured, ReactStructuredResponse):
            recommendation = structured.final_recommendation.strip()
            recommended_drugs = structured.recommended_drugs
        elif isinstance(structured, dict):
            recommendation = str(structured.get("final_recommendation", "")).strip()
            recommended_drugs = structured.get("recommended_drugs", [])

        # 如果结构化输出为空，从消息中提取
        if not recommendation:
            recommendation = self._extract_last_message_text(result.get("messages", []))

        # 构建 trace
        trace = self._build_trace(result.get("messages", []))

        # guardrail_report 从 ctx 获取
        guardrail_report = ctx.guardrail_report or GuardrailReport(
            contraindications=[], dosage_adjustments=[], whitepaper="未执行红线检查"
        )

        # audit_result 从 ctx 获取
        if ctx.audit_count > 0 and guardrail_report:
            forbidden = guardrail_report.forbidden_drug_names
            violations = [d for d in recommended_drugs if d.lower() in forbidden]
            audit_result = AuditResult(
                passed=len(violations) == 0,
                violations=violations,
                feedback="" if not violations else f"仍有禁药: {'、'.join(violations)}",
            )
        else:
            audit_result = AuditResult(passed=True, violations=[], feedback="")

        return ReactControllerOutput(
            guardrail_report=guardrail_report,
            reasoner_result=ReasonerResult(
                recommendation=recommendation,
                recommended_drugs=recommended_drugs,
                references=ctx.collected_snippets,
            ),
            audit_result=audit_result,
            trace=trace,
            retries=max(0, ctx.audit_count - 1),
        )

    def _extract_last_message_text(self, messages: list[Any]) -> str:
        """从消息列表中提取最后一条 AI 文本。"""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                continue
            content = getattr(msg, "content", "")
            text = normalize_message_content(content)
            if text:
                return text
        return ""

    def _build_trace(self, messages: list[Any]) -> list[str]:
        """从 agent 消息历史中提取工具调用轨迹。"""
        trace: list[str] = []
        for msg in messages:
            role = getattr(msg, "type", msg.__class__.__name__)
            if role == "tool":
                tool_name = getattr(msg, "name", "unknown")
                trace.append(f"Tool: {tool_name}")
            elif role == "ai":
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                        trace.append(f"Agent → {name}")
                else:
                    content = normalize_message_content(getattr(msg, "content", ""))
                    if content:
                        trace.append(f"Agent: {content[:100]}")
        return trace

    def _fallback_run(self, patient_state: PatientState, error: str = "") -> ReactControllerOutput:
        """无 LLM 或 ReAct 失败时的确定性回退。"""
        # 仍然执行确定性红线检查
        report = self._guardrail_node.run(patient_state)
        forbidden = report.forbidden_drug_names
        safe_drugs = [d for d in patient_state.current_drugs if d.lower() not in forbidden]

        lines = ["Dia-Agent方案（红线优先，模板回退）"]
        if error:
            lines.append(f"注意: ReAct 执行失败 ({error})，已回退到保守生成。")

        lines.append("1) 红线约束")
        if report.contraindications:
            for item in report.contraindications:
                lines.append(f"- 禁用: {item.drug_name}，原因: {item.reason}")
        else:
            lines.append("- 未命中硬性禁忌。")

        lines.append("2) 治疗建议")
        if safe_drugs:
            lines.append(f"- 可继续药物: {', '.join(safe_drugs)}")
        else:
            lines.append("- 当前药物需全面复核。")

        hba1c = patient_state.indicators.get("HbA1c") or patient_state.indicators.get("hba1c")
        if hba1c is not None and hba1c >= 9.0:
            lines.append("- 血糖控制欠佳，优先考虑胰岛素强化方案。")
            safe_drugs.append("个体化胰岛素方案")

        lines.append("3) 监测计划")
        lines.append("- 每2-4周复查血糖与关键肝肾功能指标。")

        return ReactControllerOutput(
            guardrail_report=report,
            reasoner_result=ReasonerResult(
                recommendation="\n".join(lines),
                recommended_drugs=safe_drugs,
                references=[],
            ),
            audit_result=AuditResult(passed=True, violations=[], feedback=""),
            trace=["Fallback: 模板生成"],
            retries=0,
        )
