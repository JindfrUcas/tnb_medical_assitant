"""Reasoner 节点：生成最终诊疗建议。

这一层会综合患者状态、红线约束和 RAG 检索片段来形成最终输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable

from dia_agent.agent_tools import AgentToolbox
from dia_agent.graph.repository import JsonGuardrailRepository, Neo4jGuardrailRepository
from dia_agent.llm import OpenAICompatibleChatClient
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import GuardrailReport, PatientState, RagSnippet, ReasonerResult


@dataclass
class ReasonerRunOutput:
    """Structured execution payload returned by reasoner nodes."""

    reasoner_result: ReasonerResult
    rag_snippets: list[RagSnippet] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_scratchpad: list[str] = field(default_factory=list)


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

    def __init__(self, llm_client: OpenAICompatibleChatClient | None = None):
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
        """构造 Prompt 并调用大模型生成建议。"""
        guardrail_lines = []
        for item in guardrail_report.contraindications:
            guardrail_lines.append(f"- 禁用 {item.drug_name}: {item.reason}")
        for item in guardrail_report.dosage_adjustments:
            guardrail_lines.append(f"- 调整 {item.drug_name}: {item.condition} -> {item.action} {item.note}")
        rag_lines = [f"[{snippet.source}] {snippet.content[:320]}" for snippet in rag_snippets]

        system_prompt = (
            "你是内分泌专病助手。必须绝对遵守红线约束；若指南与红线冲突，红线优先。"
            "输出结构包括：风险判断、治疗建议、监测计划。"
        )
        llm_client = self._llm_client
        if llm_client is None:
            return self._template_generate(patient_state, guardrail_report, rag_snippets, feedback)
        patient_json = json.dumps(patient_state.model_dump())
        user_prompt = (
            f"患者状态: {patient_json}\n"
            f"红线约束:\n{chr(10).join(guardrail_lines) or '- 无'}\n"
            f"指南片段:\n{chr(10).join(rag_lines) or '- 无'}\n"
            f"审计反馈: {feedback or '无'}"
        )
        content = llm_client.complete(system_prompt, user_prompt)
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
    """Reasoner upgraded with a controlled ReAct loop."""

    def __init__(
        self,
        llm_client: OpenAICompatibleChatClient | None,
        repository: JsonGuardrailRepository | Neo4jGuardrailRepository,
        retriever: GuidelineRetriever,
        max_steps: int = 4,
        retrieval_k: int = 4,
    ):
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
        query_hint = default_query.strip() or self._build_default_query(patient_state)
        if self._llm_client is None:
            snippets = list(rag_snippets or self._retriever.retrieve(query_hint, top_k=self._retrieval_k))
            return super().run(
                patient_state=patient_state,
                guardrail_report=guardrail_report,
                rag_snippets=snippets,
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
        toolbox.seed_snippets(list(rag_snippets or []))

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": self._build_user_prompt(
                    patient_state=patient_state,
                    guardrail_report=guardrail_report,
                    feedback=feedback,
                    query_hint=query_hint,
                    toolbox=toolbox,
                ),
            },
        ]

        scratchpad: list[str] = []
        for step in range(1, self._max_steps + 1):
            raw_text = self._call_agent(messages)
            decision = self._parse_agent_json(raw_text)

            thought = str(decision.get("thought", "")).strip()
            if thought:
                scratchpad.append(f"Step {step} Thought: {thought}")

            action = self._normalize_action(decision.get("action"))
            is_finish = action == "finish" or bool(decision.get("done"))
            if is_finish:
                recommendation = self._extract_final_recommendation(decision)
                if not recommendation:
                    recommendation = self._template_generate(
                        patient_state=patient_state,
                        guardrail_report=guardrail_report,
                        rag_snippets=toolbox.used_rag_snippets,
                        feedback=feedback,
                    )
                recommended_drugs = self._normalize_recommended_drugs(decision.get("recommended_drugs"))
                if not recommended_drugs:
                    recommended_drugs = self._extract_recommended_drugs(patient_state, guardrail_report)
                return ReasonerRunOutput(
                    reasoner_result=ReasonerResult(
                        recommendation=recommendation,
                        recommended_drugs=recommended_drugs,
                        references=toolbox.used_rag_snippets,
                    ),
                    rag_snippets=toolbox.used_rag_snippets,
                    tool_calls=toolbox.tool_calls,
                    agent_scratchpad=scratchpad,
                )

            action_input = decision.get("action_input")
            if not isinstance(action_input, dict):
                action_input = {}
            execution = toolbox.execute(action, action_input)
            scratchpad.append(f"Step {step} Action: {execution.tool_name}")
            scratchpad.append(f"Step {step} Observation: {execution.observation}")

            assistant_payload = raw_text.strip() or json.dumps(decision, ensure_ascii=False)
            messages.append({"role": "assistant", "content": assistant_payload})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"工具 `{execution.tool_name}` 的观察结果如下:\n{execution.observation}\n"
                        "请继续思考，并且只输出严格 JSON。"
                    ),
                }
            )

        fallback_snippets = toolbox.used_rag_snippets or self._retriever.retrieve(query_hint, top_k=self._retrieval_k)
        fallback_feedback = feedback or "ReAct 达到最大步数，已回退到保守生成。"
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
            agent_scratchpad=scratchpad,
        )

    def _build_system_prompt(self) -> str:
        return (
            "你是 Guardrail-First 的糖尿病专病助手。"
            "你必须绝对遵守红线约束，不能推荐已命中禁忌的药物。"
            "你可以通过受控工具补充证据，采用 ReAct 方式逐步决策。"
            "你不能编造数据库查询语句，也不能跳过 Guardrail。"
            "每次回复必须是严格 JSON。"
        )

    def _build_user_prompt(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        feedback: str,
        query_hint: str,
        toolbox: AgentToolbox,
    ) -> str:
        return (
            "你当前负责生成安全诊疗建议。\n"
            f"患者状态:\n{json.dumps(patient_state.model_dump(), ensure_ascii=False, indent=2)}\n"
            "已命中的红线摘要:\n"
            f"{guardrail_report.whitepaper}\n"
            f"建议的默认指南检索词: {query_hint or '无'}\n"
            f"审计反馈: {feedback or '无'}\n"
            "可用工具:\n"
            f"{toolbox.describe_tools()}\n"
            "请先判断是否还需要工具补充证据。"
            "如果需要工具，请输出 JSON: "
            '{"thought":"...", "action":"retrieve_guidelines", "action_input":{"query":"..."}, "done": false}\n'
            "如果证据已经足够，请输出 JSON: "
            '{"thought":"...", "action":"finish", "final_recommendation":"...", "recommended_drugs":["..."], "done": true}'
        )

    def _call_agent(self, messages: list[dict[str, Any]]) -> str:
        llm_client = self._llm_client
        if llm_client is None:
            return ""
        try:
            return llm_client.chat(messages, response_format={"type": "json_object"})
        except Exception:
            try:
                return llm_client.chat(messages)
            except Exception:
                return ""

    def _parse_agent_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {}
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _normalize_action(self, value: Any) -> str:
        return str(value or "").strip().lower()

    def _extract_final_recommendation(self, payload: dict[str, Any]) -> str:
        for key in ("final_recommendation", "final_answer", "answer", "recommendation"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _normalize_recommended_drugs(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return _dedupe(re.split(r"[，,、;\n]+", value))
        if isinstance(value, list):
            return _dedupe([str(item) for item in value])
        return []

    def _build_default_query(self, patient_state: PatientState) -> str:
        indicator_tokens = [f"{name}:{value}" for name, value in patient_state.indicators.items()]
        disease_tokens = patient_state.diseases
        return " ".join(indicator_tokens + disease_tokens) or "1型糖尿病 指南 用药"
