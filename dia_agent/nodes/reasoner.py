"""Reasoner 节点：无 LLM 时的模板回退生成器。

ReAct 主控循环已迁移到 nodes/react_controller.py。
本模块保留 ReasonerNode 作为无 LLM 可用时的确定性 fallback。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dia_agent.schemas import EvidenceBundle, GuardrailReport, PatientState, RagSnippet, ReasonerResult, _normalize_list


@dataclass
class ReasonerRunOutput:
    """Reasoner 节点的结构化执行结果。"""

    reasoner_result: ReasonerResult
    rag_snippets: list[RagSnippet] = field(default_factory=list)


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
    """诊疗建议生成节点（模板/LLM 直接生成，不含 ReAct）。"""

    def __init__(self, llm_client: BaseChatModel | None = None):
        self._llm_client = llm_client

    def run(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        evidence_bundle: EvidenceBundle | None = None,
        rag_snippets: list[RagSnippet] | None = None,
        feedback: str = "",
    ) -> ReasonerRunOutput:
        """生成建议主入口。"""
        bundle = evidence_bundle or EvidenceBundle(merged_snippets=list(rag_snippets or []))
        snippets = list(bundle.merged_snippets or rag_snippets or [])
        if self._llm_client:
            recommendation = self._llm_generate(patient_state, guardrail_report, bundle, snippets, feedback)
        else:
            recommendation = self._template_generate(patient_state, guardrail_report, bundle, snippets, feedback)

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
        evidence_bundle: EvidenceBundle,
        rag_snippets: list[RagSnippet],
        feedback: str,
    ) -> str:
        guardrail_lines = []
        for item in guardrail_report.contraindications:
            guardrail_lines.append(f"- 禁用 {item.drug_name}: {item.reason}")
        for item in guardrail_report.dosage_adjustments:
            guardrail_lines.append(f"- 调整 {item.drug_name}: {item.condition} -> {item.action} {item.note}")
        rag_lines = [f"[{snippet.source}] {snippet.content[:320]}" for snippet in rag_snippets]

        llm_client = self._llm_client
        if llm_client is None:
            return self._template_generate(patient_state, guardrail_report, evidence_bundle, rag_snippets, feedback)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是内分泌专病助手。必须绝对遵守红线约束；若指南与红线冲突，红线优先。输出结构包括：风险判断、治疗建议、监测计划。"),
            ("human", "患者状态:\n{patient_json}\n红线约束:\n{guardrail_text}\n证据包摘要:\n{evidence_summary}\n指南片段:\n{rag_text}\n审计反馈: {feedback}"),
        ])
        chain = prompt | llm_client | StrOutputParser()
        content = chain.invoke({
            "patient_json": json.dumps(patient_state.model_dump(), ensure_ascii=False, indent=2),
            "guardrail_text": "\n".join(guardrail_lines) or "- 无",
            "evidence_summary": evidence_bundle.summary or "- 无",
            "rag_text": "\n".join(rag_lines) or "- 无",
            "feedback": feedback or "无",
        })
        return content or self._template_generate(patient_state, guardrail_report, evidence_bundle, rag_snippets, feedback)

    def _template_generate(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        evidence_bundle: EvidenceBundle,
        rag_snippets: list[RagSnippet],
        feedback: str,
    ) -> str:
        forbidden = guardrail_report.forbidden_drug_names
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
            lines.append(f"- 当前可继续药物: {', '.join(_normalize_list(safe_current_drugs))}。")
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

        if evidence_bundle.summary:
            lines.append("5) 证据包摘要")
            lines.append(f"- {evidence_bundle.summary.replace(chr(10), ' | ')}")

        if feedback:
            lines.append("6) 审计修正")
            lines.append(f"- 已根据审计反馈修正: {feedback}")

        return "\n".join(lines)

    def _extract_recommended_drugs(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
    ) -> list[str]:
        forbidden = guardrail_report.forbidden_drug_names
        candidates: list[str] = []
        for drug in patient_state.current_drugs:
            if drug.strip().lower() in forbidden:
                continue
            candidates.append(drug)

        hba1c = patient_state.indicators.get("HbA1c") or patient_state.indicators.get("hba1c")
        if hba1c is not None and hba1c >= 9.0:
            candidates.append("个体化胰岛素方案")

        return _normalize_list(candidates)
