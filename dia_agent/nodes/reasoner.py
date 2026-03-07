"""Reasoner node: synthesize guardrail-safe treatment proposal."""

from __future__ import annotations

import json
from typing import Iterable

from dia_agent.llm import OpenAICompatibleChatClient
from dia_agent.schemas import GuardrailReport, PatientState, RagSnippet, ReasonerResult


def _dedupe(items: Iterable[str]) -> list[str]:
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
    return {item.drug_name.lower() for item in report.contraindications}


def baseline_llm_like_recommendation(patient_state: PatientState) -> str:
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
    def __init__(self, llm_client: OpenAICompatibleChatClient | None = None):
        self._llm_client = llm_client

    def run(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        rag_snippets: list[RagSnippet],
        feedback: str = "",
    ) -> ReasonerResult:
        if self._llm_client:
            recommendation = self._llm_generate(patient_state, guardrail_report, rag_snippets, feedback)
        else:
            recommendation = self._template_generate(patient_state, guardrail_report, rag_snippets, feedback)

        recommended_drugs = self._extract_recommended_drugs(patient_state, guardrail_report)
        return ReasonerResult(
            recommendation=recommendation,
            recommended_drugs=recommended_drugs,
            references=rag_snippets,
        )

    def _llm_generate(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        rag_snippets: list[RagSnippet],
        feedback: str,
    ) -> str:
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
