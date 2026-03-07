"""Auditor node: validate reasoner output against guardrails."""

from __future__ import annotations

from dia_agent.schemas import AuditResult, GuardrailReport, ReasonerResult


class AuditorNode:
    def run(self, reasoner_result: ReasonerResult, guardrail_report: GuardrailReport) -> AuditResult:
        forbidden = {item.drug_name.lower() for item in guardrail_report.contraindications}
        violations = [drug for drug in reasoner_result.recommended_drugs if drug.lower() in forbidden]

        if violations:
            text = "、".join(violations)
            return AuditResult(
                passed=False,
                violations=violations,
                feedback=f"建议中出现红线禁药：{text}。请删除并改为安全替代方案。",
            )

        return AuditResult(passed=True, violations=[], feedback="")
