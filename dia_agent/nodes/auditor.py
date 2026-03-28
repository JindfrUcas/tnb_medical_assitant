"""Auditor 节点：对最终输出执行最后一道安全检查。"""

from __future__ import annotations

from dia_agent.schemas import AuditResult, GuardrailReport, ReasonerResult


class AuditorNode:
    """检查生成结果中是否出现禁用药物。"""

    def run(self, reasoner_result: ReasonerResult, guardrail_report: GuardrailReport) -> AuditResult:
        """执行审计并返回是否通过。"""
        forbidden = {item.drug_name.lower() for item in guardrail_report.contraindications}
        violations = [drug for drug in reasoner_result.recommended_drugs if drug.lower() in forbidden]

        if violations:
            text = "、".join(violations)
            return AuditResult(
                passed=False,
                violations=violations,
                feedback=f"建议中出现红线禁药：{text}。请删除并改为安全替代方案。",
                failure_type="forbidden_drug_violation",
                repair_focus=f"移除禁药 {text}，并基于已组装证据寻找安全替代方案。",
            )

        return AuditResult(passed=True, violations=[], feedback="", failure_type="", repair_focus="")
