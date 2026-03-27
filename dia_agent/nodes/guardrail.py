"""Guardrail 节点：执行硬性临床约束检查。

这一层是整个系统的安全核心，负责在生成建议之前先查禁忌、排除项和剂量调整规则。
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from dia_agent.schemas import ContraindicationHit, DosageAdjustmentHit, GuardrailReport, PatientState


class GuardrailRepository(Protocol):
    """Guardrail 节点依赖的数据仓库协议。"""

    def contraindications_by_indicator(self, indicator_name: str) -> list[dict[str, Any]]:
        """按指标名查询禁忌规则。"""
        ...

    def excludes_by_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """按疾病名查询排除药物。"""
        ...

    def adjustments_by_drug(self, drug_name: str) -> list[dict[str, Any]]:
        """按药物名查询剂量调整规则。"""
        ...


def _to_float(value: Any) -> float | None:
    """尽量把输入值转换成浮点数。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare(left: float, operator: str, right: Any) -> bool:
    """执行基础数值比较。"""
    threshold = _to_float(right)
    if threshold is None:
        return False

    op = operator.strip()
    if op == "<":
        return left < threshold
    if op == "<=":
        return left <= threshold
    if op == ">":
        return left > threshold
    if op == ">=":
        return left >= threshold
    if op in {"=", "=="}:
        return left == threshold
    if op == "!=":
        return left != threshold
    return False


def _condition_mentions_indicator(condition: str, indicator_name: str) -> bool:
    """判断条件字符串里是否提到了某个指标。"""
    return indicator_name.lower() in condition.lower()


def _match_range_condition(condition: str, indicator_name: str, value: float) -> bool:
    """匹配区间条件，例如 `30-45` 这类表达。"""
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)")
    if not _condition_mentions_indicator(condition, indicator_name):
        return False
    match = pattern.search(condition)
    if not match:
        return False
    lower = float(match.group(1))
    upper = float(match.group(2))
    return lower <= value <= upper


def _match_comparator_condition(condition: str, indicator_name: str, value: float) -> bool:
    """匹配比较条件，例如 `< 30`、`>= 45`。"""
    pattern = re.compile(r"(<=|>=|<|>|=)\s*(\d+(?:\.\d+)?)")
    if not _condition_mentions_indicator(condition, indicator_name):
        return False
    match = pattern.search(condition)
    if not match:
        return False
    return _compare(value, match.group(1), float(match.group(2)))


class GuardrailNode:
    """根据患者状态构建结构化红线报告。"""

    def __init__(self, repository: GuardrailRepository):
        """初始化红线节点，并注入底层规则仓库。"""
        self._repository = repository

    def run(self, patient_state: PatientState) -> GuardrailReport:
        """执行红线检索并生成白皮书。"""
        contraindications: list[ContraindicationHit] = []
        dosage_adjustments: list[DosageAdjustmentHit] = []

        for indicator_name, indicator_value in patient_state.indicators.items():
            rows = self._repository.contraindications_by_indicator(indicator_name)
            for row in rows:
                operator = str(row.get("operator", "")).strip()
                threshold = row.get("threshold")
                if not _compare(indicator_value, operator, threshold):
                    continue
                contraindications.append(
                    ContraindicationHit(
                        drug_name=str(row.get("drug_name", "")).strip(),
                        trigger_type="indicator",
                        trigger_name=str(row.get("indicator_name", indicator_name)).strip(),
                        trigger_value=indicator_value,
                        operator=operator,
                        threshold=threshold,
                        reason=str(row.get("reason", "未提供原因")).strip() or "未提供原因",
                    )
                )

        for disease_name in patient_state.diseases:
            rows = self._repository.excludes_by_disease(disease_name)
            for row in rows:
                contraindications.append(
                    ContraindicationHit(
                        drug_name=str(row.get("drug_name", "")).strip(),
                        trigger_type="disease",
                        trigger_name=str(row.get("disease_name", disease_name)).strip(),
                        trigger_value=None,
                        operator=None,
                        threshold=None,
                        reason="疾病排除项命中",
                    )
                )

        seen_adjustments: set[tuple[str, str, str, str]] = set()
        for drug_name in patient_state.current_drugs:
            rows = self._repository.adjustments_by_drug(drug_name)
            for row in rows:
                condition = str(row.get("condition", "")).strip()
                action = str(row.get("action", "")).strip()
                note = str(row.get("note", "")).strip()
                if not self._condition_matches(patient_state, condition):
                    continue
                key = (drug_name, condition, action, note)
                if key in seen_adjustments:
                    continue
                seen_adjustments.add(key)
                dosage_adjustments.append(
                    DosageAdjustmentHit(
                        drug_name=drug_name,
                        condition=condition,
                        action=action,
                        note=note,
                    )
                )

        whitepaper = self._build_whitepaper(patient_state, contraindications, dosage_adjustments)
        return GuardrailReport(
            contraindications=contraindications,
            dosage_adjustments=dosage_adjustments,
            whitepaper=whitepaper,
        )

    def _condition_matches(self, patient_state: PatientState, condition: str) -> bool:
        """判断剂量调整条件是否与当前患者状态匹配。"""
        normalized = condition.strip()
        if not normalized:
            return True

        matched_any_indicator = False
        for indicator_name, indicator_value in patient_state.indicators.items():
            if _condition_mentions_indicator(normalized, indicator_name):
                matched_any_indicator = True
                if _match_range_condition(normalized, indicator_name, indicator_value):
                    return True
                if _match_comparator_condition(normalized, indicator_name, indicator_value):
                    return True

        if matched_any_indicator:
            return False

        for disease_name in patient_state.diseases:
            if disease_name.lower() in normalized.lower():
                return True

        return True

    def _build_whitepaper(
        self,
        patient_state: PatientState,
        contraindications: list[ContraindicationHit],
        dosage_adjustments: list[DosageAdjustmentHit],
    ) -> str:
        """把命中的规则整理成可读性较好的白皮书文本。"""
        lines: list[str] = ["《本次问诊安全约束白皮书》"]
        lines.append("一、患者关键状态")

        if patient_state.indicators:
            for name, value in patient_state.indicators.items():
                lines.append(f"- 指标 {name}: {value}")
        else:
            lines.append("- 未提供结构化指标")

        if patient_state.diseases:
            lines.append(f"- 合并疾病: {', '.join(patient_state.diseases)}")

        lines.append("二、禁用药物红线")
        if contraindications:
            for hit in contraindications:
                if hit.trigger_type == "indicator":
                    lines.append(
                        f"- {hit.drug_name}: {hit.trigger_name} {hit.operator} {hit.threshold}，患者值 {hit.trigger_value}，原因：{hit.reason}"
                    )
                else:
                    lines.append(f"- {hit.drug_name}: 命中疾病排除项 {hit.trigger_name}，原因：{hit.reason}")
        else:
            lines.append("- 暂未命中硬性禁忌")

        lines.append("三、剂量调整建议")
        if dosage_adjustments:
            for item in dosage_adjustments:
                if item.note:
                    lines.append(f"- {item.drug_name}: [{item.condition}] {item.action}（{item.note}）")
                else:
                    lines.append(f"- {item.drug_name}: [{item.condition}] {item.action}")
        else:
            lines.append("- 暂无命中剂量调整条目")

        lines.append("四、执行原则")
        lines.append("- 任何方案不得突破上述红线；若与指南冲突，红线优先。")
        return "\n".join(lines)
