"""项目核心数据结构定义。

这些 Schema 贯穿 API、工作流节点和最终输出，是整个项目的数据骨架。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_list(values: list[str]) -> list[str]:
    """统一做字符串清洗和去重。"""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = value.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


class PatientState(BaseModel):
    """标准化后的患者状态。"""

    indicators: dict[str, float] = Field(default_factory=dict)
    diseases: list[str] = Field(default_factory=list)
    current_drugs: list[str] = Field(default_factory=list)

    @field_validator("indicators", mode="before")
    @classmethod
    def normalize_indicators(cls, value: Any) -> dict[str, float]:
        """把指标字段规整成 `dict[str, float]`。"""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("indicators must be a dictionary")
        normalized: dict[str, float] = {}
        for name, indicator_value in value.items():
            key = str(name).strip()
            if not key:
                continue
            normalized[key] = float(indicator_value)
        return normalized

    @field_validator("diseases", "current_drugs", mode="before")
    @classmethod
    def normalize_array_field(cls, value: Any) -> list[str]:
        """把数组类字段规整成字符串列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("field must be a list of strings")
        return _normalize_list([str(item) for item in value])


class ContraindicationHit(BaseModel):
    """命中的禁忌规则。"""

    drug_name: str
    trigger_type: str
    trigger_name: str
    trigger_value: float | str | None = None
    operator: str | None = None
    threshold: float | str | None = None
    reason: str


class DosageAdjustmentHit(BaseModel):
    """命中的剂量调整规则。"""

    drug_name: str
    condition: str
    action: str
    note: str = ""


class GuardrailReport(BaseModel):
    """Guardrail 节点输出的结构化报告。"""

    contraindications: list[ContraindicationHit] = Field(default_factory=list)
    dosage_adjustments: list[DosageAdjustmentHit] = Field(default_factory=list)
    whitepaper: str


class RagSnippet(BaseModel):
    """RAG 检索返回的一段指南片段。"""

    source: str
    content: str
    score: float | None = None


class ReasonerResult(BaseModel):
    """Reasoner 节点输出结果。"""

    recommendation: str
    recommended_drugs: list[str] = Field(default_factory=list)
    references: list[RagSnippet] = Field(default_factory=list)


class AuditResult(BaseModel):
    """Auditor 节点输出结果。"""

    passed: bool
    violations: list[str] = Field(default_factory=list)
    feedback: str = ""


class ConsultationOutput(BaseModel):
    """一次完整问诊的最终聚合输出。"""

    patient_state: PatientState
    guardrail_report: GuardrailReport
    reasoner_result: ReasonerResult
    audit_result: AuditResult
    retries: int = 0
    trace: list[str] = Field(default_factory=list)
