"""Core data schemas used across nodes and APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_list(values: list[str]) -> list[str]:
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
    indicators: dict[str, float] = Field(default_factory=dict)
    diseases: list[str] = Field(default_factory=list)
    current_drugs: list[str] = Field(default_factory=list)

    @field_validator("indicators", mode="before")
    @classmethod
    def normalize_indicators(cls, value: Any) -> dict[str, float]:
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
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("field must be a list of strings")
        return _normalize_list([str(item) for item in value])


class ContraindicationHit(BaseModel):
    drug_name: str
    trigger_type: str
    trigger_name: str
    trigger_value: float | str | None = None
    operator: str | None = None
    threshold: float | str | None = None
    reason: str


class DosageAdjustmentHit(BaseModel):
    drug_name: str
    condition: str
    action: str
    note: str = ""


class GuardrailReport(BaseModel):
    contraindications: list[ContraindicationHit] = Field(default_factory=list)
    dosage_adjustments: list[DosageAdjustmentHit] = Field(default_factory=list)
    whitepaper: str


class RagSnippet(BaseModel):
    source: str
    content: str
    score: float | None = None


class ReasonerResult(BaseModel):
    recommendation: str
    recommended_drugs: list[str] = Field(default_factory=list)
    references: list[RagSnippet] = Field(default_factory=list)


class AuditResult(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    feedback: str = ""


class ConsultationOutput(BaseModel):
    patient_state: PatientState
    guardrail_report: GuardrailReport
    reasoner_result: ReasonerResult
    audit_result: AuditResult
    retries: int = 0
    trace: list[str] = Field(default_factory=list)
