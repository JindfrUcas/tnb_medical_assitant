"""Workflow state type definitions."""

from __future__ import annotations

from typing import Any, TypedDict

from dia_agent.schemas import AuditResult, GuardrailReport, PatientState, RagSnippet, ReasonerResult


class DiaAgentState(TypedDict, total=False):
    raw_input: str | dict[str, Any]
    rag_query: str
    history_text: str
    patient_state: PatientState
    guardrail_report: GuardrailReport
    rag_snippets: list[RagSnippet]
    reasoner_result: ReasonerResult
    audit_result: AuditResult
    retries: int
    max_retries: int
    trace: list[str]
