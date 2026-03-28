"""工作流状态定义。

LangGraph 中的每个节点都会读写同一个状态对象。
重构后只保留 perception + react_agent 两个节点所需的字段。
"""

from __future__ import annotations

from typing import Any, TypedDict

from dia_agent.schemas import AuditResult, GuardrailReport, PatientState, ReasonerResult


class DiaAgentState(TypedDict, total=False):
    """Dia-Agent 在工作流执行过程中的共享状态。"""

    raw_input: str | dict[str, Any]
    rag_query: str
    history_text: str
    patient_state: PatientState
    guardrail_report: GuardrailReport
    reasoner_result: ReasonerResult
    audit_result: AuditResult
    retries: int
    trace: list[str]
