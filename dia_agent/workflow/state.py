"""工作流状态定义。

LangGraph 中的每个节点都会读写同一个状态对象。
这里把整条链路上可能出现的字段集中定义出来，便于理解数据流向。
"""

from __future__ import annotations

from typing import Any, TypedDict

from dia_agent.schemas import AuditResult, GuardrailReport, PatientState, RagSnippet, ReasonerResult


class DiaAgentState(TypedDict, total=False):
    """Dia-Agent 在工作流执行过程中的共享状态。"""

    raw_input: str | dict[str, Any]
    rag_query: str
    history_text: str
    patient_state: PatientState
    guardrail_report: GuardrailReport
    rag_snippets: list[RagSnippet]
    reasoner_result: ReasonerResult
    audit_result: AuditResult
    tool_calls: list[dict[str, Any]]
    agent_scratchpad: list[str]
    retries: int
    max_retries: int
    trace: list[str]
