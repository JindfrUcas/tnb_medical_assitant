"""Dia-Agent 的 LangGraph 工作流编排层。

这里定义了节点执行顺序，以及审计失败时如何回流到 Reasoner 重新生成建议。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from dia_agent.nodes.auditor import AuditorNode
from dia_agent.nodes.evidence import EvidenceAssemblyNode
from dia_agent.nodes.guardrail import GuardrailNode
from dia_agent.nodes.perception import PerceptionNode
from dia_agent.nodes.reasoner import ReasonerNode, ReasonerRunOutput
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import ConsultationOutput
from dia_agent.utils import build_default_query
from dia_agent.workflow.state import DiaAgentState


class DiaAgentWorkflow:
    """把多个节点编排成一条可执行工作流。"""

    def __init__(
        self,
        perception_node: PerceptionNode,
        guardrail_node: GuardrailNode,
        evidence_node: EvidenceAssemblyNode,
        reasoner_node: ReasonerNode,
        auditor_node: AuditorNode,
        retriever: GuidelineRetriever,
        max_retries: int = 3,
    ):
        """初始化工作流依赖，并立即编译 LangGraph 执行图。"""
        self._perception = perception_node
        self._guardrail = guardrail_node
        self._evidence = evidence_node
        self._reasoner = reasoner_node
        self._auditor = auditor_node
        self._retriever = retriever
        self._max_retries = max_retries
        self._graph = self._compile()

    def invoke(self, raw_input: str | dict[str, Any], rag_query: str = "", history_text: str = "") -> ConsultationOutput:
        """执行一次完整问诊，并返回聚合后的最终结果。"""
        initial_state: DiaAgentState = {
            "raw_input": raw_input,
            "rag_query": rag_query,
            "history_text": history_text,
            "retries": 0,
            "max_retries": self._max_retries,
            "trace": [],
        }
        final_state = self._graph.invoke(initial_state)
        return ConsultationOutput(
            patient_state=final_state["patient_state"],
            guardrail_report=final_state["guardrail_report"],
            reasoner_result=final_state["reasoner_result"],
            audit_result=final_state["audit_result"],
            retries=final_state.get("retries", 0),
            trace=final_state.get("trace", []),
        )

    def _compile(self):
        """定义节点与边，并编译成可执行图。"""
        builder = StateGraph(DiaAgentState)

        builder.add_node("perception", self._run_perception)
        builder.add_node("guardrail", self._run_guardrail)
        builder.add_node("evidence", self._run_evidence)
        builder.add_node("reasoner", self._run_reasoner)
        builder.add_node("auditor", self._run_auditor)

        # 主执行链路：先感知，再查红线，再组装证据，最后推理和审计。
        builder.add_edge(START, "perception")
        builder.add_edge("perception", "guardrail")
        builder.add_edge("guardrail", "evidence")
        builder.add_edge("evidence", "reasoner")
        builder.add_edge("reasoner", "auditor")

        # 如果审计未通过，则先回流到 evidence 重组证据，再进入 reasoner；否则直接结束。
        builder.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {
                "retry": "evidence",
                "end": END,
            },
        )

        return builder.compile()

    def _run_perception(self, state: DiaAgentState) -> DiaAgentState:
        """把原始输入标准化成 `PatientState`。"""
        patient_state = self._perception.run(
            state["raw_input"],
            history_text=state.get("history_text", ""),
        )
        trace = list(state.get("trace", []))
        trace.append("Perception: 完成患者状态结构化")
        return {
            "patient_state": patient_state,
            "trace": trace,
        }

    def _run_guardrail(self, state: DiaAgentState) -> DiaAgentState:
        """根据患者状态检索命中的安全红线。"""
        report = self._guardrail.run(state["patient_state"])
        trace = list(state.get("trace", []))
        trace.append("Guardrail: 完成红线约束检索")
        return {
            "guardrail_report": report,
            "trace": trace,
        }

    def _run_reasoner(self, state: DiaAgentState) -> DiaAgentState:
        """融合患者状态、红线和 RAG 片段生成候选建议。"""
        bundle = state.get("evidence_bundle")
        snippets = bundle.merged_snippets if bundle is not None else state.get("rag_snippets", [])
        patient_state = state["patient_state"]
        query = state.get("rag_query", "").strip()
        if not query:
            query = self._build_default_query(patient_state)

        feedback = ""
        audit = state.get("audit_result")
        if audit and not audit.passed:
            feedback = audit.feedback

        execution = self._reasoner.run(
            patient_state=patient_state,
            guardrail_report=state["guardrail_report"],
            evidence_bundle=bundle,
            rag_snippets=snippets,
            feedback=feedback,
            default_query=query,
        )
        if isinstance(execution, ReasonerRunOutput):
            rag_snippets = execution.rag_snippets
            result = execution.reasoner_result
            tool_calls = execution.tool_calls
            agent_scratchpad = execution.agent_scratchpad
        else:
            rag_snippets = state.get("rag_snippets", [])
            result = execution
            tool_calls = []
            agent_scratchpad = []

        trace = list(state.get("trace", []))
        trace.append("Reasoner: 生成候选诊疗建议")
        for index, call in enumerate(tool_calls, start=1):
            tool_name = str(call.get("tool", "")).strip() or "unknown"
            trace.append(f"Reasoner Tool {index}: {tool_name}")
        return {
            "rag_snippets": rag_snippets,
            "reasoner_result": result,
            "tool_calls": tool_calls,
            "agent_scratchpad": agent_scratchpad,
            "trace": trace,
        }

    def _run_evidence(self, state: DiaAgentState) -> DiaAgentState:
        """组装本轮推理要使用的图证据与向量证据。"""
        feedback = ""
        audit = state.get("audit_result")
        if audit and not audit.passed:
            feedback = audit.repair_focus or audit.feedback

        execution = self._evidence.run(
            patient_state=state["patient_state"],
            guardrail_report=state["guardrail_report"],
            user_query=state.get("rag_query", ""),
            feedback=feedback,
        )
        trace = list(state.get("trace", []))
        trace.extend(execution.trace_lines)
        return {
            "evidence_bundle": execution.evidence_bundle,
            "rag_snippets": execution.evidence_bundle.merged_snippets,
            "trace": trace,
        }

    def _run_auditor(self, state: DiaAgentState) -> DiaAgentState:
        """检查 Reasoner 输出是否违反红线，必要时增加重试次数。"""
        audit = self._auditor.run(state["reasoner_result"], state["guardrail_report"])
        retries = int(state.get("retries", 0))
        if not audit.passed:
            retries += 1

        trace = list(state.get("trace", []))
        if audit.passed:
            trace.append("Auditor: 审计通过")
        else:
            trace.append(f"Auditor: 发现违规，触发重试 {retries}")

        return {
            "audit_result": audit,
            "retries": retries,
            "trace": trace,
        }

    def _route_after_audit(self, state: DiaAgentState) -> str:
        """根据审计结果决定下一跳：重试还是结束。"""
        audit = state["audit_result"]
        retries = int(state.get("retries", 0))
        max_retries = int(state.get("max_retries", self._max_retries))
        if not audit.passed and retries <= max_retries:
            return "retry"
        return "end"

    def _build_default_query(self, patient_state) -> str:
        """当用户没提供检索词时，自动从患者状态拼一个 RAG 查询。"""
        return build_default_query(patient_state)
