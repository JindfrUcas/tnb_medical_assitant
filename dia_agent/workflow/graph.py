"""LangGraph orchestration for Dia-Agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from dia_agent.nodes.auditor import AuditorNode
from dia_agent.nodes.guardrail import GuardrailNode
from dia_agent.nodes.perception import PerceptionNode
from dia_agent.nodes.reasoner import ReasonerNode
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import ConsultationOutput
from dia_agent.workflow.state import DiaAgentState


class DiaAgentWorkflow:
    def __init__(
        self,
        perception_node: PerceptionNode,
        guardrail_node: GuardrailNode,
        reasoner_node: ReasonerNode,
        auditor_node: AuditorNode,
        retriever: GuidelineRetriever,
        max_retries: int = 3,
    ):
        self._perception = perception_node
        self._guardrail = guardrail_node
        self._reasoner = reasoner_node
        self._auditor = auditor_node
        self._retriever = retriever
        self._max_retries = max_retries
        self._graph = self._compile()

    def invoke(self, raw_input: str | dict[str, Any], rag_query: str = "", history_text: str = "") -> ConsultationOutput:
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
        builder = StateGraph(DiaAgentState)

        builder.add_node("perception", self._run_perception)
        builder.add_node("guardrail", self._run_guardrail)
        builder.add_node("reasoner", self._run_reasoner)
        builder.add_node("auditor", self._run_auditor)

        builder.add_edge(START, "perception")
        builder.add_edge("perception", "guardrail")
        builder.add_edge("guardrail", "reasoner")
        builder.add_edge("reasoner", "auditor")
        builder.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {
                "retry": "reasoner",
                "end": END,
            },
        )

        return builder.compile()

    def _run_perception(self, state: DiaAgentState) -> DiaAgentState:
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
        report = self._guardrail.run(state["patient_state"])
        trace = list(state.get("trace", []))
        trace.append("Guardrail: 完成红线约束检索")
        return {
            "guardrail_report": report,
            "trace": trace,
        }

    def _run_reasoner(self, state: DiaAgentState) -> DiaAgentState:
        patient_state = state["patient_state"]
        query = state.get("rag_query", "").strip()
        if not query:
            query = self._build_default_query(patient_state)

        rag_snippets = self._retriever.retrieve(query, top_k=4)
        feedback = ""
        audit = state.get("audit_result")
        if audit and not audit.passed:
            feedback = audit.feedback

        result = self._reasoner.run(
            patient_state=patient_state,
            guardrail_report=state["guardrail_report"],
            rag_snippets=rag_snippets,
            feedback=feedback,
        )
        trace = list(state.get("trace", []))
        trace.append("Reasoner: 生成候选诊疗建议")
        return {
            "rag_snippets": rag_snippets,
            "reasoner_result": result,
            "trace": trace,
        }

    def _run_auditor(self, state: DiaAgentState) -> DiaAgentState:
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
        audit = state["audit_result"]
        retries = int(state.get("retries", 0))
        max_retries = int(state.get("max_retries", self._max_retries))
        if not audit.passed and retries <= max_retries:
            return "retry"
        return "end"

    def _build_default_query(self, patient_state) -> str:
        indicator_tokens = [f"{name}:{value}" for name, value in patient_state.indicators.items()]
        disease_tokens = patient_state.diseases
        return " ".join(indicator_tokens + disease_tokens) or "1型糖尿病 指南 用药"
