"""Dia-Agent 的 LangGraph 工作流编排层。

重构后只保留两个节点：perception（确定性输入标准化）+ react_agent（ReAct 主控循环）。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from dia_agent.nodes.perception import PerceptionNode
from dia_agent.nodes.react_controller import ReactControllerNode
from dia_agent.schemas import ConsultationOutput
from dia_agent.workflow.state import DiaAgentState


class DiaAgentWorkflow:
    """把 perception 和 react_agent 编排成一条可执行工作流。"""

    def __init__(
        self,
        perception_node: PerceptionNode,
        react_controller: ReactControllerNode,
    ):
        self._perception = perception_node
        self._react_controller = react_controller
        self._graph = self._compile()

    def invoke(self, raw_input: str | dict[str, Any], rag_query: str = "", history_text: str = "") -> ConsultationOutput:
        """执行一次完整问诊，并返回聚合后的最终结果。"""
        initial_state: DiaAgentState = {
            "raw_input": raw_input,
            "rag_query": rag_query,
            "history_text": history_text,
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
        builder.add_node("react_agent", self._run_react_agent)
        builder.add_edge(START, "perception")
        builder.add_edge("perception", "react_agent")
        builder.add_edge("react_agent", END)
        return builder.compile()

    def _run_perception(self, state: DiaAgentState) -> DiaAgentState:
        """把原始输入标准化成 PatientState。"""
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

    def _run_react_agent(self, state: DiaAgentState) -> DiaAgentState:
        """执行 ReAct 主控循环。"""
        output = self._react_controller.run(
            patient_state=state["patient_state"],
            rag_query=state.get("rag_query", ""),
        )
        trace = list(state.get("trace", []))
        trace.extend(output.trace)
        return {
            "guardrail_report": output.guardrail_report,
            "reasoner_result": output.reasoner_result,
            "audit_result": output.audit_result,
            "retries": output.retries,
            "trace": trace,
        }
