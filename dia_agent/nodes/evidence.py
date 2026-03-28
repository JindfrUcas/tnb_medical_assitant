"""Evidence 节点：把 Guardrail 约束和检索证据组装成统一证据包。"""

from __future__ import annotations

from dataclasses import dataclass, field

from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import EvidenceBundle, GuardrailReport, PatientState, RagSnippet


@dataclass
class EvidenceAssemblyOutput:
    """Evidence 节点的结构化执行结果。"""

    evidence_bundle: EvidenceBundle
    trace_lines: list[str] = field(default_factory=list)


class EvidenceAssemblyNode:
    """在 Reasoner 之前统一准备图证据与向量证据。"""

    def __init__(self, retriever: GuidelineRetriever, retrieval_k: int = 4):
        """初始化证据节点。"""
        self._retriever = retriever
        self._retrieval_k = max(1, retrieval_k)

    def run(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        user_query: str = "",
        feedback: str = "",
    ) -> EvidenceAssemblyOutput:
        """围绕当前患者状态和审计反馈，组装一轮可解释证据包。"""
        query_plan = self._build_query_plan(
            patient_state=patient_state,
            guardrail_report=guardrail_report,
            user_query=user_query,
            feedback=feedback,
        )

        graph_snippets: list[RagSnippet] = []
        vector_snippets: list[RagSnippet] = []
        for query in query_plan:
            graph_snippets.extend(self._retriever.retrieve_graph(query=query, top_k=self._retrieval_k))
            vector_snippets.extend(self._retriever.retrieve_vector(query=query, top_k=self._retrieval_k))

        graph_snippets = self._dedupe_snippets(graph_snippets, top_k=self._retrieval_k * 2)
        vector_snippets = self._dedupe_snippets(vector_snippets, top_k=self._retrieval_k * 2)
        merged_snippets = self._dedupe_snippets(
            graph_snippets + vector_snippets,
            top_k=max(self._retrieval_k, self._retrieval_k * 2),
        )

        summary = self._build_summary(query_plan, graph_snippets, vector_snippets, merged_snippets, feedback)
        bundle = EvidenceBundle(
            query_plan=query_plan,
            graph_snippets=graph_snippets,
            vector_snippets=vector_snippets,
            merged_snippets=merged_snippets,
            summary=summary,
        )
        return EvidenceAssemblyOutput(
            evidence_bundle=bundle,
            trace_lines=[
                f"Evidence: 生成查询计划 {len(query_plan)} 条",
                f"Evidence: 图证据 {len(graph_snippets)} 条，向量证据 {len(vector_snippets)} 条，合并后 {len(merged_snippets)} 条",
            ],
        )

    def _build_query_plan(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        user_query: str,
        feedback: str,
    ) -> list[str]:
        """根据患者状态构造一组更稳定的临床检索查询。"""
        queries: list[str] = []

        if user_query.strip():
            queries.append(user_query.strip())

        indicator_tokens = [f"{name} {value}" for name, value in patient_state.indicators.items()]
        disease_tokens = list(patient_state.diseases)
        drug_tokens = list(patient_state.current_drugs)
        if indicator_tokens or disease_tokens or drug_tokens:
            queries.append(" ".join(indicator_tokens + disease_tokens + drug_tokens))

        for hit in guardrail_report.contraindications[:3]:
            queries.append(f"{hit.drug_name} {hit.trigger_name} 用药")

        for item in guardrail_report.dosage_adjustments[:3]:
            queries.append(f"{item.drug_name} {item.condition} 剂量调整")

        if feedback.strip():
            queries.append(feedback.strip())
            for hit in guardrail_report.contraindications[:2]:
                queries.append(f"{hit.drug_name} 安全替代方案 {hit.trigger_name}")

        if not queries:
            queries.append("1型糖尿病 指南 用药")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            token = " ".join(query.split()).strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(token)
        return deduped[:6]

    def _dedupe_snippets(self, snippets: list[RagSnippet], top_k: int) -> list[RagSnippet]:
        """按来源和内容去重证据片段。"""
        unique: list[RagSnippet] = []
        seen: set[tuple[str, str]] = set()
        for snippet in snippets:
            key = (snippet.source, snippet.content)
            if key in seen:
                continue
            seen.add(key)
            unique.append(snippet)
            if len(unique) >= top_k:
                break
        return unique

    def _build_summary(
        self,
        query_plan: list[str],
        graph_snippets: list[RagSnippet],
        vector_snippets: list[RagSnippet],
        merged_snippets: list[RagSnippet],
        feedback: str,
    ) -> str:
        """把本轮证据准备结果压缩成一段短摘要。"""
        lines = ["证据包摘要:"]
        lines.append(f"- 查询计划数: {len(query_plan)}")
        lines.append(f"- 图证据数: {len(graph_snippets)}")
        lines.append(f"- 向量证据数: {len(vector_snippets)}")
        lines.append(f"- 去重后证据数: {len(merged_snippets)}")
        if feedback.strip():
            lines.append(f"- 本轮带修复反馈: {feedback.strip()}")
        return "\n".join(lines)
