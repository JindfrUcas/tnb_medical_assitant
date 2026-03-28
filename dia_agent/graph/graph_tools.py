"""ReAct 主控循环的 7 个受控工具。

每个工具都是确定性逻辑或受控图查询，模型不能自由编写 Cypher。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from dia_agent.graph.repository import Neo4jGuidelineRepository
from dia_agent.nodes.guardrail import GuardrailNode, GuardrailRepository
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import GuardrailReport, PatientState, RagSnippet


class _ToolContext:
    """工具运行时共享的上下文，在每次问诊开始时创建。"""

    def __init__(
        self,
        patient_state: PatientState,
        guardrail_node: GuardrailNode,
        guardrail_repo: GuardrailRepository,
        guideline_repo: Neo4jGuidelineRepository | None,
        retriever: GuidelineRetriever,
        max_audit_retries: int = 3,
    ):
        self.patient_state = patient_state
        self.guardrail_node = guardrail_node
        self.guardrail_repo = guardrail_repo
        self.guideline_repo = guideline_repo
        self.retriever = retriever
        self.max_audit_retries = max_audit_retries
        # 缓存 guardrail_report 供 audit_self 使用
        self.guardrail_report: GuardrailReport | None = None
        self.audit_count: int = 0
        self.collected_snippets: list[RagSnippet] = []
        self._seen_snippets: set[tuple[str, str]] = set()

    def remember_snippets(self, snippets: list[RagSnippet]) -> None:
        for s in snippets:
            key = (s.source, s.content)
            if key not in self._seen_snippets:
                self._seen_snippets.add(key)
                self.collected_snippets.append(s)


def build_react_tools(ctx: _ToolContext) -> list[BaseTool]:
    """构建 7 个 ReAct 工具，绑定到当前问诊上下文。"""

    @tool("guardrail_check")
    def guardrail_check() -> str:
        """执行确定性红线检查，返回禁忌药物、剂量调整和安全白皮书。必须第一步调用。"""
        report = ctx.guardrail_node.run(ctx.patient_state)
        ctx.guardrail_report = report
        return json.dumps(
            {
                "contraindications": [h.model_dump() for h in report.contraindications],
                "dosage_adjustments": [h.model_dump() for h in report.dosage_adjustments],
                "whitepaper": report.whitepaper,
            },
            ensure_ascii=False,
            indent=2,
        )

    @tool("graph_traverse")
    def graph_traverse(
        entity_name: str,
        entity_type: str = "Drug",
        max_hops: int = 2,
        top_k: int = 5,
    ) -> str:
        """从一个实体出发做 1-2 跳图遍历。
        1跳：找实体关联的指南 chunk。
        2跳：找同 section 相邻 chunk + 通过共享指标发现关联药物（仅 Drug 类型）。
        entity_type 可选 Drug / Disease / Indicator。
        """
        if ctx.guideline_repo is None:
            return "图数据库不可用，请改用 vector_search。"
        try:
            result = ctx.guideline_repo.traverse_from_entity(
                entity_name=entity_name,
                entity_type=entity_type,
                max_hops=max_hops,
                top_k=top_k,
            )
        except Exception as exc:
            return f"图遍历失败: {exc}"

        # 收集 snippets
        for chunk in result.get("chunks", []):
            ctx.remember_snippets([
                RagSnippet(
                    source=str(chunk.get("source", "GRAPH")),
                    content=str(chunk.get("content", "")),
                )
            ])

        return _format_traverse_result(entity_name, result)

    @tool("graph_expand")
    def graph_expand(
        entity_names: list[str],
        min_overlap: int = 2,
        top_k: int = 5,
    ) -> str:
        """给定多个实体名，提取它们共同关联的指南 chunk 和实体间关系路径。
        用于多实体联合推理，例如同时考虑某药物和某疾病的交叉证据。
        """
        if ctx.guideline_repo is None:
            return "图数据库不可用，请改用 vector_search。"
        try:
            result = ctx.guideline_repo.expand_subgraph(
                entity_names=entity_names,
                min_overlap=min_overlap,
                top_k=top_k,
            )
        except Exception as exc:
            return f"子图扩展失败: {exc}"

        for chunk in result.get("shared_chunks", []):
            ctx.remember_snippets([
                RagSnippet(
                    source=str(chunk.get("source", "GRAPH")),
                    content=str(chunk.get("content", "")),
                )
            ])

        return _format_expand_result(entity_names, result)

    @tool("vector_search")
    def vector_search(query: str, top_k: int = 4, source: str = "") -> str:
        """向量相似度检索指南片段。当图遍历结果不足时使用。"""
        snippets = ctx.retriever.retrieve_vector(
            query=query,
            top_k=top_k,
            source=source or None,
        )
        ctx.remember_snippets(snippets)
        if not snippets:
            return f"查询 `{query}` 没有检索到指南片段。"
        lines = [f"查询 `{query}` 检索到 {len(snippets)} 条片段:"]
        for s in snippets:
            preview = s.content.replace("\n", " ").strip()[:240]
            lines.append(f"- [{s.source}] {preview}")
        return "\n".join(lines)

    @tool("get_dosage_info")
    def get_dosage_info(drug_name: str) -> str:
        """查询某药物的剂量调整策略和相关指南 chunk。"""
        # 从 guardrail repo 查剂量调整规则
        rows = ctx.guardrail_repo.adjustments_by_drug(drug_name)
        lines = [f"药物 {drug_name} 的剂量调整规则:"]
        if rows:
            for row in rows:
                note = f"（{row.get('note', '')}）" if row.get("note") else ""
                lines.append(f"- {row.get('condition', '')} → {row.get('action', '')}{note}")
        else:
            lines.append("- 未查到剂量调整规则。")

        # 从图中查该药物相关的剂量 chunk
        if ctx.guideline_repo is not None:
            try:
                traverse = ctx.guideline_repo.traverse_from_entity(
                    entity_name=drug_name, entity_type="Drug", max_hops=1, top_k=3,
                )
                dosage_chunks = [
                    c for c in traverse.get("chunks", [])
                    if any(kw in str(c.get("content", "")) for kw in ("剂量", "用量", "减量", "停用", "调整"))
                ]
                if dosage_chunks:
                    lines.append("相关指南片段:")
                    for c in dosage_chunks[:3]:
                        preview = str(c.get("content", "")).replace("\n", " ").strip()[:200]
                        lines.append(f"- [{c.get('source', '')}] {preview}")
                        ctx.remember_snippets([
                            RagSnippet(source=str(c.get("source", "GRAPH")), content=str(c.get("content", "")))
                        ])
            except Exception:
                pass

        return "\n".join(lines)

    @tool("audit_self")
    def audit_self(recommended_drugs: list[str]) -> str:
        """对推荐药物列表做安全审计。检查是否包含禁用药物。生成建议后必须调用。"""
        ctx.audit_count += 1
        report = ctx.guardrail_report
        if report is None:
            return json.dumps({"passed": False, "violations": [], "feedback": "请先调用 guardrail_check。"})

        forbidden = report.forbidden_drug_names
        violations = [drug for drug in recommended_drugs if drug.lower() in forbidden]

        if violations:
            text = "、".join(violations)
            return json.dumps(
                {
                    "passed": False,
                    "violations": violations,
                    "feedback": f"建议中出现红线禁药：{text}。请删除并改为安全替代方案。",
                    "audit_count": ctx.audit_count,
                    "max_retries": ctx.max_audit_retries,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {"passed": True, "violations": [], "feedback": "", "audit_count": ctx.audit_count},
            ensure_ascii=False,
        )

    @tool("get_chapter_context")
    def get_chapter_context(chunk_key: str, context_window: int = 2) -> str:
        """获取某个指南 chunk 所在的章节标题和前后相邻 chunk，用于理解上下文。"""
        if ctx.guideline_repo is None:
            return "图数据库不可用。"
        try:
            result = ctx.guideline_repo.get_chapter_context(
                chunk_key=chunk_key, context_window=context_window,
            )
        except Exception as exc:
            return f"章节上下文查询失败: {exc}"

        lines = []
        if result.get("document"):
            lines.append(f"文档: {result['document']}")
        if result.get("chapter"):
            lines.append(f"章: {result['chapter']}")
        if result.get("section"):
            lines.append(f"节: {result['section']}")
        if result.get("target_content"):
            lines.append(f"目标 chunk:\n{result['target_content']}")
        neighbors = result.get("neighbors", [])
        if neighbors:
            lines.append(f"相邻 chunk ({len(neighbors)} 条):")
            for n in neighbors:
                preview = str(n.get("content", "")).replace("\n", " ").strip()[:200]
                lines.append(f"- [{n.get('key', '')}] {preview}")
        return "\n".join(lines) or "未找到该 chunk 的上下文信息。"

    return [
        guardrail_check,
        graph_traverse,
        graph_expand,
        vector_search,
        get_dosage_info,
        audit_self,
        get_chapter_context,
    ]


def _format_traverse_result(entity_name: str, result: dict[str, Any]) -> str:
    """格式化图遍历结果为可读文本。"""
    lines = [f"从 {entity_name} 出发的图遍历结果:"]

    chunks = result.get("chunks", [])
    if chunks:
        lines.append(f"直接关联 chunk ({len(chunks)} 条):")
        for c in chunks:
            preview = str(c.get("content", "")).replace("\n", " ").strip()[:200]
            lines.append(f"- [{c.get('source', '')}|{c.get('chapter', '')}|{c.get('section', '')}] {preview}")
    else:
        lines.append("- 未找到直接关联的指南 chunk。")

    siblings = result.get("siblings", [])
    if siblings:
        lines.append(f"同 section 相邻 chunk ({len(siblings)} 条):")
        for c in siblings:
            preview = str(c.get("content", "")).replace("\n", " ").strip()[:200]
            lines.append(f"- [{c.get('chunk_key', '')}] {preview}")

    related = result.get("related_drugs", [])
    if related:
        lines.append("通过共享禁忌指标发现的关联药物:")
        for r in related:
            drugs = ", ".join(r.get("related_drugs", []))
            lines.append(
                f"- 共享指标 {r.get('shared_indicator', '')}"
                f" ({r.get('operator', '')} {r.get('threshold', '')}): {drugs}"
                f" | 原因: {r.get('reason', '')}"
            )

    return "\n".join(lines)


def _format_expand_result(entity_names: list[str], result: dict[str, Any]) -> str:
    """格式化子图扩展结果为可读文本。"""
    lines = [f"实体 {', '.join(entity_names)} 的子图扩展结果:"]

    shared = result.get("shared_chunks", [])
    if shared:
        lines.append(f"共同关联 chunk ({len(shared)} 条):")
        for c in shared:
            entities = ", ".join(c.get("mentioned_entities", []))
            preview = str(c.get("content", "")).replace("\n", " ").strip()[:200]
            lines.append(f"- [命中实体: {entities}] {preview}")
    else:
        lines.append("- 未找到共同关联的 chunk。")

    relations = result.get("relations", [])
    if relations:
        lines.append("实体间直接关系:")
        for r in relations:
            props = r.get("properties", {})
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items()) if props else ""
            suffix = f" ({prop_str})" if prop_str else ""
            lines.append(f"- {r.get('from_entity', '')} --[{r.get('relation', '')}]--> {r.get('to_entity', '')}{suffix}")

    return "\n".join(lines)
