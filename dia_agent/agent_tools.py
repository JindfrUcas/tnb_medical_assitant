"""ReAct 推理器可调用的受控工具集合。

模型只能决定“下一步调用哪个工具”，真正的查询与规则判断仍然由
Python 代码执行，从而把医疗安全边界保留在确定性逻辑里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from dia_agent.nodes.guardrail import GuardrailRepository, _compare
from dia_agent.rag.retriever import GuidelineRetriever
from dia_agent.schemas import GuardrailReport, PatientState, RagSnippet


@dataclass
class ToolExecution:
    """执行单个受控工具后的结果。"""

    tool_name: str
    observation: str
    snippets: list[RagSnippet] = field(default_factory=list)


class AgentToolbox:
    """提供给 ReAct 推理器的小型受控工具面。"""

    def __init__(
        self,
        patient_state: PatientState,
        guardrail_report: GuardrailReport,
        repository: GuardrailRepository,
        retriever: GuidelineRetriever,
        default_query: str = "",
        default_top_k: int = 4,
    ):
        """初始化工具箱，并缓存本轮推理需要的上下文对象。"""
        self._patient_state = patient_state
        self._guardrail_report = guardrail_report
        self._repository = repository
        self._retriever = retriever
        self._default_query = default_query.strip()
        self._default_top_k = max(1, default_top_k)
        self._used_rag_snippets: list[RagSnippet] = []
        self._seen_snippets: set[tuple[str, str]] = set()
        self._tool_calls: list[dict[str, Any]] = []

    @property
    def used_rag_snippets(self) -> list[RagSnippet]:
        """返回当前工具链路中实际使用过的指南片段。"""
        return list(self._used_rag_snippets)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """返回已执行工具的轻量审计记录。"""
        return list(self._tool_calls)

    def seed_snippets(self, snippets: list[RagSnippet]) -> None:
        """把已知片段预先放入工具箱，避免重复检索。"""
        self._remember_snippets(snippets)

    def describe_tools(self) -> str:
        """生成写入 Prompt 的人类可读工具清单。"""
        return "\n".join(
            [
                "- get_patient_state: 查看当前结构化患者状态。",
                "- get_guardrail_report: 查看完整安全白皮书与已命中的红线。",
                "- search_indicator_rules: 按指标名查询相关禁忌规则，如 eGFR。",
                "- search_disease_excludes: 按疾病名查询排除药物。",
                "- search_dosage_adjustments: 按药物名查询剂量调整规则。",
                "- retrieve_guidelines: 检索指南片段，内部会优先走图证据，再补向量召回；参数可包含 query/top_k/source。",
                "- list_safe_current_drugs: 列出当前用药中未命中禁忌的部分。",
            ]
        )

    def execute(self, action: str, action_input: dict[str, Any] | None = None) -> ToolExecution:
        """执行单个受控工具，并记录一条轻量审计日志。"""
        payload = action_input or {}
        tool_name = self._normalize_action_name(action)
        try:
            if tool_name == "get_patient_state":
                execution = ToolExecution(
                    tool_name=tool_name,
                    observation=json.dumps(self._patient_state.model_dump(), ensure_ascii=False, indent=2),
                )
            elif tool_name == "get_guardrail_report":
                execution = ToolExecution(tool_name=tool_name, observation=self._build_guardrail_observation())
            elif tool_name == "search_indicator_rules":
                execution = ToolExecution(
                    tool_name=tool_name,
                    observation=self._search_indicator_rules(str(payload.get("indicator_name") or "")),
                )
            elif tool_name == "search_disease_excludes":
                execution = ToolExecution(
                    tool_name=tool_name,
                    observation=self._search_disease_excludes(str(payload.get("disease_name") or "")),
                )
            elif tool_name == "search_dosage_adjustments":
                execution = ToolExecution(
                    tool_name=tool_name,
                    observation=self._search_dosage_adjustments(str(payload.get("drug_name") or "")),
                )
            elif tool_name == "retrieve_guidelines":
                query = str(payload.get("query") or self._default_query).strip()
                source = str(payload.get("source") or "").strip() or None
                top_k = self._clamp_top_k(payload.get("top_k"), default=self._default_top_k)
                snippets = self._retriever.retrieve(query=query, top_k=top_k, source=source)
                self._remember_snippets(snippets)
                execution = ToolExecution(
                    tool_name=tool_name,
                    observation=self._format_guideline_observation(query, snippets),
                    snippets=snippets,
                )
            elif tool_name == "list_safe_current_drugs":
                execution = ToolExecution(tool_name=tool_name, observation=self._list_safe_current_drugs())
            else:
                execution = ToolExecution(
                    tool_name=tool_name,
                    observation=(
                        "未知工具。允许的工具有: get_patient_state, get_guardrail_report, "
                        "search_indicator_rules, search_disease_excludes, search_dosage_adjustments, "
                        "retrieve_guidelines, list_safe_current_drugs。"
                    ),
                )
        except Exception as exc:
            execution = ToolExecution(tool_name=tool_name, observation=f"工具执行失败: {exc}")

        self._tool_calls.append(
            {
                "tool": tool_name,
                "input": payload,
                "observation_preview": execution.observation[:240],
            }
        )
        return execution

    def _normalize_action_name(self, action: str) -> str:
        """把模型给出的动作别名统一映射成标准工具名。"""
        token = action.strip().lower()
        aliases = {
            "patient_state": "get_patient_state",
            "guardrail": "get_guardrail_report",
            "guardrail_report": "get_guardrail_report",
            "indicator_rules": "search_indicator_rules",
            "disease_rules": "search_disease_excludes",
            "dosage_rules": "search_dosage_adjustments",
            "guidelines": "retrieve_guidelines",
            "guideline_search": "retrieve_guidelines",
            "safe_current_drugs": "list_safe_current_drugs",
        }
        return aliases.get(token, token)

    def _build_guardrail_observation(self) -> str:
        """把 Guardrail 报告格式化成适合模型阅读的文本。"""
        sections = ["已命中的结构化红线摘要:"]
        if self._guardrail_report.contraindications:
            for item in self._guardrail_report.contraindications:
                sections.append(f"- 禁用 {item.drug_name}: {item.reason}")
        else:
            sections.append("- 当前未命中硬性禁忌。")

        if self._guardrail_report.dosage_adjustments:
            sections.append("已命中的剂量调整:")
            for item in self._guardrail_report.dosage_adjustments:
                note = f"（{item.note}）" if item.note else ""
                sections.append(f"- {item.drug_name}: {item.condition} -> {item.action}{note}")

        sections.append("完整白皮书:")
        sections.append(self._guardrail_report.whitepaper)
        return "\n".join(sections)

    def _search_indicator_rules(self, indicator_name: str) -> str:
        """查询某个指标对应的禁忌规则，并标出当前是否命中。"""
        indicator_name = indicator_name.strip()
        if not indicator_name:
            return "缺少 indicator_name。"

        rows = self._repository.contraindications_by_indicator(indicator_name)
        if not rows:
            return f"没有查到指标 {indicator_name} 的禁忌规则。"

        patient_value = self._lookup_indicator_value(indicator_name)
        lines = [f"指标 {indicator_name} 的规则如下:"]
        if patient_value is None:
            lines.append("- 当前患者没有这个指标的结构化值。")
        else:
            lines.append(f"- 当前患者 {indicator_name} = {patient_value}")

        for row in rows:
            operator = str(row.get("operator", "")).strip()
            threshold = row.get("threshold")
            reason = str(row.get("reason", "未提供原因")).strip() or "未提供原因"
            drug_name = str(row.get("drug_name", "")).strip()
            active = patient_value is not None and _compare(patient_value, operator, threshold)
            lines.append(
                f"- {drug_name}: 条件 {indicator_name} {operator} {threshold}，"
                f"当前是否命中: {'是' if active else '否'}，原因: {reason}"
            )
        return "\n".join(lines)

    def _search_disease_excludes(self, disease_name: str) -> str:
        """查询某个疾病对应的排除药物规则。"""
        disease_name = disease_name.strip()
        if not disease_name:
            return "缺少 disease_name。"

        rows = self._repository.excludes_by_disease(disease_name)
        if not rows:
            return f"没有查到疾病 {disease_name} 的排除药物。"

        current_hit = disease_name.lower() in {item.lower() for item in self._patient_state.diseases}
        lines = [f"疾病 {disease_name} 的排除药物如下:", f"- 当前患者是否包含该疾病: {'是' if current_hit else '否'}"]
        for row in rows:
            lines.append(f"- 排除药物: {str(row.get('drug_name', '')).strip()}")
        return "\n".join(lines)

    def _search_dosage_adjustments(self, drug_name: str) -> str:
        """查询某个药物的剂量调整规则，并展示当前命中情况。"""
        drug_name = drug_name.strip()
        if not drug_name:
            return "缺少 drug_name。"

        rows = self._repository.adjustments_by_drug(drug_name)
        if not rows:
            return f"没有查到药物 {drug_name} 的剂量调整规则。"

        matched = [
            item
            for item in self._guardrail_report.dosage_adjustments
            if item.drug_name.strip().lower() == drug_name.lower()
        ]
        lines = [f"药物 {drug_name} 的剂量调整规则如下:"]
        if matched:
            lines.append("- 当前患者已命中的剂量调整:")
            for item in matched:
                note = f"（{item.note}）" if item.note else ""
                lines.append(f"  {item.condition} -> {item.action}{note}")
        else:
            lines.append("- 当前 Guardrail 报告中尚未命中该药物的剂量调整。")

        lines.append("- 原始规则:")
        for row in rows:
            note = str(row.get("note", "")).strip()
            suffix = f"（{note}）" if note else ""
            lines.append(f"  {str(row.get('condition', '')).strip()} -> {str(row.get('action', '')).strip()}{suffix}")
        return "\n".join(lines)

    def _format_guideline_observation(self, query: str, snippets: list[RagSnippet]) -> str:
        """把检索得到的指南片段整理成易于阅读的观察结果。"""
        if not snippets:
            return f"查询 `{query}` 没有检索到指南片段。"

        lines = [f"查询 `{query}` 检索到 {len(snippets)} 条指南片段:"]
        for snippet in snippets:
            preview = snippet.content.replace("\n", " ").strip()
            lines.append(f"- [{snippet.source}] {preview[:240]}")
        return "\n".join(lines)

    def _list_safe_current_drugs(self) -> str:
        """列出当前用药里尚未命中禁忌的部分。"""
        forbidden = {item.drug_name.lower() for item in self._guardrail_report.contraindications}
        safe_drugs = [
            drug for drug in self._patient_state.current_drugs if drug.strip().lower() not in forbidden
        ]
        if not safe_drugs:
            return "当前用药中没有可直接视为安全延续的药物，建议重新评估全套方案。"
        return f"当前未命中禁忌、可继续评估的用药: {', '.join(safe_drugs)}"

    def _lookup_indicator_value(self, indicator_name: str) -> float | None:
        """按不区分大小写的方式查找患者某项指标值。"""
        for name, value in self._patient_state.indicators.items():
            if name.strip().lower() == indicator_name.lower():
                return float(value)
        return None

    def _remember_snippets(self, snippets: list[RagSnippet]) -> None:
        """去重保存片段，避免同一条 RAG 内容重复进入上下文。"""
        for snippet in snippets:
            key = (snippet.source, snippet.content)
            if key in self._seen_snippets:
                continue
            self._seen_snippets.add(key)
            self._used_rag_snippets.append(snippet)

    def _clamp_top_k(self, value: Any, default: int = 4) -> int:
        """限制检索条数，避免模型一次请求过多上下文。"""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, min(parsed, 6))


def build_langchain_tools(toolbox: AgentToolbox) -> list[BaseTool]:
    """把受控工具箱包装成 LangChain 标准 `@tool` 工具集合。"""

    @tool("get_patient_state")
    def get_patient_state() -> str:
        """查看当前结构化患者状态。"""
        return toolbox.execute("get_patient_state").observation

    @tool("get_guardrail_report")
    def get_guardrail_report() -> str:
        """查看当前患者已命中的红线摘要与完整安全白皮书。"""
        return toolbox.execute("get_guardrail_report").observation

    @tool("search_indicator_rules")
    def search_indicator_rules(indicator_name: str) -> str:
        """按指标名查询禁忌规则，例如 eGFR、HbA1c。"""
        return toolbox.execute("search_indicator_rules", {"indicator_name": indicator_name}).observation

    @tool("search_disease_excludes")
    def search_disease_excludes(disease_name: str) -> str:
        """按疾病名查询排除药物规则。"""
        return toolbox.execute("search_disease_excludes", {"disease_name": disease_name}).observation

    @tool("search_dosage_adjustments")
    def search_dosage_adjustments(drug_name: str) -> str:
        """按药物名查询剂量调整规则。"""
        return toolbox.execute("search_dosage_adjustments", {"drug_name": drug_name}).observation

    @tool("retrieve_guidelines")
    def retrieve_guidelines(query: str = "", top_k: int = 4, source: str = "") -> str:
        """检索相关指南片段，可选指定检索条数和来源标签。"""
        payload = {
            "query": query,
            "top_k": top_k,
            "source": source,
        }
        return toolbox.execute("retrieve_guidelines", payload).observation

    @tool("list_safe_current_drugs")
    def list_safe_current_drugs() -> str:
        """列出当前用药中未命中禁忌、仍可继续评估的药物。"""
        return toolbox.execute("list_safe_current_drugs").observation

    return [
        get_patient_state,
        get_guardrail_report,
        search_indicator_rules,
        search_disease_excludes,
        search_dosage_adjustments,
        retrieve_guidelines,
        list_safe_current_drugs,
    ]
