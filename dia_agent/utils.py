"""项目公共工具函数。

把多个模块中重复出现的逻辑收口到这里，避免各处独立维护。
"""

from __future__ import annotations

from typing import Any

from dia_agent.schemas import PatientState, RagSnippet

DEFAULT_RAG_FALLBACK_QUERY = "1型糖尿病 指南 用药"


def normalize_message_content(content: Any) -> str:
    """把 LangChain 消息内容压平成普通文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def build_default_query(patient_state: PatientState) -> str:
    """当外部未传入检索词时，根据患者状态拼默认 RAG 查询。"""
    indicator_tokens = [f"{name}:{value}" for name, value in patient_state.indicators.items()]
    disease_tokens = patient_state.diseases
    return " ".join(indicator_tokens + disease_tokens) or DEFAULT_RAG_FALLBACK_QUERY


def dedupe_snippets(snippets: list[RagSnippet], top_k: int | None = None) -> list[RagSnippet]:
    """按 source + content 去重 RagSnippet 列表。"""
    seen: set[tuple[str, str]] = set()
    result: list[RagSnippet] = []
    for snippet in snippets:
        key = (snippet.source, snippet.content)
        if key in seen:
            continue
        seen.add(key)
        result.append(snippet)
        if top_k is not None and len(result) >= top_k:
            break
    return result
