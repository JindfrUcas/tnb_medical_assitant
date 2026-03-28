"""GraphRAG 证据连边与实体匹配辅助函数。

这一层不直接访问数据库，只负责做两件事：
1. 把药物 / 疾病 / 指标名称展开成一组更适合匹配正文的别名。
2. 在用户查询或指南 chunk 文本里，找出命中的图谱实体。

这样脚本导入阶段和运行时检索阶段都能复用同一套实体匹配逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
_SPACE_RE = re.compile(r"\s+")
_ASCII_RE = re.compile(r"^[a-z0-9._+-]+$", re.IGNORECASE)

# 常见药品剂型/修饰词。这里不追求药典级标准化，只做实用的正文别名展开。
_DRUG_PREFIXES = (
    "盐酸",
    "磷酸",
    "枸橼酸",
    "琥珀酸",
    "富马酸",
    "酒石酸",
    "注射用",
    "重组人",
)
_DRUG_SUFFIXES = (
    "普通片",
    "缓释片",
    "控释片",
    "肠溶片",
    "分散片",
    "咀嚼片",
    "胶囊",
    "片",
    "注射液",
    "注射剂",
    "口服液",
    "口服溶液",
    "颗粒",
    "散",
)


@dataclass(frozen=True)
class EntityAlias:
    """一条可用于匹配文本的实体别名。"""

    entity_type: str
    entity_name: str
    alias: str
    alias_key: str


def normalize_match_text(value: str) -> str:
    """把待匹配文本清洗成稳定形式。

    这里会统一去空白、转小写，方便处理中英文混合的查询。
    """
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = _SPACE_RE.sub("", text)
    return text


def build_entity_alias_index(
    drug_names: list[str] | set[str],
    disease_names: list[str] | set[str],
    indicator_names: list[str] | set[str],
) -> list[EntityAlias]:
    """根据实体名称构建一个按别名长度排序的匹配索引。"""
    aliases: dict[tuple[str, str, str], EntityAlias] = {}

    for name in sorted({str(item).strip() for item in drug_names if str(item).strip()}):
        for alias in _generate_drug_aliases(name):
            alias_key = normalize_match_text(alias)
            if not _is_usable_alias(alias, alias_key):
                continue
            aliases[("Drug", name, alias_key)] = EntityAlias("Drug", name, alias, alias_key)

    for name in sorted({str(item).strip() for item in disease_names if str(item).strip()}):
        for alias in _generate_generic_aliases(name):
            alias_key = normalize_match_text(alias)
            if not _is_usable_alias(alias, alias_key):
                continue
            aliases[("Disease", name, alias_key)] = EntityAlias("Disease", name, alias, alias_key)

    for name in sorted({str(item).strip() for item in indicator_names if str(item).strip()}):
        for alias in _generate_indicator_aliases(name):
            alias_key = normalize_match_text(alias)
            if not _is_usable_alias(alias, alias_key):
                continue
            aliases[("Indicator", name, alias_key)] = EntityAlias("Indicator", name, alias, alias_key)

    # 长别名优先，这样能尽量减少短 token 造成的误命中。
    return sorted(
        aliases.values(),
        key=lambda item: (len(item.alias_key), len(item.entity_name)),
        reverse=True,
    )


def extract_entity_matches(text: str, alias_index: list[EntityAlias]) -> dict[str, list[str]]:
    """在一段文本中找出命中的实体名称。

    返回结果按实体类型分组，值为去重后的实体正式名称。
    """
    normalized_text = normalize_match_text(text)
    if not normalized_text:
        return {"Drug": [], "Disease": [], "Indicator": []}

    matches: dict[str, list[str]] = {"Drug": [], "Disease": [], "Indicator": []}
    seen: set[tuple[str, str]] = set()

    for alias in alias_index:
        if alias.alias_key not in normalized_text:
            continue
        key = (alias.entity_type, alias.entity_name)
        if key in seen:
            continue
        seen.add(key)
        matches[alias.entity_type].append(alias.entity_name)

    return matches


def _generate_generic_aliases(name: str) -> set[str]:
    """生成疾病等普通实体的匹配别名。"""
    candidates = {name.strip()}
    compact = _PAREN_RE.sub("", name).strip()
    if compact:
        candidates.add(compact)
    return {item for item in candidates if item}


def _generate_indicator_aliases(name: str) -> set[str]:
    """生成指标实体别名。

    指标本身经常有大小写差异，所以只需保留原名和去括号版本即可。
    """
    candidates = _generate_generic_aliases(name)
    normalized = name.replace(" ", "").strip()
    if normalized:
        candidates.add(normalized)
    return candidates


def _generate_drug_aliases(name: str) -> set[str]:
    """为药物名展开更贴近指南正文的别名。"""
    base = name.strip()
    if not base:
        return set()

    candidates = {base}
    no_paren = _PAREN_RE.sub("", base).strip()
    if no_paren:
        candidates.add(no_paren)

    queue = list(candidates)
    while queue:
        current = queue.pop()
        trimmed = current.strip()
        if not trimmed:
            continue
        for prefix in _DRUG_PREFIXES:
            if trimmed.startswith(prefix) and len(trimmed) > len(prefix) + 1:
                candidate = trimmed[len(prefix) :].strip()
                if candidate and candidate not in candidates:
                    candidates.add(candidate)
                    queue.append(candidate)
        for suffix in _DRUG_SUFFIXES:
            if trimmed.endswith(suffix) and len(trimmed) > len(suffix) + 1:
                candidate = trimmed[: -len(suffix)].strip()
                if candidate and candidate not in candidates:
                    candidates.add(candidate)
                    queue.append(candidate)

    return {item for item in candidates if item}


def _is_usable_alias(alias: str, alias_key: str) -> bool:
    """过滤掉过短或噪声过大的别名。"""
    if not alias or not alias_key:
        return False
    if _ASCII_RE.fullmatch(alias_key):
        return len(alias_key) >= 3
    return len(alias_key) >= 2
