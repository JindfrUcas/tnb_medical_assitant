"""基于图谱红线数据生成 SFT 训练样本。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _violating_value(operator: str, threshold: float) -> float:
    """构造一个能够稳定触发阈值违规的模拟患者值。"""
    if operator == "<":
        return threshold - 5
    if operator == "<=":
        return threshold - 1
    if operator == ">":
        return threshold + 5
    if operator == ">=":
        return threshold + 1
    return threshold


def _safe_float(value: Any, default: float = 0.0) -> float:
    """尽量把输入安全转换成浮点数，失败时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_records(graph_json_path: Path) -> list[dict[str, Any]]:
    """读取 graph.json，并过滤掉非对象条目。"""
    payload = json.loads(graph_json_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("graph.json must be a list")
    return [item for item in payload if isinstance(item, dict)]


def _build_prompt(
    indicator_name: str,
    indicator_value: float,
    drug_name: str,
    reason: str,
    diseases: list[str],
) -> str:
    """构造一条用于训练的用户侧问诊提示。"""
    diseases_text = "、".join(diseases) if diseases else "无"
    return (
        "患者结构化信息如下：\n"
        f"- 指标: {indicator_name}={indicator_value}\n"
        f"- 合并疾病: {diseases_text}\n"
        f"- 当前用药: {drug_name}\n"
        "请给出降糖治疗建议，并说明安全理由。"
        f"（临床红线提示：{reason}）"
    )


def _positive_answer(drug_name: str, reason: str) -> str:
    """生成遵守红线的正样本答案。"""
    return (
        f"建议停用或避免 {drug_name}。"
        f"原因是已触发红线禁忌：{reason}。"
        "在此基础上，优先选择不触发禁忌的替代路径，并安排肝肾功能与血糖复评。"
    )


def _negative_answer(drug_name: str) -> str:
    """生成故意忽略红线的负样本答案。"""
    return (
        f"建议继续使用 {drug_name} 作为首选降糖药。"
        "可先不考虑禁忌阈值，按常规指南剂量推进。"
    )


def generate_sft_dataset(
    graph_json_path: Path,
    output_path: Path,
    total_samples: int = 2000,
    seed: int = 42,
) -> int:
    """批量生成 SFT 数据集，并写成 JSONL 文件。"""
    records = _load_records(graph_json_path)
    rng = random.Random(seed)

    examples: list[dict[str, Any]] = []
    cursor = 1

    for record in records:
        drug_name = str(record.get("drug_name", "")).strip()
        diseases = [str(item).strip() for item in (record.get("disease_excludes") or []) if str(item).strip()]
        contraindications = record.get("contraindications") or []
        for item in contraindications:
            if not isinstance(item, dict):
                continue
            indicator_name = str(item.get("indicator", "")).strip()
            operator = str(item.get("operator", "")).strip()
            threshold = _safe_float(item.get("threshold"), 0.0)
            reason = str(item.get("reason", "")).strip() or "触发禁忌阈值"
            indicator_value = _violating_value(operator, threshold)
            user_prompt = _build_prompt(indicator_name, indicator_value, drug_name, reason, diseases)

            positive = {
                "id": f"sft-{cursor:05d}",
                "label": "positive",
                "messages": [
                    {"role": "system", "content": "你是糖尿病专病助手，必须红线优先。"},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": _positive_answer(drug_name, reason)},
                ],
                "meta": {
                    "drug_name": drug_name,
                    "indicator": indicator_name,
                    "operator": operator,
                    "threshold": threshold,
                },
            }
            cursor += 1

            negative = {
                "id": f"sft-{cursor:05d}",
                "label": "negative",
                "messages": [
                    {"role": "system", "content": "你是糖尿病专病助手。"},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": _negative_answer(drug_name)},
                ],
                "meta": {
                    "drug_name": drug_name,
                    "indicator": indicator_name,
                    "operator": operator,
                    "threshold": threshold,
                },
            }
            cursor += 1

            examples.extend([positive, negative])

    if not examples:
        raise ValueError("No SFT examples generated from graph.json")

    if total_samples < len(examples):
        examples = rng.sample(examples, total_samples)
    elif total_samples > len(examples):
        extra = [rng.choice(examples) for _ in range(total_samples - len(examples))]
        examples.extend(extra)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for index, row in enumerate(examples, start=1):
            item = dict(row)
            item["id"] = f"sft-{index:05d}"
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(examples)
