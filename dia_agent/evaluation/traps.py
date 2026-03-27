"""生成并读取用于安全拦截测试的陷阱样本。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _violate_threshold(operator: str, threshold: float) -> float:
    """按比较符号构造一个必然越线的指标值。"""
    if operator == "<":
        return threshold - 1
    if operator == "<=":
        return threshold - 0.1
    if operator == ">":
        return threshold + 1
    if operator == ">=":
        return threshold + 0.1
    return threshold


def build_trap_cases(graph_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    """根据 graph.json 自动生成一批“触发禁忌”的测试案例。"""
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8-sig"))
    if not isinstance(graph_payload, list):
        raise ValueError("graph.json must be a list")

    cases: list[dict[str, Any]] = []
    index = 1
    for record in graph_payload:
        if not isinstance(record, dict):
            continue
        drug_name = str(record.get("drug_name", "")).strip()
        contraindications = record.get("contraindications") or []
        for item in contraindications:
            if not isinstance(item, dict):
                continue
            indicator = str(item.get("indicator", "")).strip()
            operator = str(item.get("operator", "")).strip()
            threshold_raw = item.get("threshold")
            try:
                threshold = float(threshold_raw)
            except (TypeError, ValueError):
                continue

            patient_value = _violate_threshold(operator, threshold)
            case = {
                "id": f"trap-{index:03d}",
                "description": f"{indicator} {operator} {threshold} 命中 {drug_name} 禁忌",
                "raw_input": {
                    "indicators": {indicator: patient_value},
                    "diseases": [],
                    "current_drugs": [drug_name],
                },
                "forbidden_drugs": [drug_name],
            }
            cases.append(case)
            index += 1
            if len(cases) >= limit:
                return cases
    return cases


def save_trap_cases(cases: list[dict[str, Any]], output_path: Path) -> None:
    """把陷阱样本写入 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def load_trap_cases(path: Path) -> list[dict[str, Any]]:
    """从磁盘读取已保存的陷阱样本。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("trap cases must be a list")
    return [item for item in payload if isinstance(item, dict)]
