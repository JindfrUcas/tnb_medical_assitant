"""可选的 Ragas 评测集成。"""

from __future__ import annotations

from typing import Any


def run_ragas(records: list[dict[str, Any]]) -> dict[str, float]:
    """把问答记录转换成 Ragas 数据集并计算指标均值。"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [str(item.get("question", "")) for item in records],
            "answer": [str(item.get("answer", "")) for item in records],
            "contexts": [item.get("contexts", []) for item in records],
            "ground_truth": [str(item.get("ground_truth", "")) for item in records],
        }
    )
    result = evaluate(dataset, metrics=[answer_relevancy, context_precision, faithfulness])
    payload = result.to_pandas().mean(numeric_only=True).to_dict()
    return {str(key): float(value) for key, value in payload.items()}
