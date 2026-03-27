#!/usr/bin/env python3
"""根据图谱红线规则构建 SFT 训练数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.sft.generator import generate_sft_dataset


def parse_args() -> argparse.Namespace:
    """解析 SFT 数据集构建参数。"""
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Dia-Agent")
    parser.add_argument("--graph", type=Path, default=Path("dataset/graph.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/sft_guardrail_2k.jsonl"))
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """执行 SFT 数据集生成脚本。"""
    args = parse_args()
    if not args.graph.exists():
        raise FileNotFoundError(f"graph file not found: {args.graph}")
    rows = generate_sft_dataset(
        graph_json_path=args.graph,
        output_path=args.output,
        total_samples=args.size,
        seed=args.seed,
    )
    print(f"SFT dataset generated: {args.output} (rows={rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
