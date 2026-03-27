"""指南 PDF 抽取辅助函数。"""

from __future__ import annotations

from pathlib import Path

import fitz


def pdf_to_markdown(pdf_path: Path) -> str:
    """优先用 `pymupdf4llm` 抽取 PDF；失败时退回基础文本解析。"""
    try:
        import pymupdf4llm  # type: ignore

        return str(pymupdf4llm.to_markdown(str(pdf_path))).strip()
    except Exception:
        with fitz.open(str(pdf_path)) as document:
            pages: list[str] = []
            for index, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if not page_text:
                    continue
                pages.append(f"## Page {index}\n\n{page_text}")
            return "\n\n".join(pages).strip()


def infer_source_label(pdf_path: Path) -> str:
    """根据文件名推断指南来源标签，便于后续筛选。"""
    file_name = pdf_path.name.lower()
    if "2021" in file_name or "中国" in file_name:
        return "CHN"
    if "dc26" in file_name or "ada" in file_name:
        return "ADA"
    return "UNKNOWN"
