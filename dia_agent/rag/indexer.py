"""RAG 索引构建模块。

负责把 PDF 指南解析成文本切片，再写入 Chroma 向量库。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-not-found]
    from langchain_chroma import Chroma  # type: ignore[import-not-found]
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from dia_agent.rag.extractor import infer_source_label, pdf_to_markdown


CHINESE_GUIDELINE_NAME = "中国1型糖尿病诊治指南（2021版）.pdf"
CHINA_DIABETES_GUIDELINE_2024_NAME = "中国糖尿病防治指南（2024版）.pdf"
PREFACE_RE = re.compile(r"^前\s*言$")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章\s*.+$")
SECTION_RE = re.compile(r"^第\s*[0-9一二三四五六七八九十]+\s*节\s*.+$")
KEYPOINT_RE = re.compile(r"^要点提示[:：]$")
REFERENCE_RE = re.compile(r"^参考文献")
CN2024_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章\s*.+$")
CN2024_SECTION_RE = re.compile(r"^[一二三四五六七八九十]+、.+$")
CN2024_APPENDIX_RE = re.compile(r"^附录\s*[0-9一二三四五六七八九十]+\s*.+$")
PAGE_MARKER_RE = re.compile(r"^·\s*\d+\s*·$")


class RagIndexer:
    """把指南 PDF 构建成可检索的向量索引。"""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model: str,
        collection_name: str,
        embedding_device: str = "cpu",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        """初始化索引器，并准备好文本切片器。"""
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device.strip() or "cpu"
        self._collection_name = collection_name
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=420,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def build(self, pdf_paths: list[Path], reset: bool = False) -> int:
        """构建索引并返回最终写入的 chunk 数量。"""
        if reset and self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        docs = self.extract_documents(pdf_paths)
        embedding = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._embedding_device},
        )
        vector_store = Chroma(
            collection_name=self._collection_name,
            persist_directory=str(self._persist_dir),
            embedding_function=embedding,
        )
        if docs:
            vector_store.add_documents(docs)
        persist = getattr(vector_store, "persist", None)
        if callable(persist):
            persist()
        return len(docs)

    def extract_documents(self, pdf_paths: list[Path]) -> list[Document]:
        """公开 PDF 切片接口，便于 GraphRAG 入图脚本复用同一套 chunk 规则。"""
        return self._extract_documents(pdf_paths)

    def _extract_documents(self, pdf_paths: list[Path]) -> list[Document]:
        """把 PDF 转成切片后的 LangChain `Document` 列表。"""
        docs: list[Document] = []
        for pdf_path in pdf_paths:
            source = infer_source_label(pdf_path)
            if pdf_path.name == CHINESE_GUIDELINE_NAME:
                docs.extend(self._extract_chinese_guideline_documents(pdf_path, source))
                continue
            if pdf_path.name == CHINA_DIABETES_GUIDELINE_2024_NAME:
                docs.extend(self._extract_cn2024_guideline_documents(pdf_path, source))
                continue

            markdown = pdf_to_markdown(pdf_path)
            chunks = self._splitter.split_text(markdown)
            for index, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "file_name": pdf_path.name,
                            "chunk_index": index,
                        },
                    )
                )
        return docs

    def _extract_chinese_guideline_documents(self, pdf_path: Path, source: str) -> list[Document]:
        """按中文指南的章/节/要点结构切分文档。"""
        entries = self._load_chinese_guideline_entries(pdf_path)
        structured_entries = self._trim_chinese_guideline_entries(entries)

        docs: list[Document] = []
        chunk_index = 0
        current_chapter = ""
        current_section = ""
        current_lines: list[tuple[str, int]] = []
        keypoint_lines: list[tuple[str, int]] = []
        in_keypoints = False

        def flush_keypoints() -> None:
            nonlocal chunk_index, keypoint_lines
            if not keypoint_lines:
                return
            body = "\n".join(text for text, _ in keypoint_lines).strip()
            if not body:
                keypoint_lines = []
                return
            page_values = [page for _, page in keypoint_lines]
            prefix = self._build_chunk_prefix(
                source=source,
                chapter=current_chapter,
                section=current_section or "要点提示",
                chunk_type="要点提示",
            )
            docs.append(
                Document(
                    page_content=f"{prefix}\n{body}",
                    metadata={
                        "source": source,
                        "file_name": pdf_path.name,
                        "chunk_index": chunk_index,
                        "chapter": current_chapter,
                        "section": current_section or "要点提示",
                        "chunk_type": "summary",
                        "page_start": min(page_values),
                        "page_end": max(page_values),
                    },
                )
            )
            chunk_index += 1
            keypoint_lines = []

        def flush_section_lines() -> None:
            nonlocal chunk_index, current_lines
            if not current_lines:
                return
            paragraphs = self._merge_lines_to_paragraphs([text for text, _ in current_lines])
            page_values = [page for _, page in current_lines]
            for chunk in self._split_section_paragraphs(
                paragraphs,
                chapter=current_chapter,
                section=current_section or current_chapter,
                source=source,
            ):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "file_name": pdf_path.name,
                            "chunk_index": chunk_index,
                            "chapter": current_chapter,
                            "section": current_section or current_chapter,
                            "chunk_type": "detail",
                            "page_start": min(page_values),
                            "page_end": max(page_values),
                        },
                    )
                )
                chunk_index += 1
            current_lines = []

        for line, page_no in structured_entries:
            if CHAPTER_RE.match(line):
                if in_keypoints:
                    flush_keypoints()
                    in_keypoints = False
                flush_section_lines()
                current_chapter = line
                current_section = ""
                continue
            if SECTION_RE.match(line):
                if not current_chapter:
                    continue
                if in_keypoints:
                    flush_keypoints()
                    in_keypoints = False
                flush_section_lines()
                current_section = line
                continue
            if KEYPOINT_RE.match(line):
                if not current_chapter:
                    continue
                flush_section_lines()
                keypoint_lines = []
                in_keypoints = True
                continue
            if not current_chapter:
                # 前言和方法学描述不进入临床 RAG，避免污染问答召回结果。
                continue
            if in_keypoints:
                keypoint_lines.append((line, page_no))
                continue
            current_lines.append((line, page_no))

        if in_keypoints:
            flush_keypoints()
        flush_section_lines()
        return docs

    def _extract_cn2024_guideline_documents(self, pdf_path: Path, source: str) -> list[Document]:
        """按 2024 版糖尿病防治指南的章/条/附录结构切分文档。"""
        entries = self._load_chinese_guideline_entries(pdf_path)
        structured_entries = self._trim_cn2024_guideline_entries(entries)

        docs: list[Document] = []
        chunk_index = 0
        current_chapter = ""
        current_section = ""
        current_lines: list[tuple[str, int]] = []

        def flush_section_lines() -> None:
            nonlocal chunk_index, current_lines
            if not current_lines or not current_chapter:
                current_lines = []
                return

            paragraphs = self._merge_cn2024_lines_to_paragraphs([text for text, _ in current_lines])
            page_values = [page for _, page in current_lines]
            for chunk in self._split_section_paragraphs(
                paragraphs,
                chapter=current_chapter,
                section=current_section or current_chapter,
                source=source,
            ):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "file_name": pdf_path.name,
                            "chunk_index": chunk_index,
                            "chapter": current_chapter,
                            "section": current_section or current_chapter,
                            "chunk_type": "detail",
                            "page_start": min(page_values),
                            "page_end": max(page_values),
                        },
                    )
                )
                chunk_index += 1
            current_lines = []

        for line, page_no in structured_entries:
            if CN2024_CHAPTER_RE.match(line) or CN2024_APPENDIX_RE.match(line):
                flush_section_lines()
                current_chapter = line
                current_section = ""
                continue
            if CN2024_SECTION_RE.match(line):
                if not current_chapter:
                    continue
                flush_section_lines()
                current_section = line
                continue
            if not current_chapter:
                continue
            current_lines.append((line, page_no))

        flush_section_lines()
        return docs

    def _load_chinese_guideline_entries(self, pdf_path: Path) -> list[tuple[str, int]]:
        """读取中文指南的逐行文本，并保留页码。"""
        entries: list[tuple[str, int]] = []
        with fitz.open(str(pdf_path)) as document:
            for page_no, page in enumerate(document, start=1):
                for raw_line in page.get_text("text").splitlines():
                    line = self._normalize_guideline_line(raw_line)
                    if line:
                        entries.append((line, page_no))
        return entries

    def _trim_chinese_guideline_entries(self, entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """去掉目录和参考文献，只保留正文主体。"""
        start_index = self._find_guideline_body_start(entries)
        end_index = self._find_guideline_body_end(entries, start_index)
        return entries[start_index:end_index]

    def _trim_cn2024_guideline_entries(self, entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """截取 2024 版指南的正文与附录，排除前置目录和尾部参考文献。"""
        start_index = self._find_cn2024_body_start(entries)
        end_index = self._find_cn2024_body_end(entries, start_index)
        return entries[start_index:end_index]

    def _find_cn2024_body_start(self, entries: list[tuple[str, int]]) -> int:
        """定位 2024 版指南正文起点，避开跨行目录项。"""
        for index, (line, page_no) in enumerate(entries):
            if page_no <= 1:
                continue
            if not CN2024_CHAPTER_RE.match(line):
                continue
            if self._is_cn2024_body_candidate(entries, index):
                return index
        return 0

    def _is_cn2024_body_candidate(self, entries: list[tuple[str, int]], start_index: int) -> bool:
        """判断某个章节标题是否已经进入正文，而不是目录中的章目。"""
        lookahead = entries[start_index + 1 : start_index + 10]
        if len(lookahead) < 3:
            return False

        prose_count = 0
        dot_line_count = 0
        for line, _ in lookahead:
            compact = line.replace(" ", "")
            if "…" in line or compact.isdigit():
                dot_line_count += 1
                continue
            if CN2024_CHAPTER_RE.match(line) or CN2024_APPENDIX_RE.match(line):
                continue
            if CN2024_SECTION_RE.match(line):
                continue
            prose_count += 1

        # 正文里的章标题后很快会进入正文叙述；目录则会继续停留在条目+页码。
        return prose_count >= 2 and dot_line_count <= 2

    def _find_cn2024_body_end(self, entries: list[tuple[str, int]], start_index: int) -> int:
        """定位 2024 版指南正文结束位置，停在参考文献之前。"""
        for index in range(start_index, len(entries)):
            line, page_no = entries[index]
            compact = line.replace(" ", "")
            if page_no <= 2:
                continue
            if compact == "参考文献":
                return index
            if compact == "参" and self._is_split_reference_heading(entries, index):
                return index
        return len(entries)

    def _is_split_reference_heading(self, entries: list[tuple[str, int]], start_index: int) -> bool:
        """识别被 PDF 拆成单字多行的“参考文献”标题。"""
        target = ["参", "考", "文", "献"]
        if start_index + len(target) > len(entries):
            return False

        page_no = entries[start_index][1]
        for offset, token in enumerate(target):
            line, candidate_page = entries[start_index + offset]
            if candidate_page != page_no or line.replace(" ", "") != token:
                return False
        return True

    def _find_guideline_body_start(self, entries: list[tuple[str, int]]) -> int:
        """定位正文真正开始的位置，避免把目录页中的“前言”误判为正文。"""
        candidates = [index for index, (line, _) in enumerate(entries) if PREFACE_RE.match(line)]
        if not candidates:
            return 0

        for index in candidates:
            if self._is_preface_body_candidate(entries, index):
                return index

        # 兜底策略选最后一个“前言”，通常目录页会出现在更前面。
        return candidates[-1]

    def _is_preface_body_candidate(self, entries: list[tuple[str, int]], start_index: int) -> bool:
        """判断某个“前言”是否像正文起点而不是目录项。"""
        _, page_no = entries[start_index]
        if page_no <= 1:
            return False

        lookahead = entries[start_index + 1 : start_index + 16]
        if not lookahead:
            return False

        heading_count = 0
        prose_count = 0
        long_line_count = 0

        for line, _ in lookahead:
            if CHAPTER_RE.match(line) or SECTION_RE.match(line) or KEYPOINT_RE.match(line):
                heading_count += 1
                continue
            prose_count += 1
            if len(line) >= 20:
                long_line_count += 1

        # 正文前言后面应该快速进入成段叙述，而目录页会连续出现大量章/节标题。
        return prose_count >= 4 and long_line_count >= 3 and heading_count <= 2

    def _find_guideline_body_end(
        self,
        entries: list[tuple[str, int]],
        start_index: int,
    ) -> int:
        """定位正文结束位置，优先命中正文末尾的参考文献标记。"""
        start_page = entries[start_index][1] if entries else 0
        for index in range(len(entries) - 1, start_index, -1):
            line, page_no = entries[index]
            if not REFERENCE_RE.match(line):
                continue
            # 至少跨过目录所在前几页，避免再次命中目录中的“参考文献”。
            if page_no > start_page:
                return index
        return len(entries)

    def _normalize_guideline_line(self, raw_line: str) -> str:
        """清洗 PDF 抽取后的单行文本。"""
        line = raw_line.replace("\u3000", " ").replace("\xa0", " ").strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            return ""
        if PAGE_MARKER_RE.match(line):
            return ""
        # 页眉页脚和来源信息对临床问答帮助不大，直接过滤。
        useless_tokens = [
            "【规范与指南】",
            "文章来源：",
            "作者：",
            "通信作者：",
            "中华糖尿病杂志",
        ]
        if any(token in line for token in useless_tokens):
            return ""
        return line

    def _merge_lines_to_paragraphs(self, lines: list[str]) -> list[str]:
        """把 PDF 换行合并成更接近自然段的文本。"""
        paragraphs: list[str] = []
        buffer: list[str] = []
        for line in lines:
            if CHAPTER_RE.match(line) or SECTION_RE.match(line) or KEYPOINT_RE.match(line):
                if buffer:
                    paragraphs.append("".join(buffer).strip())
                    buffer = []
                continue
            if re.match(r"^[一二三四五六七八九十]+、", line):
                if buffer:
                    paragraphs.append("".join(buffer).strip())
                    buffer = []
                buffer.append(line)
                continue
            buffer.append(line)
            if line.endswith(("。", "！", "？")) and len("".join(buffer)) >= 120:
                paragraphs.append("".join(buffer).strip())
                buffer = []
        if buffer:
            paragraphs.append("".join(buffer).strip())
        return [item for item in paragraphs if item]

    def _merge_cn2024_lines_to_paragraphs(self, lines: list[str]) -> list[str]:
        """把 2024 版指南的多行正文、附录表格尽量合并成稳定段落。"""
        paragraphs: list[str] = []
        buffer: list[str] = []

        for line in lines:
            if CN2024_CHAPTER_RE.match(line) or CN2024_SECTION_RE.match(line) or CN2024_APPENDIX_RE.match(line):
                if buffer:
                    paragraphs.append("".join(buffer).strip())
                    buffer = []
                continue

            if re.match(r"^[0-9]+\.", line) or re.match(r"^（[0-9一二三四五六七八九十]+）", line):
                if buffer:
                    paragraphs.append("".join(buffer).strip())
                    buffer = []
                buffer.append(line)
                continue

            buffer.append(line)

            joined = "".join(buffer)
            if line.endswith(("。", "！", "？", "；")) and len(joined) >= 120:
                paragraphs.append(joined.strip())
                buffer = []
                continue

            # 附录中的药物表格往往是一行一个字段，凑到一定长度就切一段。
            if len(joined) >= 180 and not re.search(r"[。！？；]$", line):
                paragraphs.append(joined.strip())
                buffer = []

        if buffer:
            paragraphs.append("".join(buffer).strip())

        return [item for item in paragraphs if item]

    def _split_section_paragraphs(
        self,
        paragraphs: list[str],
        chapter: str,
        section: str,
        source: str,
    ) -> list[str]:
        """在节内进一步把正文拆成适合检索的小块。"""
        prefix = self._build_chunk_prefix(source=source, chapter=chapter, section=section, chunk_type="正文")
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n{paragraph}"
            if len(candidate) <= 420:
                current = candidate
                continue
            if current:
                chunks.append(f"{prefix}\n{current.strip()}")
            if len(paragraph) <= 520:
                current = paragraph
                continue
            split_parts = self._section_splitter.split_text(paragraph)
            for part in split_parts[:-1]:
                chunks.append(f"{prefix}\n{part.strip()}")
            current = split_parts[-1].strip() if split_parts else ""

        if current.strip():
            chunks.append(f"{prefix}\n{current.strip()}")
        return chunks

    def _build_chunk_prefix(self, source: str, chapter: str, section: str, chunk_type: str) -> str:
        """为每个 chunk 加上结构化标题前缀，增强语义表达。"""
        return "\n".join(
            [
                f"[来源] {source}",
                f"[章节] {chapter}",
                f"[小节] {section}",
                f"[类型] {chunk_type}",
            ]
        )
