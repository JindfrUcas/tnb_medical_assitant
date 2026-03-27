"""RAG 索引构建模块。

负责把 PDF 指南解析成文本切片，再写入 Chroma 向量库。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-not-found]
    from langchain_chroma import Chroma  # type: ignore[import-not-found]
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from dia_agent.rag.extractor import infer_source_label, pdf_to_markdown


class RagIndexer:
    """把指南 PDF 构建成可检索的向量索引。"""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model: str,
        collection_name: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        """初始化索引器，并准备好文本切片器。"""
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build(self, pdf_paths: list[Path], reset: bool = False) -> int:
        """构建索引并返回最终写入的 chunk 数量。"""
        if reset and self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        docs = self._extract_documents(pdf_paths)
        embedding = HuggingFaceEmbeddings(model_name=self._embedding_model)
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

    def _extract_documents(self, pdf_paths: list[Path]) -> list[Document]:
        """把 PDF 转成切片后的 LangChain `Document` 列表。"""
        docs: list[Document] = []
        for pdf_path in pdf_paths:
            markdown = pdf_to_markdown(pdf_path)
            source = infer_source_label(pdf_path)
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
