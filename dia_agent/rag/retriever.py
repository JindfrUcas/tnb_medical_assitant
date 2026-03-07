"""Retriever for guideline snippets from ChromaDB."""

from __future__ import annotations

from pathlib import Path

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-not-found]
    from langchain_chroma import Chroma  # type: ignore[import-not-found]
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

from dia_agent.schemas import RagSnippet


class GuidelineRetriever:
    def __init__(self, persist_dir: Path, embedding_model: str, collection_name: str):
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._vector_store: Chroma | None = None

    def retrieve(self, query: str, top_k: int = 4, source: str | None = None) -> list[RagSnippet]:
        query = query.strip()
        if not query:
            return []
        if not self._persist_dir.exists():
            return []

        vector_store = self._get_vector_store()

        metadata_filter = {"source": source} if source else None
        docs = vector_store.similarity_search(query, k=top_k, filter=metadata_filter)
        snippets: list[RagSnippet] = []
        for doc in docs:
            snippets.append(
                RagSnippet(
                    source=str(doc.metadata.get("source", "UNKNOWN")),
                    content=doc.page_content,
                    score=None,
                )
            )
        return snippets

    def _get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            embedding = HuggingFaceEmbeddings(model_name=self._embedding_model)
            self._vector_store = Chroma(
                collection_name=self._collection_name,
                persist_directory=str(self._persist_dir),
                embedding_function=embedding,
            )
        return self._vector_store
