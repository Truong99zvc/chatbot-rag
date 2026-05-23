from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import settings


class Retriever:
    """Wraps a FAISS vector store for similarity-based document retrieval."""

    def __init__(self, vector_store: FAISS) -> None:
        self._store = vector_store

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Return the top-k most relevant document chunks for *query*."""
        k = k or settings.TOP_K_RESULTS
        return self._store.similarity_search(query, k=k)

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        """Concatenate document page contents into a single context block."""
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def format_sources(docs: list[Document]) -> str:
        """Return a deduplicated markdown list of (source file, page) pairs."""
        seen: set[tuple] = set()
        lines: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "Không rõ")
            page = doc.metadata.get("page", "?")
            key = (source, page)
            if key not in seen:
                seen.add(key)
                lines.append(f"- **{source}**, trang {page}")
        return "\n".join(lines)
