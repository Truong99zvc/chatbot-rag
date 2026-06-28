"""
Retriever for the UIT Quy Chế Đào Tạo Chatbot.

Provides:
  - Semantic similarity search (main Q&A flow)
  - Article-based search: find chunks mentioning a specific Điều number
"""
from __future__ import annotations

import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import settings


class Retriever:
    """Wraps a FAISS vector store for similarity-based document retrieval."""

    def __init__(self, vector_store: FAISS) -> None:
        self._store = vector_store

        # Initialize BM25 retriever from FAISS docstore for Hybrid Search
        self._bm25 = None
        try:
            docstore = getattr(vector_store, "docstore", None)
            if docstore and hasattr(docstore, "_dict"):
                all_docs = list(docstore._dict.values())
                if all_docs:
                    from langchain_community.retrievers import BM25Retriever
                    self._bm25 = BM25Retriever.from_documents(all_docs)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to initialize BM25Retriever for Hybrid Search: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Return the top-k most relevant chunks using Hybrid Search (FAISS + BM25)."""
        k = k or settings.TOP_K_RESULTS

        # 1. Semantic search via FAISS
        vector_results = self._store.similarity_search(query, k=k * 2)

        # 2. Keyword search via BM25
        bm25_results = []
        if self._bm25:
            try:
                self._bm25.k = k * 2
                bm25_results = self._bm25.invoke(query)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("BM25 retrieval failed: %s", e)

        # 3. Interleave and deduplicate results
        seen: set[tuple] = set()
        combined: list[Document] = []

        for i in range(max(len(vector_results), len(bm25_results))):
            if i < len(vector_results):
                doc = vector_results[i]
                # Identify doc uniquely by content & page
                doc_id = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
                if doc_id not in seen:
                    seen.add(doc_id)
                    combined.append(doc)
            if i < len(bm25_results):
                doc = bm25_results[i]
                doc_id = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
                if doc_id not in seen:
                    seen.add(doc_id)
                    combined.append(doc)

        return combined[:k]

    def search_by_article(self, article_number: str | int, k: int = 6) -> list[Document]:
        """
        Find chunks that contain a specific Điều (article) number.

        Uses semantic search with an article-specific query, then
        post-filters results to those containing the article reference.

        Args:
            article_number: The article number to search for (e.g. 15 → "Điều 15")
            k: Max number of results to return.

        Returns:
            List of Documents referencing the given article.
        """
        query = f"Điều {article_number}"
        # Retrieve more candidates, then filter for exact article reference
        candidates = self._store.similarity_search(query, k=k * 3)
        pattern = re.compile(
            rf"\bĐiều\s+{re.escape(str(article_number))}\b",
            re.IGNORECASE,
        )
        matched = [doc for doc in candidates if pattern.search(doc.page_content)]
        return matched[:k] if matched else candidates[:k]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        """Concatenate document page contents into a single context block."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def format_sources(docs: list[Document]) -> str:
        """Return a deduplicated markdown list of (source file, page) pairs."""
        seen: set[tuple] = set()
        lines: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "Quy chế UIT")
            page = doc.metadata.get("page", "?")
            key = (source, page)
            if key not in seen:
                seen.add(key)
                lines.append(f"- **{source}**, trang {page}")
        return "\n".join(lines)

    @staticmethod
    def format_article_results(docs: list[Document], article_number: str | int) -> str:
        """Format article search results into a readable response."""
        if not docs:
            return f"Không tìm thấy nội dung về Điều {article_number} trong Quy chế Đào tạo UIT."
        parts = [f"## Nội dung Điều {article_number} trong Quy chế Đào tạo UIT\n"]
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "?")
            parts.append(f"**[Trang {page}]**\n{doc.page_content}")
            if i < len(docs):
                parts.append("---")
        return "\n\n".join(parts)
