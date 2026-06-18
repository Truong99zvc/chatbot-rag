"""
Tests for RAG components of the UIT Quy Chế Đào Tạo Chatbot.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.rag.retriever import Retriever
from app.rag.prompt_builder import build_memory_context, build_rag_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_docs(texts: list[str] | None = None) -> list[Document]:
    """Helper: create sample Documents with UIT-style content."""
    if texts is None:
        texts = [
            "Điều 15. Điều kiện xét tốt nghiệp\n1. Sinh viên phải tích lũy đủ số tín chỉ.",
            "Điều 16. Thang điểm\nThang điểm chữ A tương đương 4.0.",
        ]
    return [
        Document(page_content=t, metadata={"source": "quy_che_uit.pdf", "page": i + 1})
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_format_context_contains_content(self):
        docs = make_docs()
        context = Retriever.format_context(docs)
        assert "Điều 15" in context
        assert "Điều 16" in context

    def test_format_context_uses_separator(self):
        docs = make_docs()
        context = Retriever.format_context(docs)
        assert "---" in context  # separator between chunks

    def test_format_context_empty(self):
        assert Retriever.format_context([]) == ""

    def test_format_sources_deduplication(self):
        docs = [
            Document(page_content="x", metadata={"source": "quy_che.pdf", "page": 1}),
            Document(page_content="y", metadata={"source": "quy_che.pdf", "page": 1}),  # dup
            Document(page_content="z", metadata={"source": "quy_che.pdf", "page": 2}),
        ]
        sources = Retriever.format_sources(docs)
        assert sources.count("quy_che.pdf, trang 1") == 1 or sources.count("trang 1") == 1
        assert "trang 2" in sources

    def test_format_article_results_found(self):
        docs = make_docs(["Điều 15. Điều kiện tốt nghiệp\nCần 130 tín chỉ."])
        result = Retriever.format_article_results(docs, 15)
        assert "Điều 15" in result
        assert "130 tín chỉ" in result

    def test_format_article_results_not_found(self):
        result = Retriever.format_article_results([], 99)
        assert "Điều 99" in result
        assert "Không tìm thấy" in result

    def test_search_by_article_filters_by_number(self):
        """search_by_article should return docs containing the article number."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Điều 15. Nội dung điều 15", metadata={"source": "x.pdf", "page": 1}),
            Document(page_content="Điều 20. Nội dung khác", metadata={"source": "x.pdf", "page": 2}),
        ]
        retriever = Retriever(mock_store)
        results = retriever.search_by_article(15)
        # Should prefer the doc that mentions Điều 15
        assert any("Điều 15" in doc.page_content for doc in results)


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_build_memory_context_empty(self):
        ctx = build_memory_context([])
        assert "Không có" in ctx

    def test_build_memory_context_with_history(self):
        history = [{"question": "Điều kiện tốt nghiệp?", "answer": "Cần 130 tín chỉ."}]
        ctx = build_memory_context(history)
        assert "Điều kiện tốt nghiệp?" in ctx
        assert "Cần 130 tín chỉ." in ctx

    def test_build_memory_context_uses_uit_labels(self):
        history = [{"question": "Q", "answer": "A"}]
        ctx = build_memory_context(history)
        # Should use "Sinh viên" / "Tư vấn" labels, not generic ones
        assert "Sinh viên" in ctx or "Lượt" in ctx

    def test_build_memory_context_max_turns(self):
        history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        ctx = build_memory_context(history, max_turns=3)
        assert "Q9" in ctx
        assert "Q0" not in ctx  # old turns truncated

    def test_build_rag_prompt_has_required_variables(self):
        prompt = build_rag_prompt()
        # Template must contain all three required input variables
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables
        assert "memory_context" in prompt.input_variables

    def test_build_rag_prompt_contains_uit_context(self):
        prompt = build_rag_prompt()
        template_str = str(prompt)
        assert "UIT" in template_str
        assert "Điều" in template_str or "quy chế" in template_str.lower()
