import pytest
from langchain_core.documents import Document

from app.rag.retriever import Retriever
from app.rag.prompt_builder import build_memory_context, build_rag_prompt


def make_docs(n: int = 2) -> list[Document]:
    return [
        Document(
            page_content=f"Sample content {i}",
            metadata={"source": "test.pdf", "page": i},
        )
        for i in range(1, n + 1)
    ]


class TestRetriever:
    def test_format_context(self):
        docs = make_docs(2)
        context = Retriever.format_context(docs)
        assert "Sample content 1" in context
        assert "Sample content 2" in context

    def test_format_sources_deduplication(self):
        docs = [
            Document(page_content="x", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="y", metadata={"source": "a.pdf", "page": 1}),  # duplicate
            Document(page_content="z", metadata={"source": "b.pdf", "page": 2}),
        ]
        sources = Retriever.format_sources(docs)
        assert sources.count("a.pdf") == 1
        assert "b.pdf" in sources

    def test_format_context_empty(self):
        assert Retriever.format_context([]) == ""


class TestPromptBuilder:
    def test_build_memory_context_empty(self):
        ctx = build_memory_context([])
        assert "Không có" in ctx

    def test_build_memory_context_with_history(self):
        history = [{"question": "Xin chào", "answer": "Chào bạn"}]
        ctx = build_memory_context(history)
        assert "Xin chào" in ctx
        assert "Chào bạn" in ctx

    def test_build_memory_context_max_turns(self):
        history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        ctx = build_memory_context(history, max_turns=3)
        assert "Q9" in ctx
        assert "Q0" not in ctx

    def test_build_rag_prompt(self):
        prompt = build_rag_prompt()
        assert prompt is not None
