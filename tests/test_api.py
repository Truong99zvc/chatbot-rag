"""
API integration tests for the UIT Quy Chế Đào Tạo Chatbot.

Tests run against the FastAPI app using httpx AsyncClient.
RAGPipeline is mocked to avoid needing the FAISS index during CI.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient

from app.main import app


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body
    assert "index_ready" in body  # new field


# ---------------------------------------------------------------------------
# RAG query endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_empty_question():
    """Pydantic validator should reject empty questions with 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/rag/query", json={"question": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_whitespace_only():
    """Whitespace-only question should also be rejected."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/rag/query", json={"question": "   "})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_no_index_returns_503():
    """When FAISS index is missing, should return 503."""
    with patch("app.api.rag.RAGPipeline", side_effect=RuntimeError("Index not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"question": "Điều kiện tốt nghiệp?"},
            )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_rag_query_success():
    """Mock a successful RAG query."""
    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value={
        "answer": "Sinh viên cần tích lũy đủ 130 tín chỉ.",
        "sources": "- **quy_che_uit.pdf**, trang 15",
    })
    with patch("app.api.rag.RAGPipeline", return_value=mock_pipeline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"question": "Điều kiện tốt nghiệp?", "session_id": "test-session"},
            )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert body["session_id"] == "test-session"


# ---------------------------------------------------------------------------
# Article search endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_article_success():
    """Mock a successful article search."""
    mock_pipeline = MagicMock()
    mock_pipeline.search_article.return_value = {
        "article": "15",
        "content": "## Nội dung Điều 15\n...",
        "sources": "- **quy_che_uit.pdf**, trang 15",
    }
    with patch("app.api.rag.RAGPipeline", return_value=mock_pipeline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/rag/search?article=15")
    assert response.status_code == 200
    body = response.json()
    assert body["article"] == "15"
    assert "content" in body


# ---------------------------------------------------------------------------
# Session history endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_empty():
    """Getting a session that doesn't exist should return empty turns."""
    with patch("app.api.rag.get_session_history", return_value=[]):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/rag/sessions/nonexistent-session")
    assert response.status_code == 200
    body = response.json()
    assert body["total_turns"] == 0
    assert body["turns"] == []


@pytest.mark.asyncio
async def test_clear_session():
    """Clearing a session should return 204 No Content."""
    with patch("app.api.rag._load_all_sessions", return_value={"test-sess": []}), \
         patch("app.api.rag._save_all_sessions"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/v1/rag/sessions/test-sess")
    assert response.status_code == 204
