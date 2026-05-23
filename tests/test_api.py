import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body


@pytest.mark.asyncio
async def test_rag_query_empty_question():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/rag/query", json={"question": ""})
    # Pydantic validator should reject empty string
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_documents_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/documents/list")
    assert response.status_code == 200
    assert "documents" in response.json()


@pytest.mark.asyncio
async def test_documents_delete_not_implemented():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/v1/documents/somefile.pdf")
    assert response.status_code == 501
