"""
RAG API endpoints for the UIT Quy Chế Đào Tạo Chatbot.

Endpoints:
  POST /api/v1/rag/query          — Free-form Q&A
  GET  /api/v1/rag/search         — Look up a specific Điều (article)
  GET  /api/v1/rag/sessions/{id}  — Retrieve conversation history
  DELETE /api/v1/rag/sessions/{id} — Clear a session's history
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, field_validator

from app.config.settings import settings
from app.rag.pipeline import RAGPipeline, get_session_history, _save_all_sessions, _load_all_sessions

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Câu hỏi không được để trống.")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: str
    session_id: str


class ArticleSearchResponse(BaseModel):
    article: str
    content: str
    sources: str


class SessionTurn(BaseModel):
    question: str
    answer: str


class SessionResponse(BaseModel):
    session_id: str
    turns: list[SessionTurn]
    total_turns: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Hỏi đáp về Quy chế Đào tạo UIT",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Đặt câu hỏi về quy chế, quy định đào tạo đại học chính quy của UIT.

    **Ví dụ câu hỏi:**
    - Điều kiện để được xét tốt nghiệp là gì?
    - Sinh viên được phép nghỉ học tối đa bao nhiêu buổi?
    - Thang điểm chữ của UIT được quy định như thế nào?
    """
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    result = await pipeline.run(request.question, session_id=request.session_id)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )


@router.get(
    "/search",
    response_model=ArticleSearchResponse,
    summary="Tìm nội dung theo số Điều",
)
async def search_by_article(
    article: str = Query(..., description="Số điều cần tìm, ví dụ: 15 hoặc '15'"),
) -> ArticleSearchResponse:
    """
    Tra cứu nội dung của một Điều cụ thể trong Quy chế Đào tạo UIT.

    **Ví dụ:** `/api/v1/rag/search?article=15` → trả về nội dung Điều 15.
    """
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    result = pipeline.search_article(article)
    return ArticleSearchResponse(**result)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Lấy lịch sử hội thoại của một phiên",
)
async def get_session(session_id: str) -> SessionResponse:
    """Trả về toàn bộ lịch sử câu hỏi–trả lời của *session_id*."""
    history = get_session_history(session_id)
    return SessionResponse(
        session_id=session_id,
        turns=[SessionTurn(**turn) for turn in history],
        total_turns=len(history),
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Xóa lịch sử hội thoại của một phiên",
)
async def clear_session(session_id: str) -> None:
    """Xóa toàn bộ lịch sử hội thoại của *session_id*."""
    sessions = _load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
        _save_all_sessions(sessions)
