from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, field_validator
from typing import Optional

from app.rag.pipeline import RAGPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: str
    session_id: str


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a question to the RAG pipeline.  
    Returns the answer and the source documents (file + page) used.
    """
    pipeline = RAGPipeline()
    result = await pipeline.run(request.question, session_id=request.session_id)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )
