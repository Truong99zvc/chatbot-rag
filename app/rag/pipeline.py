"""
RAG Pipeline for the UIT Quy Chế Đào Tạo Chatbot.

Orchestrates the full Retrieve-Augment-Generate flow using LangGraph Agent:
  1. Load session history from SQL database (SQLAlchemy)
  2. Invoke LangGraph Agent (Routing -> Rewriting -> Hybrid Retrieve -> Grade -> Generate -> Hallucination Check)
  3. Persist updated session history in the database.
"""
from __future__ import annotations

import logging
from typing import List

from app.database.connection import SessionLocal
from app.database.models import ChatSession, ChatTurn
from app.embeddings.embedder import get_embeddings
from app.rag.agent import UITAcademicAgent
from app.rag.generator import Generator
from app.rag.retriever import Retriever
from app.vectorstore.vector_db import VectorDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL Database Session Store Helper Functions
# ---------------------------------------------------------------------------

def get_session_history(session_id: str) -> List[dict]:
    """Return the conversation history for *session_id* from database."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            return []
        # Sort turns by id (insertion order)
        turns = sorted(session.turns, key=lambda t: t.id)
        return [{"question": t.question, "answer": t.answer} for t in turns]
    except Exception as e:
        logger.error("Failed to query session history: %s", e)
        return []
    finally:
        db.close()


def append_to_session(session_id: str, question: str, answer: str, sources: str = "") -> None:
    """Append a Q&A turn to the session history and persist to database."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            session = ChatSession(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        turn = ChatTurn(
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources
        )
        db.add(turn)
        db.commit()
    except Exception as e:
        logger.error("Failed to save chat turn: %s", e)
        db.rollback()
    finally:
        db.close()


def clear_session_history(session_id: str) -> None:
    """Clear conversation history for *session_id* from database."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            logger.info("Session %s history cleared.", session_id)
    except Exception as e:
        logger.error("Failed to clear session %s history: %s", session_id, e)
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the full Agentic RAG flow for UIT training regulations Q&A.
    Uses LangGraph for decision routing and self-correction.
    """

    def __init__(self) -> None:
        embeddings = get_embeddings()
        db = VectorDB(embeddings)

        try:
            vector_store = db.load()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FAISS index not found. "
                "Please run `make build-index` to build the knowledge base first."
            ) from exc

        self._retriever = Retriever(vector_store)
        self._generator = Generator()
        # Initialize LangGraph Agent
        self._agent = UITAcademicAgent(self._retriever, self._generator)

    async def run(self, question: str, session_id: str = "default") -> dict:
        """
        Execute the LangGraph Agentic RAG pipeline for a given question.

        Returns:
            dict with keys: answer, sources, session_id
        """
        history = get_session_history(session_id)

        # Run the LangGraph Agent
        result = self._agent.run(question, history=history)

        answer = result["answer"]
        sources = result["sources"]

        # Persist Q&A turn to SQL DB
        append_to_session(session_id, question, answer, sources)

        return {"answer": answer, "sources": sources, "session_id": session_id}

    def search_article(self, article_number: str | int) -> dict:
        """
        Find and return content related to a specific Điều (article) number.

        Returns:
            dict with keys: article, content, sources
        """
        docs = self._retriever.search_by_article(article_number)
        content = self._retriever.format_article_results(docs, article_number)
        sources = self._retriever.format_sources(docs) if docs else ""
        return {"article": str(article_number), "content": content, "sources": sources}
