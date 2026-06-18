"""
RAG Pipeline for the UIT Quy Chế Đào Tạo Chatbot.

Orchestrates the full Retrieve-Augment-Generate flow:
  1. Load session history from JSON file store
  2. Retrieve relevant chunks from pre-built FAISS index
  3. Build context + memory block
  4. Generate answer with Gemini
  5. Persist updated session history

Session history is stored in a JSON file (SESSION_STORE_FILE) keyed by session_id.
This is simple and sufficient for single-server deployment.
"""
from __future__ import annotations

import json
import logging

from app.config.settings import settings
from app.embeddings.embedder import get_embeddings
from app.rag.generator import Generator
from app.rag.prompt_builder import build_memory_context, build_rag_prompt
from app.rag.retriever import Retriever
from app.vectorstore.vector_db import VectorDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-backed session store
# ---------------------------------------------------------------------------

def _load_all_sessions() -> dict[str, list[dict]]:
    """Load all session histories from the JSON store file."""
    store_file = settings.SESSION_STORE_FILE
    if not store_file.exists():
        return {}
    try:
        data = json.loads(store_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Session store file corrupted, starting fresh.")
        return {}


def _save_all_sessions(sessions: dict[str, list[dict]]) -> None:
    """Persist all session histories to the JSON store file."""
    store_file = settings.SESSION_STORE_FILE
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_session_history(session_id: str) -> list[dict]:
    """Return the conversation history for *session_id*."""
    return _load_all_sessions().get(session_id, [])


def append_to_session(session_id: str, question: str, answer: str) -> None:
    """Append a Q&A turn to the session history and persist."""
    sessions = _load_all_sessions()
    sessions.setdefault(session_id, []).append(
        {"question": question, "answer": answer}
    )
    _save_all_sessions(sessions)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the full RAG flow for UIT training regulations Q&A.

    The FAISS index must be pre-built via `scripts/build_index.py`.
    If the index does not exist, initialization raises a clear error.
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
        self._prompt = build_rag_prompt()

    async def run(self, question: str, session_id: str = "default") -> dict:
        """
        Execute the RAG pipeline for a given question.

        Returns:
            dict with keys: answer, sources, session_id
        """
        history = get_session_history(session_id)

        docs = self._retriever.retrieve(question)
        if not docs:
            no_context_answer = (
                "Tôi không tìm thấy thông tin liên quan trong Quy chế Đào tạo UIT. "
                "Bạn vui lòng liên hệ Phòng Đào tạo – phòng A101 hoặc "
                "email daotao@uit.edu.vn để được hỗ trợ."
            )
            append_to_session(session_id, question, no_context_answer)
            return {"answer": no_context_answer, "sources": ""}

        context = self._retriever.format_context(docs)
        sources = self._retriever.format_sources(docs)
        memory_context = build_memory_context(history, max_turns=settings.MAX_HISTORY_TURNS)

        answer = self._generator.generate(
            self._prompt,
            context=context,
            question=question,
            memory_context=memory_context,
        )

        append_to_session(session_id, question, answer)
        return {"answer": answer, "sources": sources}

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
