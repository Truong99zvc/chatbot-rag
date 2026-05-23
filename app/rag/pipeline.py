from app.embeddings.embedder import get_embeddings
from app.vectorstore.vector_db import VectorDB
from app.rag.retriever import Retriever
from app.rag.generator import Generator
from app.rag.prompt_builder import build_rag_prompt, build_memory_context
from app.config.settings import settings

# In-process session store  (replace with Redis/DB for multi-worker deployments)
_session_histories: dict[str, list[dict]] = {}


class RAGPipeline:
    """
    Orchestrates the full Retrieve-Augment-Generate flow:
    1. Retrieve relevant chunks from FAISS
    2. Build context + memory block
    3. Generate answer with Gemini
    """

    def __init__(self) -> None:
        embeddings = get_embeddings()
        self._db = VectorDB(embeddings)
        self._retriever = Retriever(self._db.load())
        self._generator = Generator()
        self._prompt = build_rag_prompt()

    async def run(self, question: str, session_id: str = "default") -> dict:
        history = _session_histories.get(session_id, [])

        docs = self._retriever.retrieve(question)
        if not docs:
            return {
                "answer": "Không tìm thấy ngữ cảnh phù hợp trong tài liệu đã nạp.",
                "sources": "",
            }

        context = self._retriever.format_context(docs)
        sources = self._retriever.format_sources(docs)
        memory_context = build_memory_context(
            history, max_turns=settings.MAX_HISTORY_TURNS
        )

        answer = self._generator.generate(
            self._prompt,
            context=context,
            question=question,
            memory_context=memory_context,
        )

        _session_histories.setdefault(session_id, []).append(
            {"question": question, "answer": answer}
        )
        return {"answer": answer, "sources": sources}
