import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config.settings import settings


class VectorDB:
    """FAISS-backed vector store with load / save / merge / reset helpers."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._index_dir = Path(settings.FAISS_INDEX_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed *documents* and merge them into the on-disk FAISS index.
        If no index exists yet, one is created from scratch.
        """
        if not documents:
            raise ValueError("No documents to index.")
        new_store = FAISS.from_documents(documents, embedding=self._embeddings)
        if self._index_dir.exists():
            existing = self.load()
            existing.merge_from(new_store)
            self._save(existing)
        else:
            self._save(new_store)

    def load(self) -> FAISS:
        """Load the FAISS index from disk."""
        if not self._index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self._index_dir}'. "
                "Upload and process documents first."
            )
        return FAISS.load_local(
            str(self._index_dir),
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def reset(self) -> None:
        """Delete the entire FAISS index from disk."""
        if self._index_dir.exists():
            shutil.rmtree(self._index_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, store: FAISS) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self._index_dir))
