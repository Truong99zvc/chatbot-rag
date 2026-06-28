import logging
import shutil
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config.settings import settings

logger = logging.getLogger(__name__)


class VectorDB:
    """Qdrant-backed vector store with load / save / reset helpers."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._collection_name = settings.QDRANT_COLLECTION
        self._local_path = settings.QDRANT_PATH
        self._url = settings.QDRANT_URL

    def _get_client(self) -> QdrantClient:
        """Create and return a QdrantClient based on current settings."""
        if self._url:
            logger.info("Connecting to remote Qdrant server at: %s", self._url)
            return QdrantClient(url=self._url)
        else:
            # Use local persistent storage on disk
            logger.info("Using local persistent Qdrant storage at: %s", self._local_path)
            self._local_path.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(self._local_path))

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed *documents* and insert them into the Qdrant collection.
        If the collection does not exist, it will be automatically created.
        """
        if not documents:
            raise ValueError("No documents to index.")

        client = self._get_client()

        # Check if collection exists, if not create it
        try:
            if not client.collection_exists(self._collection_name):
                # Dynamically retrieve embedding dimension
                dummy_vector = self._embeddings.embed_query("dummy")
                dimension = len(dummy_vector)
                logger.info("Creating Qdrant collection '%s' with dimension %d...", self._collection_name, dimension)
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=dimension,
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.warning("Error checking or creating Qdrant collection: %s. Proceeding with insertion...", e)

        store = QdrantVectorStore(
            client=client,
            collection_name=self._collection_name,
            embedding=self._embeddings,
        )
        store.add_documents(documents)
        logger.info("Successfully added %d documents to Qdrant collection '%s'.", len(documents), self._collection_name)

    def load(self) -> QdrantVectorStore:
        """Load the Qdrant vector store wrapper."""
        client = self._get_client()

        # Verify collection exists before loading
        if not self._url and not self._local_path.exists():
            raise FileNotFoundError(
                f"Qdrant local storage path '{self._local_path}' does not exist. "
                "Please run `make build-index` first."
            )

        if not client.collection_exists(self._collection_name):
            raise FileNotFoundError(
                f"Qdrant collection '{self._collection_name}' not found. "
                "Please build the index first."
            )

        return QdrantVectorStore(
            client=client,
            collection_name=self._collection_name,
            embedding=self._embeddings,
        )

    def reset(self) -> None:
        """Delete the collection from Qdrant server or clean the local folder."""
        client = self._get_client()
        try:
            if client.collection_exists(self._collection_name):
                logger.info("Deleting Qdrant collection '%s'...", self._collection_name)
                client.delete_collection(self._collection_name)
        except Exception as e:
            logger.warning("Failed to delete Qdrant collection: %s", e)

        # Clean local folder if using disk
        if not self._url and self._local_path.exists():
            logger.info("Deleting local Qdrant directory: %s", self._local_path)
            # Close client connection to release database lock on Windows
            client.close()
            try:
                shutil.rmtree(self._local_path)
            except Exception as e:
                logger.error("Failed to delete local Qdrant directory: %s", e)
