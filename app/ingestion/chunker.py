from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of Documents into smaller overlapping chunks.
    Source metadata (file name, page number) is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
