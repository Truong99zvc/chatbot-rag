"""
Chunking strategy tuned for Vietnamese legal/regulatory documents
output from Docling (Markdown format).

Uses MarkdownTextSplitter as primary strategy so chunk boundaries
follow heading hierarchy (Chương → Điều → Khoản).
Falls back to RecursiveCharacterTextSplitter for non-Markdown content.
"""
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from app.config.settings import settings

# Separators ordered to prefer splitting at legal document boundaries
_LEGAL_SEPARATORS = [
    "\n## ",    # Chương / Mục heading level
    "\n### ",   # Điều heading level
    "\n#### ",  # Khoản / Điểm heading level
    "\n\n",     # Paragraph break
    "\n",       # Line break
    " ",        # Word boundary (last resort)
]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks.

    For Markdown content (Docling output): uses MarkdownTextSplitter
    which respects heading boundaries.

    For plain text: falls back to RecursiveCharacterTextSplitter with
    legal-document-aware separators.
    """
    if not documents:
        return []

    # Detect if content looks like Markdown (has headings)
    sample = documents[0].page_content[:500]
    is_markdown = sample.lstrip().startswith("#") or "\n#" in sample

    if is_markdown:
        splitter = MarkdownTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=_LEGAL_SEPARATORS,
        )

    return splitter.split_documents(documents)
