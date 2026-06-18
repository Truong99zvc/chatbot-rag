"""
File loader for the UIT Quy Chế Đào Tạo Chatbot.

Since the knowledge base is pre-indexed (PDF is fixed), this module
only provides loading from local filesystem paths, not upload handling.

Docling is used for PDF parsing (supports OCR for scanned pages).
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.parser import parse_pdf_with_docling

SUPPORTED_EXTENSIONS = {".pdf"}


def load_file_from_path(path: str | Path) -> list[Document]:
    """
    Load and parse a PDF file from a local filesystem path.

    Args:
        path: Absolute or relative path to a PDF file.

    Returns:
        List of LangChain Documents, one per non-empty page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return parse_pdf_with_docling(path)
