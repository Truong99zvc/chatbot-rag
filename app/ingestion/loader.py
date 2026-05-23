from pathlib import Path

from fastapi import UploadFile
from langchain_core.documents import Document

from app.ingestion.parser import parse_pdf, parse_txt

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


async def load_uploaded_file(file: UploadFile) -> list[Document]:
    """Read an uploaded file object and dispatch to the correct parser."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    content = await file.read()
    return parse_pdf(content, filename=file.filename) if ext == ".pdf" else parse_txt(content, filename=file.filename)


def load_file_from_path(path: str | Path) -> list[Document]:
    """Load a file from a local filesystem path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")
    content = path.read_bytes()
    return parse_pdf(content, filename=path.name) if ext == ".pdf" else parse_txt(content, filename=path.name)
