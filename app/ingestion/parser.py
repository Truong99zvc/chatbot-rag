import io

from langchain_core.documents import Document
from pypdf import PdfReader


def parse_pdf(content: bytes, filename: str = "unknown.pdf") -> list[Document]:
    """
    Extract text from PDF bytes.
    Returns one Document per page (pages with no extractable text are skipped).
    """
    reader = PdfReader(io.BytesIO(content))
    documents: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num},
                )
            )
    return documents


def parse_txt(content: bytes, filename: str = "unknown.txt") -> list[Document]:
    """Parse plain-text bytes into a single Document."""
    text = content.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": filename, "page": 1})]
