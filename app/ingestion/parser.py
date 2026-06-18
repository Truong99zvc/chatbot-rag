"""
Document parsers for the UIT Quy Chế Đào Tạo Chatbot.

PDF parsing uses Docling which handles:
  - Native text extraction for selectable text pages
  - OCR for scanned image pages
  - Structure-aware output (headings, tables, lists) → clean Markdown
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def parse_pdf_with_docling(pdf_path: str | Path) -> list[Document]:
    """
    Parse a PDF file using Docling.

    Docling handles mixed PDFs (text + scanned images) and outputs
    structured Markdown that preserves document hierarchy
    (Chương → Điều → Khoản → Điểm).

    Returns one Document per page, with metadata:
        - source: file name
        - page:   1-indexed page number
    """
    from docling.document_converter import DocumentConverter

    pdf_path = Path(pdf_path)
    logger.info("Parsing PDF with Docling: %s", pdf_path.name)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    documents: list[Document] = []

    # Docling exposes per-page content via result.document.pages
    # We export each page as Markdown to preserve structure.
    doc = result.document
    total_pages = len(doc.pages) if doc.pages else 0

    if total_pages == 0:
        # Fallback: export the whole document as one chunk
        full_md = doc.export_to_markdown()
        if full_md.strip():
            documents.append(
                Document(
                    page_content=full_md,
                    metadata={"source": pdf_path.name, "page": 1},
                )
            )
        return documents

    for page_no, page in enumerate(doc.pages.values(), start=1):
        # Export just this page's content
        page_md = page.export_to_markdown() if hasattr(page, "export_to_markdown") else ""

        # Fallback: filter exported markdown by page reference
        if not page_md.strip():
            page_md = _extract_page_text_fallback(doc, page_no)

        if page_md.strip():
            documents.append(
                Document(
                    page_content=page_md,
                    metadata={"source": pdf_path.name, "page": page_no},
                )
            )

    logger.info(
        "Docling parsed %d pages → %d non-empty pages from '%s'",
        total_pages,
        len(documents),
        pdf_path.name,
    )
    return documents


def _extract_page_text_fallback(doc, page_no: int) -> str:
    """Collect text elements belonging to a specific page number."""
    lines: list[str] = []
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if prov and any(p.page_no == page_no for p in prov):
            text = getattr(item, "text", "") or ""
            if text.strip():
                lines.append(text)
    return "\n".join(lines)


def parse_pdf_whole_doc(pdf_path: str | Path) -> list[Document]:
    """
    Alternative: export the whole document as a single Markdown string,
    then return as one Document. Used when per-page splitting is not needed.
    """
    from docling.document_converter import DocumentConverter

    pdf_path = Path(pdf_path)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    if not markdown.strip():
        return []

    return [Document(page_content=markdown, metadata={"source": pdf_path.name, "page": 0})]
