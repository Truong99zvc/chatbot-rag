"""
scripts/build_index.py
======================
One-time script to build the FAISS vector index from UIT training
regulations PDF files in the `data/` directory.

Usage:
    python scripts/build_index.py           # index all PDFs in data/
    python scripts/build_index.py --reset   # delete existing index first

This script must be run before starting the API server.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `app.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.config.settings import settings
from app.embeddings.embedder import get_embeddings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_file_from_path
from app.vectorstore.vector_db import VectorDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def build_index(reset: bool = False) -> None:
    data_dir = settings.DATA_DIR

    # Validate data directory
    if not data_dir.exists():
        logger.error("Data directory '%s' does not exist. Create it and add PDF files.", data_dir)
        sys.exit(1)

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in '%s'.", data_dir)
        sys.exit(1)

    logger.info("Found %d PDF file(s) to index:", len(pdf_files))
    for f in pdf_files:
        logger.info("  • %s (%.1f MB)", f.name, f.stat().st_size / 1_048_576)

    # Optionally reset existing index
    embeddings = get_embeddings()
    db = VectorDB(embeddings)

    if reset and settings.FAISS_INDEX_DIR.exists():
        logger.info("--reset flag: deleting existing FAISS index...")
        db.reset()

    # Process each PDF
    total_chunks = 0
    start = time.perf_counter()

    for pdf_path in pdf_files:
        logger.info("=" * 60)
        logger.info("Processing: %s", pdf_path.name)
        logger.info("Step 1/3 — Parsing PDF with Docling (OCR enabled)...")

        t0 = time.perf_counter()
        try:
            documents = load_file_from_path(pdf_path)
        except Exception as exc:
            logger.error("Failed to parse '%s': %s", pdf_path.name, exc)
            continue

        logger.info(
            "  → Extracted %d pages in %.1fs",
            len(documents),
            time.perf_counter() - t0,
        )

        if not documents:
            logger.warning("  → No text extracted from '%s', skipping.", pdf_path.name)
            continue

        logger.info("Step 2/3 — Chunking documents...")
        t1 = time.perf_counter()
        chunks = chunk_documents(documents)
        logger.info(
            "  → Created %d chunks in %.1fs",
            len(chunks),
            time.perf_counter() - t1,
        )

        if not chunks:
            logger.warning("  → No chunks produced, skipping.")
            continue

        logger.info("Step 3/3 — Embedding & saving to FAISS...")
        t2 = time.perf_counter()
        try:
            db.add_documents(chunks)
        except Exception as exc:
            logger.error("Failed to index '%s': %s", pdf_path.name, exc)
            continue

        logger.info(
            "  → Indexed %d chunks in %.1fs",
            len(chunks),
            time.perf_counter() - t2,
        )
        total_chunks += len(chunks)

    # Summary
    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info("✅ Build complete!")
    logger.info("   Total chunks indexed : %d", total_chunks)
    logger.info("   Total time           : %.1fs", elapsed)
    logger.info("   Index location       : %s", settings.FAISS_INDEX_DIR)
    logger.info("")
    logger.info("You can now start the API with: make dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS index from UIT training regulation PDFs."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing FAISS index before building.",
    )
    args = parser.parse_args()
    build_index(reset=args.reset)
