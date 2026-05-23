from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.embeddings.embedder import get_embeddings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_uploaded_file
from app.vectorstore.vector_db import VectorDB

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED, summary="Upload & index PDF/TXT files")
async def upload_documents(files: List[UploadFile] = File(...)) -> dict:
    """
    Upload one or more PDF or TXT files.  
    Each file is parsed, chunked, embedded and merged into the FAISS index.
    """
    processed: list[dict] = []
    errors: list[dict] = []

    embeddings = get_embeddings()
    db = VectorDB(embeddings)

    for file in files:
        try:
            docs = await load_uploaded_file(file)
            chunks = chunk_documents(docs)
            db.add_documents(chunks)
            processed.append({"file": file.filename, "chunks": len(chunks)})
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": file.filename, "error": str(exc)})

    return {
        "processed": processed,
        "errors": errors,
        "total_files": len(files),
    }


@router.get("/list", summary="List uploaded documents")
async def list_documents() -> dict:
    """Return a list of files currently in the upload directory."""
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": files, "count": len(files)}


@router.delete("/reset", summary="Reset knowledge base")
async def reset_knowledge_base() -> dict:
    """
    Delete the FAISS index from disk.  
    Re-upload documents to rebuild the knowledge base.
    """
    embeddings = get_embeddings()
    db = VectorDB(embeddings)
    db.reset()
    return {"status": "Knowledge base reset successfully."}


@router.delete("/{filename}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def delete_document(filename: str) -> dict:  # noqa: ARG001
    """
    Per-document deletion is not supported by FAISS natively.  
    Use /reset to clear the index and re-upload remaining documents.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "FAISS does not support per-document deletion. "
            "Use DELETE /api/v1/documents/reset and re-upload remaining files."
        ),
    )
