# UIT Academic Policies Chatbot

A RAG-powered chatbot for answering questions about the **official regulations, policies, and procedures for UIT's formal undergraduate programs** — University of Information Technology, VNU-HCM.

Students can ask any question about academic rules and procedures and receive answers with citations to specific articles and clauses.

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google text-embedding-004 |
| Vector Store | FAISS (local) |
| PDF Parsing | Docling (with built-in OCR for scanned pages) |
| Session Storage | JSON file (disk-backed) |

## Architecture

```
chatbot-rag/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── api/
│   │   ├── rag.py               # Endpoints: /query, /search, /sessions
│   │   └── health.py            # Health check + index status
│   ├── config/settings.py       # Pydantic settings (env-driven)
│   ├── rag/
│   │   ├── pipeline.py          # RAG orchestrator + JSON session store
│   │   ├── retriever.py         # FAISS retrieval + article-based search
│   │   ├── generator.py         # Gemini LLM wrapper
│   │   └── prompt_builder.py    # UIT academic advisor prompt
│   ├── ingestion/
│   │   ├── loader.py            # Load PDF from local path
│   │   ├── parser.py            # Docling PDF parser (OCR enabled)
│   │   └── chunker.py           # Markdown-aware chunker for legal documents
│   ├── embeddings/              # Google embedding wrapper
│   ├── vectorstore/             # FAISS load / save / merge / reset
│   └── middleware/              # Logger, rate limiter, error handler
├── data/
│   └── *.pdf                    # UIT regulation PDF files
├── scripts/
│   └── build_index.py           # One-time script to build the FAISS index
├── vectorstores/faiss/          # FAISS index (generated after running build-index)
└── tests/
```

## Quick Start

### 1. Install dependencies
```bash
uv sync
# or
make install
```

> **Note:** Docling will automatically download its OCR model (~few hundred MB) on the first run.

### 2. Configure your API key
```bash
cp .env.example .env
# Open .env and fill in your GOOGLE_API_KEY
```
Get your API key at: https://ai.google.dev/

### 3. Build the FAISS index from the PDF
```bash
make build-index
```
This will:
- Read all PDF files in the `data/` directory
- Parse them with Docling (automatically applies OCR to scanned pages)
- Generate embeddings and save the FAISS index to disk

> For a 250-page document this takes approximately **5–15 minutes**. It only needs to be run **once**.

### 4. Start the API server
```bash
make dev
```
API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + index status |
| `POST` | `/api/v1/rag/query` | Free-form Q&A about UIT regulations |
| `GET` | `/api/v1/rag/search?article=N` | Look up a specific article by number |
| `GET` | `/api/v1/rag/sessions/{id}` | Get conversation history for a session |
| `DELETE` | `/api/v1/rag/sessions/{id}` | Clear a session's conversation history |

### Example — Free-form Q&A
```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the requirements to qualify for graduation?",
    "session_id": "student-123"
  }'
```

**Response:**
```json
{
  "answer": "According to Article 30 of the UIT Training Regulations, students qualify for graduation when they have: accumulated the required number of credits in their program...",
  "sources": "- **qui-che-qui-dinh.pdf**, page 45",
  "session_id": "student-123"
}
```

### Example — Look up a specific article
```bash
curl "http://localhost:8000/api/v1/rag/search?article=15"
```

### Example — Get conversation history
```bash
curl "http://localhost:8000/api/v1/rag/sessions/student-123"
```

---

## Configuration

All settings are controlled via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Google AI Studio API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `FAISS_INDEX_DIR` | `vectorstores/faiss/current_index` | FAISS index path |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |

---

## Development

```bash
make test                # Run pytest
make lint                # Lint with ruff
make format              # Auto-format with ruff
make build-index-reset   # Delete existing index and rebuild from scratch
make clean               # Remove __pycache__ and build artifacts
```
