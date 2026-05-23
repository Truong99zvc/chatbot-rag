# Chatbot PDF RAG

A production-ready PDF Question-Answering API built with **FastAPI**, **LangChain**, **Google Gemini**, and **FAISS**.

## Architecture

```
rag-project/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── api/                   # Route handlers (rag, documents, health)
│   ├── middleware/            # Error handler, request logger, rate limiter
│   ├── rag/                   # Pipeline, retriever, generator, prompt builder
│   ├── ingestion/             # Loader, parser (PDF/TXT), chunker
│   ├── embeddings/            # Google text-embedding-004
│   ├── vectorstore/           # FAISS operations (add, load, merge, reset)
│   ├── db/                    # PostgreSQL models & async session
│   ├── config/                # Pydantic settings (env-driven)
│   └── utils/                 # Helper functions
├── tests/
├── docker/
├── .github/workflows/
├── requirements.txt
├── Makefile
└── .env.example
```

## Quick Start

### 1. Setup environment
```bash
cp .env.example .env
# Fill in GOOGLE_API_KEY in .env
```

### 2. Install dependencies
```bash
make install
```

### 3. Run locally (with auto-reload)
```bash
make dev
```
API docs: http://localhost:8000/docs

### 4. Run with Docker
```bash
make docker-up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/api/v1/documents/upload` | Upload & index PDF/TXT files |
| GET | `/api/v1/documents/list` | List uploaded documents |
| DELETE | `/api/v1/documents/reset` | Reset knowledge base |
| POST | `/api/v1/rag/query` | Ask a question |

### Example — Upload a PDF
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@my_document.pdf"
```

### Example — Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Tóm tắt nội dung chính của tài liệu", "session_id": "user-123"}'
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI + Uvicorn |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google text-embedding-004 |
| Vector Store | FAISS (local) |
| PDF Parsing | pypdf |
| ORM | SQLAlchemy 2.0 (async) |
| Database | PostgreSQL 16 |
| Container | Docker + Docker Compose |

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(required)* | Google AI Studio API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL URL |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

## Development

```bash
make test    # run pytest
make lint    # lint with ruff
make format  # auto-format with ruff
make clean   # remove __pycache__ and build artifacts
```
