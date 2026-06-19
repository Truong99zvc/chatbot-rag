# UIT Academic Policies Chatbot

A RAG-powered chatbot for answering questions about the **official regulations, policies, and procedures for UIT's formal undergraduate programs** — University of Information Technology, VNU-HCM.

Students can ask any question about academic rules and procedures and receive answers with citations to specific articles and clauses.

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| Web UI | HTML / CSS / JavaScript (served by FastAPI) |
| LLM | Mistral-7B-Instruct-v0.3 (via HuggingFace Inference API) |
| Embeddings | multilingual-e5-large (local, sentence-transformers) |
| Vector Store | FAISS (local) |
| PDF Parsing | Docling (with built-in OCR for scanned pages) |
| Session Storage | JSON file (disk-backed) |

## Architecture

```
chatbot-rag/
├── app/
│   ├── main.py                  # FastAPI entry point + static file mount
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
├── static/
│   ├── index.html               # Chat web interface (dark-mode)
│   ├── style.css                # Premium dark UI styles
│   └── app.js                   # Chat logic (fetch, markdown, session)
├── data/
│   └── *.pdf                    # UIT regulation PDF files
├── scripts/
│   └── build_index.py           # One-time script to build the FAISS index
├── vectorstores/faiss/          # FAISS index (generated after running build-index)
├── tests/
│   └── evaluation/              # RAGAS evaluation pipeline
└── ...
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
# Open .env and fill in your HF_TOKEN
```
Get your free HuggingFace token at: https://huggingface.co/settings/tokens

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

Open **http://localhost:8000** in your browser — the chat interface loads automatically.

API docs (Swagger): http://localhost:8000/docs

---

## Web Interface

The chat UI is served directly by FastAPI at `http://localhost:8000`.

| Feature | Description |
|---|---|
| **Chat** | Free-form Q&A with Markdown rendering and source citations |
| **Article lookup** | Sidebar search field — enter a number to retrieve that Điều's content |
| **Suggested questions** | Quick-start chips for the most common student queries |
| **Session memory** | Conversation history persisted per browser session |
| **Index status** | Header indicator shows green when FAISS index is ready |
| **New chat** | Reset conversation with one click |

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
| `HF_TOKEN` | *(required)* | HuggingFace API token |
| `LLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | HF Inference API model |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Local embedding model (multilingual) |
| `FAISS_INDEX_DIR` | `vectorstores/faiss/current_index` | FAISS index path |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |

---

## Evaluation (RAGAS)

The chatbot is evaluated using **[RAGAS](https://docs.ragas.io/)** — a framework purpose-built for RAG pipeline evaluation. Google Gemini is used as the evaluator LLM (same API key as the app).

### Metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **Faithfulness** | Does the answer contain only information from the retrieved context? | Most critical — prevents hallucinated regulations |
| **Answer Relevancy** | Is the answer relevant and on-topic for the question? | Ensures answers address what students actually asked |
| **Context Precision** | Are the retrieved chunks relevant (no noisy irrelevant chunks)? | Measures retrieval quality |
| **Context Recall** | Were all necessary information chunks retrieved? | Measures retrieval coverage |

### Eval dataset

`tests/evaluation/eval_dataset.json` — 20 hand-crafted Q&A pairs covering key policy topics:
graduation requirements, grading scale, academic warnings, leave of absence, credit transfer, attendance rules, and more.

### Running the evaluation

```bash
# Step 1 — Generate answers from the live RAG pipeline
make generate-eval-answers
# Output: tests/evaluation/evaluated_answers.json

# Step 2 — Run RAGAS evaluation (may take ~10 minutes for 20 questions)
make evaluate
# Output: tests/evaluation/evaluation_results.json   (per-question scores)
#         tests/evaluation/evaluation_summary.json   (aggregated averages)
```

> **Prerequisites:** FAISS index must be built first (`make build-index`).

### Example output

```
📊 RAGAS Evaluation Summary — UIT Academic Policies Chatbot
--------------------------------------------------
  Faithfulness           0.912  [██████████████████░░]
  AnswerRelevancy        0.883  [█████████████████░░░]
  ContextPrecision       0.847  [████████████████░░░░]
  ContextRecall          0.791  [███████████████░░░░░]
--------------------------------------------------
  Total samples : 20
  Eval duration : 487.3s
```

---

## Development

```bash
make test                    # Run pytest
make lint                    # Lint with ruff
make format                  # Auto-format with ruff
make build-index-reset       # Delete existing index and rebuild from scratch
make generate-eval-answers   # Generate RAG answers for the eval dataset
make evaluate                # Run RAGAS evaluation
make clean                   # Remove __pycache__ and build artifacts
```
