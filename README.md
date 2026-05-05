# Chatbot PDF RAG

A Streamlit-based PDF RAG chatbot built with LangChain, Google Gemini, and a local FAISS vector store.

## Features

- Upload one or multiple PDF files
- Extract text and split into retrieval chunks
- Build and persist a local FAISS index
- Ask questions grounded in uploaded documents
- Persist conversation memory to disk
- Reuse recent conversation context in follow-up questions
- Export conversation history as CSV

## Requirements

- Python 3.10+
- Google AI API key (Gemini)

## Installation

### Option 1: uv

```bash
uv sync
```

### Option 2: pip + venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## API Key Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

You can also enter the API key directly in the Streamlit sidebar.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to Use

1. Enter your Google API key.
2. Upload one or more PDF files.
3. Click **Submit and Process** to build the FAISS index.
4. Ask questions in the chat box.
5. Optionally download conversation history as CSV.

## Conversation Memory

The app now includes persistent conversation memory:

- Chat history is stored in `vectorstores/faiss/conversation_memory.json`.
- Memory is loaded automatically when the app starts.
- The latest conversation turns are added to the LLM prompt to support follow-up questions.
- Clicking **Clear Chat** clears both UI history and persisted memory.

This memory is conversation-level context. It does not replace document retrieval from FAISS.

## Project Structure

```text
chatbot-pdf-rag/
├─ app.py
├─ pyproject.toml
├─ README.md
└─ vectorstores/
   └─ faiss/
      ├─ current_index/            # created after processing PDFs
      └─ conversation_memory.json  # created after first chat message
```

## Notes

- Scanned/image-only PDFs may not extract text correctly with PyPDF2.
- The current implementation supports Google AI (Gemini).
- Local FAISS is good for prototypes and small projects.

## Suggested Next Improvements

- Add OCR support for scanned PDFs
- Add source metadata (file name, page number) to answers
- Add answer citations for retrieved passages
- Add automated tests for extraction and retrieval flow
