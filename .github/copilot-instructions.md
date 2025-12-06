<!-- Copilot instructions tailored for the RAGify (Ollama) MVP repo -->
# RAGify AI — Copilot Instructions

This file gives focused, actionable guidance for an AI coding assistant working in this repository.

Key points
- **Architecture**: A single FastAPI backend (`main.py`) serves a small static frontend (`static/index.html`) and exposes two APIs: `POST /api/upload` and `POST /api/query`.
- **Data flow**: Upload -> saved under `app/uploads` -> converted to text in `app/services/ingestion.py` -> chunked -> embedded via Ollama in `app/services/rag_service.py` -> stored in ChromaDB under `vectorstore/`.
- **Models / External services**: Uses a local Ollama server at `http://localhost:11434` for both embeddings (`nomic-embed-text`) and chat (`llama3`). The vector DB uses `chromadb` with `duckdb+parquet` persisted to `vectorstore`.

Essential files to inspect
- `main.py` — app entrypoint, CORS, static mount, API endpoints and request/response models.
- `app/services/ingestion.py` — file saving, PDF/DOCX/TXT readers, and `chunk_text(text, chunk_size=800, overlap=200)` (character-based sliding window).
- `app/services/rag_service.py` — embedding calls to Ollama (`embed_texts`), ChromaDB collection usage (`collection.add`, `collection.query`), and chat call `_call_chat_model` which streams responses.
- `app/config.py` — paths: `UPLOAD_DIR` and `VECTOR_DIR` are created at startup.
- `static/index.html` — small frontend that posts to `/api/upload` and `/api/query`; useful for end-to-end manual checks.

Developer workflows and commands
- Install dependencies:
  - `pip install -r requirements.txt`
- Run local Ollama (outside this repo):
  - Ensure Ollama daemon is running and reachable at `http://localhost:11434`.
  - Pull required models before running the app: `ollama pull nomic-embed-text` and `ollama pull llama3`.
- Start the backend (development):
  - `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Quick debug checklist:
  - If embeddings fail: verify Ollama is running and models are pulled; check `OLLAMA_BASE_URL` in `app/services/rag_service.py`.
  - To reset indexed data: stop app and delete the `vectorstore/` directory (Chroma persists there).
  - Uploaded files live in `app/uploads` (configured in `app/config.py`).

Repository conventions & patterns
- Chunking is character-based (not sentence-aware). The default chunk size is `800` with `200` overlap — change in `ingestion.chunk_text` if needed.
- IDs injected into Chroma use the source filename plus an index (`{source}_{i}`); metadata includes `source_file` and `chunk` fields.
- The chat model is called with a strict instruction to answer only from the provided context. Keep prompt modifications localized to `_call_chat_model`.

If you modify embeddings or the retrieval pipeline
- Update `app/services/rag_service.py` only. When switching embedding providers, adapt `embed_texts()` and keep the input/output shape (list[str] -> list[list[float]]).
- Maintain the Chroma collection schema: `ids`, `embeddings`, `documents`, `metadatas`.

Examples to reference when editing
- Adding docs to index: `main.py` -> `/api/upload` uses `ingestion.save_upload`, `load_file_to_text`, `chunk_text`, then `add_documents(chunks, filename)`.
- Query flow: `main.py` -> `/api/query` calls `query_collection(question, top_k)` which embeds the question, queries Chroma, formats context, then calls `_call_chat_model`.

Notes for Copilot-style suggestions
- Prefer minimal, focused edits: adjust chunk sizes or embedding calls where the logic is centralized.
- When suggesting new dependencies, reference `requirements.txt` and include why (e.g., sentence tokenizer for smarter chunking).
- Do not change the UI/backend contract: the frontend expects `/api/upload` and `/api/query` shapes as implemented in `main.py`.

If anything is unclear or you want me to expand any section (run commands, add examples, or include troubleshooting scripts), say which part to iterate on.
