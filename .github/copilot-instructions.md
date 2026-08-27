<!-- Copilot instructions tailored for the RAGify (Ollama) MVP repo -->
# RAGify AI — Copilot Instructions

This file gives focused, actionable guidance for an AI coding assistant working in this repository.

Key points
- **Architecture**: The monorepo contains a FastAPI backend under `apps/backend` and a React/Vite frontend under `apps/frontend`. The backend also retains legacy static pages and exposes APIs including `POST /api/upload` and `POST /api/query`.
- **Data flow**: Upload -> saved under `apps/backend/app/uploads` -> converted to text in `apps/backend/app/services/ingestion.py` -> chunked -> embedded via Ollama in `apps/backend/app/services/rag_service.py` -> stored in ChromaDB under `apps/backend/vectorstore/`.
- **Models / External services**: Uses a local Ollama server at `http://localhost:11434` for both embeddings (`nomic-embed-text`) and chat (`llama3`). The vector DB uses `chromadb` with `duckdb+parquet` persisted to `vectorstore`.

Essential files to inspect
- `apps/backend/main.py` — app entrypoint, CORS, static mount, API endpoints and request/response models.
- `apps/backend/app/services/ingestion.py` — file saving, PDF/DOCX/TXT readers, and `chunk_text(text, chunk_size=800, overlap=200)` (character-based sliding window).
- `apps/backend/app/services/rag_service.py` — embedding calls to Ollama (`embed_texts`), ChromaDB collection usage (`collection.add`, `collection.query`), and chat call `_call_chat_model` which streams responses.
- `apps/backend/app/config.py` — paths: `UPLOAD_DIR` and `VECTOR_DIR` are created at startup.
- `apps/frontend/src/` — React UI, API client, and SSE integration.

Developer workflows and commands
- Install dependencies:
  - `pip install -r requirements.txt`
- Run local Ollama (outside this repo):
  - Ensure Ollama daemon is running and reachable at `http://localhost:11434`.
  - Pull required models before running the app: `ollama pull nomic-embed-text` and `ollama pull llama3`.
- Start the backend (development):
  - `npm run dev:backend` from the repository root
- Start both services:
  - `npm run dev` from the repository root
- Quick debug checklist:
  - If embeddings fail: verify Ollama is running and models are pulled; check `OLLAMA_BASE_URL` in `apps/backend/app/services/rag_service.py`.
  - To reset indexed data: stop app and delete the `apps/backend/vectorstore/` directory (Chroma persists there).
  - Uploaded files live in `apps/backend/app/uploads` (configured in `apps/backend/app/config.py`).

Repository conventions & patterns
- Chunking is character-based (not sentence-aware). The default chunk size is `800` with `200` overlap — change in `ingestion.chunk_text` if needed.
- IDs injected into Chroma use the source filename plus an index (`{source}_{i}`); metadata includes `source_file` and `chunk` fields.
- The chat model is called with a strict instruction to answer only from the provided context. Keep prompt modifications localized to `_call_chat_model`.

If you modify embeddings or the retrieval pipeline
- Update `apps/backend/app/services/rag_service.py` only. When switching embedding providers, adapt `embed_texts()` and keep the input/output shape (list[str] -> list[list[float]]).
- Maintain the Chroma collection schema: `ids`, `embeddings`, `documents`, `metadatas`.

Examples to reference when editing
- Adding docs to index: `main.py` -> `/api/upload` uses `ingestion.save_upload`, `load_file_to_text`, `chunk_text`, then `add_documents(chunks, filename)`.
- Query flow: `main.py` -> `/api/query` calls `query_collection(question, top_k)` which embeds the question, queries Chroma, formats context, then calls `_call_chat_model`.

Notes for Copilot-style suggestions
- Prefer minimal, focused edits: adjust chunk sizes or embedding calls where the logic is centralized.
- When suggesting new dependencies, reference `requirements.txt` and include why (e.g., sentence tokenizer for smarter chunking).
- Do not change the UI/backend contract: the frontend expects `/api/upload` and `/api/query` shapes as implemented in `main.py`.

If anything is unclear or you want me to expand any section (run commands, add examples, or include troubleshooting scripts), say which part to iterate on.
