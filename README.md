# RAGify AI – Ollama RAG MVP

FastAPI backend + static frontend for a local RAG demo powered by Ollama (embeddings + chat) and ChromaDB.

## Prerequisites
- Python 3.11 (venv recommended)
- Ollama daemon running on `http://localhost:11434`
- Models pulled in Ollama:
  - `ollama pull nomic-embed-text`
  - `ollama pull llama3`

## Setup
```powershell
python -m venv .venv
& .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (real stack)
```powershell
# optional: extend Ollama timeout for cold starts
$env:RAGIFY_OLLAMA_TIMEOUT = '300'
Remove-Item Env:RAGIFY_MOCK -ErrorAction SilentlyContinue

# start server
& .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
```
Open http://127.0.0.1:8000, upload a file, then ask a question.

## Run (mock mode)
```powershell
$env:RAGIFY_MOCK = '1'
& .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
```
Mock mode skips Ollama and Chroma calls for fast UI testing.

## Smoke test (programmatic)
```powershell
# server must be running
& .\.venv\Scripts\python.exe scripts\smoke_test.py
```
If your models cold-start slowly, increase timeout in the script or set `DEFAULT_TIMEOUT` higher.

## Health check
```
GET /health
```
Returns status, mock flag, and current Ollama timeout.

## Notes
- Vector store persists to `vectorstore/`; uploads go to `uploads/` (ignored by git).
- Increase `RAGIFY_OLLAMA_TIMEOUT` (e.g., `600`) if chat/embeddings need longer.
