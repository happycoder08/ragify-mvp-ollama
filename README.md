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

# optional: tune chunk size for faster/better results (default: 500 chars, 100 overlap)
# smaller chunks = faster embeddings; larger chunks = better context
# $env:RAGIFY_CHUNK_SIZE = '300'
# $env:RAGIFY_CHUNK_OVERLAP = '50'

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

## Performance Tuning

**Embedding Cache**: Single-query embeddings are cached to avoid redundant requests.

**Batch Embeddings**: Multiple chunks use batch API calls instead of parallel individual requests (more efficient).

**Connection Pooling**: Persistent HTTP client reuses connections across all Ollama API calls.

**Configurable Chunk Size**:
```powershell
# For faster performance (smaller chunks = faster embeddings)
$env:RAGIFY_CHUNK_SIZE = '300'
$env:RAGIFY_CHUNK_OVERLAP = '50'

# For better context (larger chunks = slower but more coherent)
$env:RAGIFY_CHUNK_SIZE = '1000'
$env:RAGIFY_CHUNK_OVERLAP = '200'
```

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

## Cleanup & Reset

### Option 1: Reset via API
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/reset" -Method Post
```
Clears all indexed embeddings and uploads. Server must be running.

### Option 2: Run cleanup script
```powershell
& .\.venv\Scripts\python.exe scripts\cleanup_vectorstore.py
```
Interactive cleanup for both vector store and uploaded files.

### Option 3: Manual cleanup
```powershell
# Remove vector store directory
Remove-Item -Recurse -Force vectorstore/

# Remove uploads directory
Remove-Item -Recurse -Force app/uploads/
```

## Notes
- Vector store persists to `vectorstore/`; uploads go to `app/uploads/` (ignored by git).
- Increase `RAGIFY_OLLAMA_TIMEOUT` (e.g., `600`) if chat/embeddings need longer.
