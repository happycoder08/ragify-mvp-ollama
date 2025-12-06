from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time
import json

from app.services import ingestion
from app.services.rag_service import add_documents, query_collection, is_mock_mode, REQUEST_TIMEOUT, reset_collection

import os

app = FastAPI(title="RAGify AI – Ollama RAG Backend")
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

# Allow simple CORS for local demo / front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mock_mode": is_mock_mode(),
        "ollama_timeout": REQUEST_TIMEOUT,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_chunks = 0

    for file in files:
        start = time.time()
        logger.info("Uploading file %s", file.filename)
        raw_bytes = await file.read()
        logger.info("Saved %d bytes for %s", len(raw_bytes), file.filename)
        saved_path = ingestion.save_upload(raw_bytes, file.filename)
        logger.info("Saved file to %s (took %.2fs)", saved_path, time.time() - start)

        t0 = time.time()
        text = ingestion.load_file_to_text(saved_path)
        logger.info("Loaded text length %d (took %.2fs)", len(text), time.time() - t0)

        t1 = time.time()
        try:
            chunks = ingestion.chunk_text(text)
            logger.info("Created %d chunks (took %.2fs)", len(chunks), time.time() - t1)
        except MemoryError:
            logger.exception("Out of memory while chunking text; falling back to single-chunk")
            # Fallback: use the whole document as a single chunk so indexing can proceed
            chunks = [text]
        except Exception:
            logger.exception("Unexpected error while chunking text; falling back to single-chunk")
            chunks = [text]

        t2 = time.time()
        logger.info("Indexing chunks for %s...", saved_path)
        num = await add_documents(chunks, os.path.basename(saved_path))
        logger.info("Indexed %d chunks for %s (took %.2fs)", num, saved_path, time.time() - t2)
        total_chunks += num

    return {"status": "ok", "indexed_chunks": total_chunks}


@app.post("/api/query")
async def query(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer_gen, sources = await query_collection(payload.question, payload.top_k)
    
    # Stream the answer tokens back to client
    async def stream_response():
        # First send the sources as a JSON prefix
        yield json.dumps({"sources": sources}) + "\n"
        
        # Then stream answer tokens
        async for token in answer_gen:
            yield json.dumps({"token": token}) + "\n"
    
    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.post("/api/reset")
async def reset():
    """
    Reset the vector store and clear all indexed documents.
    WARNING: This is destructive and cannot be undone!
    """
    try:
        reset_collection()
        return {"status": "ok", "message": "Vector store reset successfully"}
    except Exception as e:
        logger.exception("Reset failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
