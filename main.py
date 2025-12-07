from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import time
import json

from app.services import ingestion
from app.services import rag_service
from app.services.rag_service import index_files, answer_question, is_mock_mode, REQUEST_TIMEOUT, reset_collection
from app.auth import authenticate_user, create_access_token, get_current_user
from app.database import init_db, get_db, test_connection
from app.models import Document
from app.tenant_config import get_tenant_config
from sqlalchemy.orm import Session

import os

app = FastAPI(title="RAGify AI – Ollama RAG Backend (Multi-Tenant)")
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)
security = HTTPBearer()


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Initializing database...")
    try:
        init_db()
        if test_connection():
            logger.info("Database initialized successfully")
        else:
            logger.warning("Database connection test failed - app will run without Postgres")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. App will run without Postgres.")


@app.on_event("shutdown")
async def shutdown_event():
    """Close HTTP client on server shutdown."""
    if rag_service._http_client is not None and not rag_service._http_client.is_closed:
        await rag_service._http_client.aclose()
        logger.info("HTTP client closed")

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


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    tenant_id: str


@app.post("/api/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Public endpoint: authenticate user and return JWT token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(user["username"], user["tenant_id"])
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        tenant_id=user["tenant_id"]
    )


@app.get("/api/config")
async def get_config(current_user: dict = Depends(get_current_user)):
    """Protected endpoint: return tenant-specific configuration."""
    tenant_id = current_user["tenant_id"]
    config = get_tenant_config(tenant_id)
    if not config:
        raise HTTPException(status_code=404, detail="Tenant configuration not found")
    return config


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.post("/api/upload")
async def upload(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint: upload and index documents for the authenticated user's tenant."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    tenant_id = current_user["tenant_id"]
    total_chunks = 0

    for file in files:
        start = time.time()
        logger.info("Uploading file %s for tenant %s", file.filename, tenant_id)
        raw_bytes = await file.read()
        logger.info("Saved %d bytes for %s", len(raw_bytes), file.filename)
        saved_path = ingestion.save_upload(raw_bytes, file.filename)
        logger.info("Saved file to %s (took %.2fs)", saved_path, time.time() - start)

        # Create DB record if database is available
        if db:
            try:
                doc_record = Document(
                    tenant_id=tenant_id,
                    filename=file.filename,
                    file_path=saved_path,
                    status="indexing"
                )
                db.add(doc_record)
                db.commit()
                logger.info("Created DB record for %s", file.filename)
            except Exception as e:
                logger.warning("Could not create DB record: %s", e)
                db.rollback()

        t0 = time.time()
        text = ingestion.load_file_to_text(saved_path)
        logger.info("Loaded text length %d (took %.2fs)", len(text), time.time() - t0)

        t1 = time.time()
        try:
            chunks = ingestion.chunk_text(text)
            logger.info("Created %d chunks (took %.2fs)", len(chunks), time.time() - t1)
        except MemoryError:
            logger.exception("Out of memory while chunking text; falling back to single-chunk")
            chunks = [text]
        except Exception:
            logger.exception("Unexpected error while chunking text; falling back to single-chunk")
            chunks = [text]

        t2 = time.time()
        logger.info("Indexing chunks for %s...", saved_path)
        try:
            num = await index_files(tenant_id, chunks, os.path.basename(saved_path))
            logger.info("Indexed %d chunks for %s (took %.2fs)", num, saved_path, time.time() - t2)
            total_chunks += num
            
            # Update DB record to indexed
            if db:
                try:
                    doc_record.status = "indexed"
                    db.commit()
                except Exception as e:
                    logger.warning("Could not update DB record: %s", e)
        except Exception as e:
            logger.exception("Indexing failed for %s: %s", file.filename, e)
            if db:
                try:
                    doc_record.status = "failed"
                    doc_record.error_message = str(e)
                    db.commit()
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
        total_chunks += num

    return {"status": "ok", "indexed_chunks": total_chunks}


@app.post("/api/query")
async def query(payload: QueryRequest, current_user: dict = Depends(get_current_user)):
    """Protected endpoint: query documents in the authenticated user's tenant collection."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    tenant_id = current_user["tenant_id"]
    answer_gen, sources = await answer_question(tenant_id, payload.question, payload.top_k)
    
    # Stream the answer tokens back to client
    async def stream_response():
        # First send the sources as a JSON prefix
        yield json.dumps({"sources": sources}) + "\n"
        
        # Then stream answer tokens
        async for token in answer_gen:
            yield json.dumps({"token": token}) + "\n"
    
    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.post("/api/reset")
async def reset(current_user: dict = Depends(get_current_user)):
    """
    Protected endpoint: reset the vector store for the authenticated user's tenant.
    WARNING: This is destructive and cannot be undone!
    """
    tenant_id = current_user["tenant_id"]
    try:
        reset_collection(tenant_id)
        return {"status": "ok", "message": f"Vector store reset successfully for tenant {tenant_id}"}
    except Exception as e:
        logger.exception("Reset failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/api/documents")
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint: list all documents for the authenticated user's tenant."""
    # Check if database is available
    if db is None:
        return {"documents": [], "message": "Database not available"}
    
    tenant_id = current_user["tenant_id"]
    try:
        docs = db.query(Document).filter(Document.tenant_id == tenant_id).order_by(Document.created_at.desc()).all()
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat(),
                    "error_message": doc.error_message
                }
                for doc in docs
            ]
        }
    except Exception as e:
        logger.warning("Failed to list documents: %s", e)
        return {"documents": [], "message": "Database error"}
