from typing import List, Optional
import uuid
import contextvars

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
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
from app.services import clients
from app.services.rag_service import index_files, answer_question, is_mock_mode, reset_collection
from app.auth import authenticate_user, create_access_token, get_current_user
from app.database import init_db, get_db, test_connection
from app.models import Document, Conversation, Message
from app.tenant_config import get_tenant_config
from app.guardrails import (
    get_rate_limiter,
    get_guardrail_config,
    validate_file_extension,
    validate_file_size,
    validate_file_count,
)
from app.config import (
    RAGIFY_MODE,
    DEFAULT_MODE,
    TOP_K_FAST,
    TOP_K_FULL,
    ENABLE_TIMING_LOGS,
    MAX_CONVERSATION_TURNS,
    get_config_summary,
)
from sqlalchemy.orm import Session
from sqlalchemy import func

import os

app = FastAPI(title="RAGify AI – Ollama RAG Backend (Multi-Tenant)")
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)
security = HTTPBearer()

# Context variable for request tracking
request_id_var = contextvars.ContextVar('request_id', default=None)

def log_timing(event: str, duration: float, tenant_id: str, **extra):
    """Log timing events with structured JSON."""
    if not ENABLE_TIMING_LOGS:
        return
    request_id = request_id_var.get()
    log_data = {
        "event": event,
        "duration_ms": round(duration * 1000, 2),
        "tenant_id": tenant_id,
        "request_id": request_id,
        **extra
    }
    logger.info(json.dumps(log_data))


@app.on_event("startup")
async def startup_event():
    """Initialize database and clients on startup."""
    logger.info("Initializing database...")
    try:
        init_db()
        if test_connection():
            logger.info("Database initialized successfully")
        else:
            logger.warning("Database connection test failed - app will run without Postgres")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. App will run without Postgres.")
    
    # Initialize shared clients
    try:
        clients.initialize_chroma_client()
        await clients.initialize_http_client()
        logger.info("All clients initialized successfully")
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        raise
    
    # Log active configuration
    config_summary = get_config_summary()
    logger.info("RAGify configuration: %s", json.dumps(config_summary, indent=2))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup and close all clients on server shutdown."""
    logger.info("Shutting down...")
    await clients.shutdown_clients()

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
    try:
        return {
            "status": "ok",
            "mock_mode": is_mock_mode(),
            "ragify_mode": RAGIFY_MODE,
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "degraded", "error": str(e)}


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


@app.get("/api/guardrails")
async def get_guardrails(current_user: dict = Depends(get_current_user)):
    """Protected endpoint: return tenant-specific guardrail limits."""
    tenant_id = current_user["tenant_id"]
    config = get_guardrail_config(tenant_id)
    return config.to_dict()


@app.get("/api/rate-limit-status")
async def get_rate_limit_status(current_user: dict = Depends(get_current_user)):
    """Protected endpoint: return current rate limit usage for tenant."""
    tenant_id = current_user["tenant_id"]
    rate_limiter = get_rate_limiter()
    usage = rate_limiter.get_current_usage(tenant_id)
    return usage


@app.get("/api/system/config")
async def get_system_config():
    """Public endpoint: return active RAGify system configuration."""
    return get_config_summary()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    mode: str = DEFAULT_MODE  # Configured via RAGIFY_MODE (dev/demo/prod)
    conversation_id: Optional[int] = None  # Optional conversation context
    doc_ids: Optional[List[int]] = None  # Optional document IDs to filter search scope


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class MessageCreate(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[str]] = None


@app.post("/api/conversations")
async def create_conversation(
    payload: ConversationCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation for the authenticated user."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    conversation = Conversation(
        tenant_id=tenant_id,
        title=payload.title or f"Conversation {int(time.time())}"
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    logger.info(f"Created conversation {conversation.id} for tenant {tenant_id}")
    return conversation.to_dict()


@app.get("/api/conversations")
async def list_conversations(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50
):
    """List all conversations for the authenticated user."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    conversations = db.query(Conversation)\
        .filter(Conversation.tenant_id == tenant_id)\
        .order_by(Conversation.updated_at.desc())\
        .limit(limit)\
        .all()
    
    return [conv.to_dict() for conv in conversations]


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific conversation with all messages."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id, Conversation.tenant_id == tenant_id)\
        .first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation.to_dict(include_messages=True)


@app.post("/api/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: int,
    payload: MessageCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a message to a conversation."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    # Verify conversation exists and belongs to tenant
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id, Conversation.tenant_id == tenant_id)\
        .first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Create message
    sources_json = json.dumps(payload.sources) if payload.sources else None
    message = Message(
        conversation_id=conversation_id,
        role=payload.role,
        content=payload.content,
        sources=sources_json
    )
    
    db.add(message)
    conversation.updated_at = func.now()  # Update conversation timestamp
    db.commit()
    db.refresh(message)
    
    logger.info(f"Added {payload.role} message to conversation {conversation_id}")
    return message.to_dict()


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id, Conversation.tenant_id == tenant_id)\
        .first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.delete(conversation)
    db.commit()
    
    logger.info(f"Deleted conversation {conversation_id} for tenant {tenant_id}")
    return {"status": "ok", "deleted_id": conversation_id}


async def process_document_background(doc_id: int, tenant_id: str, file_path: str, filename: str):
    """
    Background task to process and index a document.
    Updates document status in database.
    """
    from app.database import SessionLocal
    db = SessionLocal()
    
    try:
        logger.info(f"Background processing started for document {doc_id}: {filename}")
        
        # Load and parse document
        t0 = time.time()
        text = ingestion.load_file_to_text(file_path)
        log_timing("document_parsing", time.time() - t0, tenant_id, filename=filename, text_length=len(text))

        # Chunk text
        t1 = time.time()
        try:
            chunks = ingestion.chunk_text(text)
            log_timing("text_chunking", time.time() - t1, tenant_id, filename=filename, num_chunks=len(chunks))
        except MemoryError:
            logger.exception("Out of memory while chunking text; falling back to single-chunk")
            chunks = [text]
        except Exception:
            logger.exception("Unexpected error while chunking text; falling back to single-chunk")
            chunks = [text]

        # Index chunks with doc_id for filtering
        t2 = time.time()
        logger.info("Indexing chunks for %s...", file_path)
        num = await index_files(tenant_id, chunks, filename, doc_id=doc_id)
        log_timing("indexing_total", time.time() - t2, tenant_id, filename=filename, num_chunks=num)
        
        # Update DB record to indexed
        doc_record = db.query(Document).filter(Document.id == doc_id).first()
        if doc_record:
            doc_record.status = "indexed"
            doc_record.error_message = None
            db.commit()
            logger.info(f"Document {doc_id} indexed successfully ({num} chunks)")
        
    except Exception as e:
        logger.exception(f"Background indexing failed for document {doc_id}: {e}")
        doc_record = db.query(Document).filter(Document.id == doc_id).first()
        if doc_record:
            doc_record.status = "failed"
            doc_record.error_message = str(e)
            db.commit()
    finally:
        db.close()


@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint: upload documents and start background indexing."""
    logger.info(f"Upload endpoint called with {len(files) if files else 0} files")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    tenant_id = current_user["tenant_id"]
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    # Validate file count
    valid, error_msg = validate_file_count(len(files), tenant_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Calculate total upload size and validate each file
    total_upload_size_bytes = 0
    for file in files:
        # Validate file extension
        valid, error_msg = validate_file_extension(file.filename, tenant_id)
        if not valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read file size (FastAPI UploadFile doesn't expose size directly)
        content = await file.read()
        file_size = len(content)
        total_upload_size_bytes += file_size
        
        # Validate file size
        valid, error_msg = validate_file_size(file_size, tenant_id)
        if not valid:
            raise HTTPException(status_code=413, detail=error_msg)
        
        # Reset file pointer for later reading
        await file.seek(0)
    
    # Check rate limits
    total_upload_mb = total_upload_size_bytes / (1024 * 1024)
    rate_limiter = get_rate_limiter()
    allowed, error_msg = rate_limiter.check_rate_limit(tenant_id, total_upload_mb)
    if not allowed:
        raise HTTPException(status_code=429, detail=error_msg)
    
    # Record request for rate limiting
    rate_limiter.record_request(tenant_id, total_upload_mb)
    
    uploaded_docs = []
    overall_start = time.time()
    logger.info(f"Processing {len(files)} files for upload")

    for file in files:
        file_start = time.time()
        logger.info(f"Processing file: {file.filename}")
        raw_bytes = await file.read()
        logger.info("Saved %d bytes for %s", len(raw_bytes), file.filename)
        t_save = time.time()
        saved_path = ingestion.save_upload(raw_bytes, file.filename)
        log_timing("file_save", time.time() - t_save, tenant_id, filename=file.filename, bytes=len(raw_bytes))

        # Create DB record with "pending" status
        doc_record = None
        if db:
            try:
                doc_record = Document(
                    tenant_id=tenant_id,
                    filename=file.filename,
                    file_path=saved_path,
                    status="pending"
                )
                db.add(doc_record)
                db.commit()
                db.refresh(doc_record)
                logger.info("Created DB record for %s (id=%d)", file.filename, doc_record.id)
                uploaded_docs.append(doc_record.to_dict())
                
            except Exception as e:
                logger.exception("Could not create DB record: %s", e)
                if db:
                    db.rollback()
                # Continue without DB record - still process the document
                logger.warning("Continuing without DB record for %s", file.filename)
        else:
            logger.info("No DB available, processing %s without database record", file.filename)
        
        # Schedule background processing regardless of DB status
        doc_id = doc_record.id if doc_record else -1  # Use -1 to indicate no DB record
        background_tasks.add_task(
            process_document_background,
            doc_id,
            tenant_id,
            saved_path,
            file.filename
        )
        logger.info(f"Scheduled background processing for {file.filename} (doc_id={doc_id})")
        
        log_timing("file_upload_complete", time.time() - file_start, tenant_id, filename=file.filename)

    log_timing("upload_complete", time.time() - overall_start, tenant_id, files_count=len(files))
    return {
        "status": "ok", 
        "message": f"{len(files)} file(s) uploaded. Processing in background.",
        "documents": uploaded_docs,
        "files_processed": len(files),
        "files_with_db_record": len(uploaded_docs)
    }


@app.post("/api/query")
async def query(
    payload: QueryRequest, 
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint: query documents in the authenticated user's tenant collection."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    tenant_id = current_user["tenant_id"]
    
    # Check rate limits
    rate_limiter = get_rate_limiter()
    allowed, error_msg = rate_limiter.check_rate_limit(tenant_id, upload_size_mb=0)
    if not allowed:
        raise HTTPException(status_code=429, detail=error_msg)
    
    # Record request for rate limiting
    rate_limiter.record_request(tenant_id, upload_size_mb=0)
    
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    # Adjust top_k based on mode (using config defaults)
    mode = payload.mode.lower() if payload.mode else DEFAULT_MODE
    top_k = TOP_K_FAST if mode == "fast" else (payload.top_k if payload.top_k else TOP_K_FULL)
    
    # Retrieve conversation history if conversation_id provided
    conversation_history = []
    conversation = None
    if payload.conversation_id and db:
        # Verify conversation ownership
        conversation = db.query(Conversation)\
            .filter(Conversation.id == payload.conversation_id, Conversation.tenant_id == tenant_id)\
            .first()
        
        if conversation:
            # Get last N messages for context (configured per mode)
            messages = db.query(Message)\
                .filter(Message.conversation_id == payload.conversation_id)\
                .order_by(Message.created_at.desc())\
                .limit(MAX_CONVERSATION_TURNS)\
                .all()
            conversation_history = list(reversed([msg.to_dict() for msg in messages]))
            
            # Save user question to conversation
            user_msg = Message(
                conversation_id=payload.conversation_id,
                role="user",
                content=payload.question
            )
            db.add(user_msg)
            db.commit()
    
    logger.info("Query: %s (tenant=%s, request_id=%s, mode=%s, top_k=%d, conversation_id=%s, history_len=%d, doc_ids=%s)", 
                payload.question, tenant_id, request_id, mode, top_k, payload.conversation_id, len(conversation_history), payload.doc_ids)
    
    # Query includes: embedding, retrieval, filtering, prompt building, LLM generation
    query_start = time.time()
    answer_gen, sources = await answer_question(tenant_id, payload.question, top_k, mode=mode, conversation_history=conversation_history, doc_ids=payload.doc_ids)
    
    # Stream the answer tokens back to client
    async def stream_response():
        first_token = True
        token_count = 0
        full_answer = ""  # Collect full answer for saving to conversation
        logger.info("Starting to stream response for request_id=%s", request_id)
        
        async for chunk in answer_gen:
            if first_token:
                log_timing("time_to_first_token", time.time() - query_start, tenant_id, question_length=len(payload.question))
                logger.info("First token received for request_id=%s", request_id)
                first_token = False
            token_count += 1
            full_answer += chunk  # Collect for saving
            # Send each token as JSON + newline
            token_json = json.dumps({"token": chunk}) + "\n"
            logger.debug("Yielding token %d (len=%d) for request_id=%s", token_count, len(chunk), request_id)
            yield token_json
        
        logger.info("Finished streaming %d tokens for request_id=%s", token_count, request_id)
        log_timing("query_complete", time.time() - query_start, tenant_id, 
                   question_length=len(payload.question), 
                   num_sources=len(sources),
                   tokens_generated=token_count)
        
        # Save assistant response to conversation
        if conversation and db:
            try:
                assistant_msg = Message(
                    conversation_id=payload.conversation_id,
                    role="assistant",
                    content=full_answer,
                    sources=json.dumps(sources) if sources else None
                )
                db.add(assistant_msg)
                conversation.updated_at = func.now()
                db.commit()
                logger.info("Saved assistant response to conversation %d", payload.conversation_id)
            except Exception as e:
                logger.error("Failed to save assistant response: %s", e)
                # Don't fail the request if saving fails
        
        # Final message with sources
        sources_json = json.dumps({"sources": sources}) + "\n"
        logger.info("Sending %d sources for request_id=%s: %s", len(sources), request_id, sources)
        yield sources_json
    
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
    tenant_id = current_user["tenant_id"]
    documents = []
    
    # Try to get documents from database first
    if db:
        try:
            db_docs = db.query(Document).filter(Document.tenant_id == tenant_id).order_by(Document.created_at.desc()).all()
            documents.extend([
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                    "error_message": doc.error_message
                }
                for doc in db_docs
            ])
        except Exception as e:
            logger.warning("Failed to list documents from DB: %s", e)
    
    # Also get documents from ChromaDB (for documents without DB records)
    try:
        from app.services.rag_service import rag_service
        chroma_docs = rag_service.get_indexed_documents(tenant_id)
        
        # Add ChromaDB documents that aren't already in the list
        db_filenames = {doc["filename"] for doc in documents}
        for chroma_doc in chroma_docs:
            if chroma_doc["filename"] not in db_filenames:
                documents.append(chroma_doc)
    except Exception as e:
        logger.debug("Failed to get documents from ChromaDB: %s", e)
    
    # Sort by creation time (newest first)
    documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {"documents": documents}


@app.get("/api/documents/{doc_id}/status")
async def get_document_status(
    doc_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint: get status of a specific document."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.tenant_id == tenant_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return doc.to_dict()


@app.post("/api/documents/{doc_id}/reindex")
async def reindex_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint: reindex a specific document."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    tenant_id = current_user["tenant_id"]
    
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.tenant_id == tenant_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Update status to pending
    doc.status = "pending"
    doc.error_message = None
    db.commit()
    
    # Schedule background reindexing
    background_tasks.add_task(
        process_document_background,
        doc.id,
        tenant_id,
        doc.file_path,
        doc.filename
    )
    
    logger.info(f"Scheduled reindexing for document {doc_id}: {doc.filename}")
    
    return {
        "status": "ok",
        "message": f"Reindexing started for {doc.filename}",
        "document": doc.to_dict()
    }
