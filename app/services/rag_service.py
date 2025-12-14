from typing import List, Tuple, AsyncGenerator, Dict, Any
import json
import time
import os
import logging
import httpx

from . import clients
from .llm_providers import create_llm_provider, LLMProvider
from .reranker_providers import create_reranker_provider, RerankerProvider
from app.guardrails import get_guardrail_config
from app.config import (
    REQUEST_TIMEOUT,
    SIMILARITY_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_TOKENS_FAST,
    MAX_TOKENS_FULL,
    TOP_K_FAST,
    TOP_K_FULL,
    EMBEDDING_MODEL,
    ENABLE_TIMING_LOGS,
    ENABLE_RERANKING,
    RERANKER_TOP_N,
    CONTEXT_BUDGET_CHARS,
)

OLLAMA_BASE_URL = "http://localhost:11434"

logger = logging.getLogger(__name__)

# Global LLM provider instance (initialized on first use)
_llm_provider: LLMProvider = None

# Global reranker provider instance (initialized on first use)
_reranker_provider: RerankerProvider = None

def log_timing_rag(event: str, duration: float, tenant_id: str, **extra):
    """Log timing events with structured JSON (RAG service)."""
    if not ENABLE_TIMING_LOGS:
        return
    log_data = {
        "event": event,
        "duration_ms": round(duration * 1000, 2),
        "tenant_id": tenant_id,
        **extra
    }
    logger.info(json.dumps(log_data))


def _lexical_overlap_score(query: str, doc: str) -> float:
    """
    Compute lexical overlap score between query and document.
    Returns a score between 0 and 1 based on token intersection.
    
    Args:
        query: User's question
        doc: Document chunk text
        
    Returns:
        Score from 0 (no overlap) to 1 (complete overlap)
    """
    # Simple tokenization: lowercase, split on whitespace and punctuation
    def tokenize(text: str) -> set:
        # Remove punctuation and convert to lowercase
        cleaned = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
        # Split and filter empty strings
        tokens = set(t for t in cleaned.split() if len(t) > 2)  # Ignore 1-2 char tokens
        return tokens
    
    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    # Jaccard similarity: intersection / union
    intersection = len(query_tokens & doc_tokens)
    union = len(query_tokens | doc_tokens)
    
    return intersection / union if union > 0 else 0.0


def _hybrid_rerank_score(query: str, doc: str, vector_distance: float) -> float:
    """
    Combine lexical overlap and vector distance for hybrid scoring.
    
    Args:
        query: User's question
        doc: Document chunk text
        vector_distance: ChromaDB distance (lower = better)
        
    Returns:
        Combined score (higher = better)
    """
    # Lexical score (0-1, higher is better)
    lexical_score = _lexical_overlap_score(query, doc)
    
    # Normalize vector distance to 0-1 range (invert so higher is better)
    # Typical distances: 0-500, with good matches < 400
    # Normalize and invert: 1 - (dist / 500)
    normalized_distance = min(vector_distance / 500.0, 1.0)
    vector_score = 1.0 - normalized_distance
    
    # Combine scores: 60% semantic (vector), 40% lexical
    combined_score = 0.6 * vector_score + 0.4 * lexical_score
    
    return combined_score


# Multi-tenant support: maintain separate collections per tenant
_tenant_collections: Dict[str, Any] = {}

# Embedding cache for frequently queried questions (LRU, max cache)
_embedding_cache: Dict[str, List[float]] = {}


def is_mock_mode() -> bool:
    return os.getenv("RAGIFY_MOCK", "0") == "1"


def _get_llm_provider() -> LLMProvider:
    """Get or create the global LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        http_client = clients.get_http_client()
        _llm_provider = create_llm_provider(http_client=http_client)
    return _llm_provider


def _get_reranker_provider() -> RerankerProvider:
    """Get or create the global reranker provider instance."""
    global _reranker_provider
    if _reranker_provider is None:
        _reranker_provider = create_reranker_provider()
    return _reranker_provider


def _get_collection(tenant_id: str = "default"):
    """Get tenant-specific collection from centralized ChromaDB client."""
    chroma_client = clients.get_chroma_client()
    
    # Get or create tenant-specific collection
    if tenant_id not in _tenant_collections:
        collection_name = f"documents_{tenant_id}"
        _tenant_collections[tenant_id] = chroma_client.get_or_create_collection(collection_name)
        logger.info(f"Initialized collection for tenant: {tenant_id}")
    
    return _tenant_collections[tenant_id]


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using a local Ollama embedding model.
    Model: nomic-embed-text (run `ollama pull nomic-embed-text` first).
    Uses persistent async client and parallel requests with connection pooling.
    
    For single queries, uses embedding cache to avoid redundant requests.
    """
    logger.info("Embedding %d texts via Ollama (timeout=%ds) mock=%s", len(texts), REQUEST_TIMEOUT, is_mock_mode())
    # If mock mode is enabled, return deterministic small vectors to avoid Ollama.
    if is_mock_mode():
        emb = [0.0] * 8
        return [list(emb) for _ in texts]

    client = clients.get_http_client()
    embeddings: List[List[float]] = []
    
    # Try to get cached embeddings for single-text queries
    if len(texts) == 1:
        text = texts[0]
        if text in _embedding_cache:
            logger.debug("Cache hit for embedding: %s...", text[:50])
            return [_embedding_cache[text]]
    
    # Parallel embedding requests using persistent client (connection pooled)
    async def embed_one(idx: int, text: str) -> List[float]:
        """Embed a single text with error handling."""
        payload = {"model": "nomic-embed-text", "prompt": text}
        logger.debug("Requesting embedding for text %d (len=%d)", idx, len(text))
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.exception("Embedding request failed for text %d: %s", idx, e)
            raise RuntimeError(
                f"Failed to get embeddings from Ollama at {OLLAMA_BASE_URL}. "
                f"Check that Ollama is running and the model 'nomic-embed-text' is pulled. Original error: {e}"
            )
        data = resp.json()
        emb = data.get("embedding")
        if emb is None:
            raise RuntimeError("No embedding returned from Ollama.")
        logger.debug("Received embedding for text %d (len=%d)", idx, len(emb))
        return emb
    
    # Parallelize all embedding calls using persistent client connection pool
    import asyncio
    t_start = time.time()
    embeddings = await asyncio.gather(*[embed_one(i, t) for i, t in enumerate(texts)])
    total_time = time.time() - t_start
    
    # Log timing with all metrics in one JSON object
    avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 0
    log_timing_rag("embedding", total_time, "system", 
                   num_texts=len(texts), 
                   avg_length=int(avg_text_length),
                   total_ms=round(total_time * 1000, 2),
                   avg_ms_per_chunk=round((total_time * 1000) / len(texts), 2) if texts else 0)
    
    # Cache single-text embeddings for query optimization
    if len(texts) == 1:
        _embedding_cache[texts[0]] = embeddings[0]
        
    return embeddings


def get_indexed_documents(tenant_id: str) -> List[Dict[str, Any]]:
    """
    Get list of unique documents indexed in ChromaDB for a tenant.
    Returns list of documents with metadata.
    
    Args:
        tenant_id: Tenant identifier
    
    Returns:
        List of document metadata dicts with filename, created_at, status
    """
    if is_mock_mode():
        logger.info("MOCK_MODE: returning empty document list for tenant=%s", tenant_id)
        return []
    
    try:
        # Check if ChromaDB client is available
        try:
            chroma_client = clients.get_chroma_client()
        except RuntimeError:
            logger.debug("ChromaDB client not initialized, skipping document retrieval")
            return []
        
        collection = _get_collection(tenant_id)
        
        # Get all documents from the collection with a timeout
        all_items = collection.get()
        
        if not all_items or not all_items.get("metadatas"):
            return []
        
        # Extract unique filenames with their metadata
        seen_files = {}
        for metadata in all_items["metadatas"]:
            # Support both old format (source_file) and new format (filename)
            filename = metadata.get("filename") or metadata.get("source_file", "unknown")
            if filename not in seen_files:
                # Create document entry from metadata
                seen_files[filename] = {
                    "id": metadata.get("doc_id", -1),
                    "filename": filename,
                    "status": "indexed",  # All docs in ChromaDB are indexed
                    "created_at": metadata.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    "updated_at": metadata.get("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    "error_message": None
                }
        
        return list(seen_files.values())
        
    except Exception as e:
        logger.warning("Failed to get indexed documents from ChromaDB: %s", e, exc_info=True)
        return []


async def add_documents(tenant_id: str, chunks: List[str], source_filename: str, doc_id: int = -1) -> int:
    """
    Add chunks for a given source file into the tenant-specific Chroma collection.
    Returns number of chunks indexed.
    
    Args:
        tenant_id: Tenant identifier
        chunks: List of text chunks
        source_filename: Original filename
        doc_id: Database document ID for filtering (optional)
    """
    if not chunks:
        return 0

    # If mock mode is enabled, skip actual Chroma operations to avoid external dependencies
    if is_mock_mode():
        logger.info("MOCK_MODE: skipping Chroma indexing for %s (tenant=%s) — returning %d chunks", source_filename, tenant_id, len(chunks))
        return len(chunks)

    logger.info("Indexing %d chunks from %s for tenant %s (doc_id=%s)", len(chunks), source_filename, tenant_id, doc_id)
    
    # Embed all chunks with timing
    t_embed_start = time.time()
    embeddings = await embed_texts(chunks)
    embed_duration = time.time() - t_embed_start
    # Detailed embedding timing already logged in embed_texts
    
    collection = _get_collection(tenant_id)
    
    base_id = source_filename.replace(" ", "_")
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    # Store doc_id and filename metadata for filtering
    metadatas = [
        {
            "source_file": source_filename, 
            "chunk": i,
            "doc_id": doc_id if doc_id is not None else -1,
            "filename": source_filename
        } 
        for i in range(len(chunks))
    ]
    
    # Chroma upsert with timing
    t_upsert = time.time()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    upsert_duration = time.time() - t_upsert
    log_timing_rag("chroma_upsert", upsert_duration, tenant_id, 
                   num_chunks=len(chunks),
                   upsert_ms=round(upsert_duration * 1000, 2),
                   source_file=source_filename)
    
    # Persist ChromaDB
    chroma_client = clients.get_chroma_client()
    chroma_client.persist()
    
    return len(chunks)


async def query_collection(
    tenant_id: str, 
    question: str, 
    top_k: int = 4, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    doc_ids: List[int] = None
) -> Tuple[AsyncGenerator[str, None], List[str]]:
    """
    Perform a similarity search in the tenant-specific collection and answer the question using retrieved context.
    Returns (answer_generator, list_of_source_files) where answer_generator yields tokens.
    
    Args:
        tenant_id: Tenant identifier
        question: User's question
        top_k: Number of chunks to retrieve
        mode: "fast" (concise, max_tokens=50) or "full" (detailed, no token limit)
        conversation_history: Optional list of previous messages for context
        doc_ids: Optional list of document IDs to filter retrieval (document-scoped search)
    """
    logger.info("Query received from tenant %s: %s (history_len=%d, doc_ids=%s)", 
                tenant_id, question, len(conversation_history) if conversation_history else 0, doc_ids)

    # In mock mode, skip Chroma and Ollama chat and return deterministic answer.
    if is_mock_mode():
        async def mock_gen():
            yield "(mocked) This is a canned answer used for local UI testing."
        return mock_gen(), []

    # Check if collection is empty
    collection = _get_collection(tenant_id)
    if collection.count() == 0:
        async def empty_gen():
            yield "I don't have enough information in the provided documents to answer that question."
        return empty_gen(), []

    # embed question
    t_embed = time.time()
    q_embedding = (await embed_texts([question]))[0]
    # Embedding timing already logged in embed_texts

    t_retrieval = time.time()
    # Build metadata filter if doc_ids provided
    # ChromaDB doesn't support $in operator, so we need to use $or with multiple $eq
    where_filter = None
    if doc_ids:
        if len(doc_ids) == 1:
            # Single doc_id: use simple $eq
            where_filter = {"doc_id": {"$eq": doc_ids[0]}}
        else:
            # Multiple doc_ids: use $or with multiple $eq conditions
            where_filter = {"$or": [{"doc_id": {"$eq": doc_id}} for doc_id in doc_ids]}
        logger.info("Applying doc_ids filter: %s", where_filter)
    
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        where=where_filter,
    )
    log_timing_rag("chroma_retrieval", time.time() - t_retrieval, tenant_id, top_k=top_k, doc_filter=bool(doc_ids))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        async def not_found_gen():
            yield "I could not find anything relevant in the indexed documents."
        return not_found_gen(), []

    # Log distances for debugging
    logger.info("Retrieved %d chunks with distances: %s", len(distances), distances)

    # Filter out chunks with low similarity
    t_filter = time.time()
    # ChromaDB uses squared Euclidean distance by default (not cosine)
    # Lower distance = higher similarity. For squared euclidean, typical relevant results are < 500
    # Very relevant: 0-200, Moderately relevant: 200-350, Irrelevant: > 350
    # Threshold configured in config.py based on RAGIFY_MODE
    # NOTE: When doc_ids are specified (document-scoped search), skip threshold filtering
    # to allow hybrid reranking to pick the best matches
    if doc_ids:
        # Document-scoped query: use all retrieved chunks, let hybrid reranking filter
        filtered_results = [(doc, meta, dist) for doc, meta, dist in zip(docs, metas, distances)]
        logger.info("Document-scoped query: using all %d retrieved chunks, threshold filter skipped", len(filtered_results))
    else:
        # Global query: apply similarity threshold
        filtered_results = [
            (doc, meta, dist) for doc, meta, dist in zip(docs, metas, distances)
            if dist < SIMILARITY_THRESHOLD
        ]

    log_timing_rag("similarity_filtering", time.time() - t_filter, tenant_id, 
                   before=len(docs), after=len(filtered_results), threshold=SIMILARITY_THRESHOLD)
    logger.info("After filtering (threshold=%.2f): %d chunks remain", SIMILARITY_THRESHOLD, len(filtered_results))

    if not filtered_results:
        async def not_relevant_gen():
            yield "I could not find anything relevant in the indexed documents to answer that question."
        return not_relevant_gen(), []

    # Sort by distance (best first) and limit to best chunks
    # In demo mode: retrieve 15-20, filter by threshold, then take best 4-6
    # Apply free lexical reranking if external reranker is disabled
    if not ENABLE_RERANKING and len(filtered_results) > 1:
        t_rerank = time.time()
        logger.info("Applying free lexical+semantic hybrid reranking")
        
        # Compute hybrid scores for all filtered results
        scored_results = []
        for doc, meta, dist in filtered_results:
            hybrid_score = _hybrid_rerank_score(question, doc, dist)
            scored_results.append((doc, meta, dist, hybrid_score))
        
        # Sort by hybrid score (higher is better)
        scored_results.sort(key=lambda x: x[3], reverse=True)
        
        # Extract scores for logging
        hybrid_scores = [score for _, _, _, score in scored_results]
        
        rerank_duration = time.time() - t_rerank
        log_timing_rag("lexical_reranking", rerank_duration, tenant_id,
                      before=len(filtered_results), after=len(scored_results),
                      rerank_ms=round(rerank_duration * 1000, 2),
                      hybrid_scores=[round(s, 4) for s in hybrid_scores[:10]])  # Log top 10 scores
        
        logger.info("After lexical reranking: %d chunks, top scores: %s", 
                   len(scored_results), [round(s, 4) for s in hybrid_scores[:5]])
        
        # Update filtered_results with reranked order (drop hybrid score)
        filtered_results = [(doc, meta, dist) for doc, meta, dist, _ in scored_results]
    else:
        # Just sort by distance if not reranking
        filtered_results = sorted(filtered_results, key=lambda x: x[2])  # Sort by distance (lower = better)
    
    # Limit to best N chunks after sorting/reranking
    max_chunks = RERANKER_TOP_N if RERANKER_TOP_N else min(6, len(filtered_results))
    if len(filtered_results) > max_chunks:
        logger.info("Limiting from %d to top %d best chunks after scoring", 
                   len(filtered_results), max_chunks)
        filtered_results = filtered_results[:max_chunks]

    # External reranking (Jina, Cohere, etc.) if enabled
    if ENABLE_RERANKING and len(filtered_results) > 1:
        t_rerank = time.time()
        
        # Extract documents and metadata from filtered results
        rerank_docs = [doc for doc, meta, dist in filtered_results]
        rerank_metas = [meta for doc, meta, dist in filtered_results]
        rerank_dists = [dist for doc, meta, dist in filtered_results]
        
        # Get reranker and rerank
        reranker = _get_reranker_provider()
        top_n = RERANKER_TOP_N if RERANKER_TOP_N is not None else len(rerank_docs)
        
        try:
            indices, scores = reranker.rerank(
                query=question,
                documents=rerank_docs,
                top_n=top_n,
                metadata=rerank_metas
            )
            
            # Reorder filtered_results based on reranker indices
            reranked_results = [
                (rerank_docs[i], rerank_metas[i], rerank_dists[i], scores[idx])
                for idx, i in enumerate(indices)
            ]
            
            rerank_duration = time.time() - t_rerank
            reranked_doc_ids = [meta.get("source_file", "unknown") for _, meta, _, _ in reranked_results]
            
            log_timing_rag("reranking", rerank_duration, tenant_id,
                          before=len(filtered_results), after=len(reranked_results),
                          rerank_ms=round(rerank_duration * 1000, 2),
                          reranked_doc_ids=reranked_doc_ids,
                          rerank_scores=[round(s, 4) for s in scores])
            
            logger.info("After reranking: %d chunks (top_n=%s), scores: %s", 
                       len(reranked_results), top_n, [round(s, 4) for s in scores])
            
            # Use reranked results (now with 4-tuple including rerank score)
            filtered_results = [(doc, meta, dist) for doc, meta, dist, _ in reranked_results]
            
        except Exception as e:
            logger.exception(f"Reranking failed: {e}. Using original order.")
            # Continue with filtered_results as-is

    context_pieces: List[str] = []
    sources: List[str] = []
    for doc, meta, dist in filtered_results:
        src = meta.get("source_file", "unknown")
        context_pieces.append(f"[{src}] {doc}")
        sources.append(src)

    # dedupe sources while preserving order
    seen = set()
    dedup_sources: List[str] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            dedup_sources.append(s)

    t_prompt = time.time()
    context = "\n\n".join(context_pieces)
    
    # Apply context budget if configured
    context_char_count = len(context)
    if CONTEXT_BUDGET_CHARS and context_char_count > CONTEXT_BUDGET_CHARS:
        logger.info("Context exceeds budget (%d > %d chars), truncating", 
                   context_char_count, CONTEXT_BUDGET_CHARS)
        context = context[:CONTEXT_BUDGET_CHARS]
        context_char_count = len(context)
    
    log_timing_rag("prompt_building", time.time() - t_prompt, tenant_id, 
                   context_length=context_char_count,
                   context_char_count=context_char_count,
                   num_chunks=len(filtered_results),
                   budget_chars=CONTEXT_BUDGET_CHARS or "unlimited")
    
    answer_gen = _call_chat_model(question, context, tenant_id, mode=mode, conversation_history=conversation_history)
    return answer_gen, dedup_sources


async def _call_chat_model(
    question: str, 
    context: str, 
    tenant_id: str, 
    mode: str = "full",
    conversation_history: List[Dict] = None
) -> AsyncGenerator[str, None]:
    """
    Call the configured LLM provider with the retrieved context.
    Yields answer tokens as they arrive for streaming.
    Supports multiple providers via LLM_PROVIDER env var (ollama, openai).
    
    Args:
        question: User's question
        context: Retrieved context from documents
        tenant_id: Tenant identifier
        mode: "fast" (concise, limited tokens) or "full" (detailed, more tokens)
        conversation_history: Optional list of previous messages for context
    """
    # Build conversation history text
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role_prefix = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role_prefix}: {msg['content']}\n\n"
    
    # Mode-specific prompts and token limits (from config.py)
    if mode == "fast":
        prompt = f"""Answer briefly in 1-2 sentences.

{history_text if history_text else ""}Context:
{context}

Question: {question}

Answer:"""
        max_tokens = MAX_TOKENS_FAST
    else:
        prompt = f"""Answer the question based on the context below. Use the information provided to give a direct answer.

{history_text if history_text else ""}Context:
{context}

Question: {question}

Answer:"""
        max_tokens = MAX_TOKENS_FULL

    logger.info("Calling LLM for question (len=%d) with context length %d mode=%s max_tokens=%s mock=%s history_len=%d", 
                len(question), len(context), mode, max_tokens, is_mock_mode(), len(conversation_history) if conversation_history else 0)
    
    # If mock mode, yield canned response immediately
    if is_mock_mode():
        yield "(mocked) This is a canned answer used for local UI testing."
        return

    # Get LLM provider and guardrail config for timeout
    llm_provider = _get_llm_provider()
    guardrail_config = get_guardrail_config(tenant_id)
    llm_timeout = guardrail_config.llm_timeout_seconds
    
    t_llm = time.time()
    first_token_logged = False
    
    def on_first_token(duration: float):
        """Callback when first token arrives."""
        nonlocal first_token_logged
        if not first_token_logged:
            log_timing_rag("llm_first_token", duration, tenant_id, prompt_length=len(prompt))
            first_token_logged = True
    
    try:
        async for token in llm_provider.generate_stream(
            prompt, 
            tenant_id, 
            max_tokens=max_tokens, 
            on_first_token=on_first_token,
            timeout=llm_timeout
        ):
            yield token
        
        # Log completion timing
        log_timing_rag("llm_generation_complete", time.time() - t_llm, tenant_id)
        
    except Exception as e:
        logger.exception("LLM generation request failed: %s", e)
        raise


def clear_embedding_cache() -> None:
    """Clear the query embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


def reset_collection(tenant_id: str = "default") -> None:
    """
    Reset the vector store for a specific tenant by clearing all documents and reinitializing.
    Also clears the embedding cache.
    Useful for testing or starting fresh with new documents.
    """
    global _tenant_collections
    
    logger.info("Resetting ChromaDB collection for tenant: %s", tenant_id)
    
    # Clear cache first
    clear_embedding_cache()
    
    # Delete tenant-specific collection
    chroma_client = clients.get_chroma_client()
    collection_name = f"documents_{tenant_id}"
    try:
        chroma_client.delete_collection(collection_name)
        logger.info("Deleted collection: %s", collection_name)
    except Exception as e:
        logger.warning("Could not delete collection %s: %s", collection_name, e)
    
    # Remove from cache
    if tenant_id in _tenant_collections:
        del _tenant_collections[tenant_id]
    
    logger.info("Collection reset complete for tenant: %s", tenant_id)


# Wrapper functions for backward compatibility and convenience
async def index_files(tenant_id: str, chunks: List[str], source_filename: str, doc_id: int = None) -> int:
    """
    Convenience wrapper for add_documents with tenant support.
    """
    return await add_documents(tenant_id, chunks, source_filename, doc_id=doc_id)


async def answer_question(
    tenant_id: str, 
    question: str, 
    top_k: int = 4, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    doc_ids: List[int] = None
) -> Tuple[AsyncGenerator[str, None], List[str]]:
    """
    Convenience wrapper for query_collection with tenant support.
    
    Args:
        tenant_id: Tenant identifier
        question: User's question
        top_k: Number of chunks to retrieve
        mode: "fast" (concise, max_tokens=50) or "full" (detailed, no token limit)
        conversation_history: Optional list of previous messages for context
        doc_ids: Optional list of document IDs to filter retrieval (document-scoped search)
    """
    return await query_collection(tenant_id, question, top_k, mode=mode, conversation_history=conversation_history, doc_ids=doc_ids)

