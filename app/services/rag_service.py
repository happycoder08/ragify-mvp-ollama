from typing import List, Tuple, AsyncGenerator, Dict, Any
import json
import httpx
import chromadb
from chromadb.config import Settings
from functools import lru_cache

from ..config import VECTOR_DIR
import os

OLLAMA_BASE_URL = "http://localhost:11434"
# timeout (seconds) for requests to Ollama to avoid hanging the app
# configurable via env for slow model cold-starts
REQUEST_TIMEOUT = int(os.getenv("RAGIFY_OLLAMA_TIMEOUT", "300"))

import logging
logger = logging.getLogger(__name__)

# Initialise Chroma client and collection
# Lazy-initialized Chroma client/collection to avoid blocking imports or
# initialization when running in mock mode or when Chroma isn't available.
chroma_client = None
collection = None

# Multi-tenant support: maintain separate collections per tenant
_tenant_collections: Dict[str, Any] = {}

# Persistent async HTTP client for connection pooling
_http_client: httpx.AsyncClient = None

# Embedding cache for frequently queried questions (LRU, max 128 entries)
_embedding_cache: Dict[str, List[float]] = {}


def is_mock_mode() -> bool:
    return os.getenv("RAGIFY_MOCK", "0") == "1"


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create persistent async HTTP client for pooled connections."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _http_client


def _get_collection(tenant_id: str = "default"):
    """Ensure Chroma client/collection are initialized and return tenant-specific collection."""
    global chroma_client, _tenant_collections
    
    if chroma_client is None:
        chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=VECTOR_DIR)
        )
    
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

    client = await _get_http_client()
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
    embeddings = await asyncio.gather(*[embed_one(i, t) for i, t in enumerate(texts)])
    
    # Cache single-text embeddings for query optimization
    if len(texts) == 1:
        _embedding_cache[texts[0]] = embeddings[0]
        
    return embeddings


async def add_documents(tenant_id: str, chunks: List[str], source_filename: str) -> int:
    """
    Add chunks for a given source file into the tenant-specific Chroma collection.
    Returns number of chunks indexed.
    """
    if not chunks:
        return 0

    # If mock mode is enabled, skip actual Chroma operations to avoid external dependencies
    if is_mock_mode():
        logger.info("MOCK_MODE: skipping Chroma indexing for %s (tenant=%s) — returning %d chunks", source_filename, tenant_id, len(chunks))
        return len(chunks)

    logger.info("Indexing %d chunks from %s for tenant %s", len(chunks), source_filename, tenant_id)
    embeddings = await embed_texts(chunks)
    collection = _get_collection(tenant_id)
    
    base_id = source_filename.replace(" ", "_")
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source_file": source_filename, "chunk": i} for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    
    global chroma_client
    if chroma_client:
        chroma_client.persist()
    
    return len(chunks)


async def query_collection(tenant_id: str, question: str, top_k: int = 4) -> Tuple[AsyncGenerator[str, None], List[str]]:
    """
    Perform a similarity search in the tenant-specific collection and answer the question using retrieved context.
    Returns (answer_generator, list_of_source_files) where answer_generator yields tokens.
    """
    logger.info("Query received from tenant %s: %s", tenant_id, question)

    # In mock mode, skip Chroma and Ollama chat and return deterministic answer.
    if is_mock_mode():
        async def mock_gen():
            yield "(mocked) This is a canned answer used for local UI testing."
        return mock_gen(), []

    # embed question
    q_embedding = (await embed_texts([question]))[0]

    results = _get_collection(tenant_id).query(
        query_embeddings=[q_embedding],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        async def not_found_gen():
            yield "I could not find anything relevant in the indexed documents."
        return not_found_gen(), []

    context_pieces: List[str] = []
    sources: List[str] = []
    for doc, meta in zip(docs, metas):
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

    context = "\n\n".join(context_pieces)
    answer_gen = _call_chat_model(question, context)
    return answer_gen, dedup_sources


async def _call_chat_model(question: str, context: str) -> AsyncGenerator[str, None]:
    """
    Call a local Ollama chat model (llama3) with the retrieved context.
    Yields answer tokens as they arrive for streaming.
    Uses persistent async client for connection pooling.
    Run `ollama pull llama3` first.
    """
    prompt = f"""
You are a business assistant that answers questions STRICTLY using the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""

    # Streamed response from Ollama
    logger.info("Calling chat model (llama3) for question (len=%d) with context length %d mock=%s", len(question), len(context), is_mock_mode())
    
    # If mock mode, yield canned response immediately
    if is_mock_mode():
        yield "(mocked) This is a canned answer used for local UI testing."
        return

    client = await _get_http_client()
    
    try:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": True},
        ) as resp:
            resp.raise_for_status()
            
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
                    
    except httpx.HTTPError as e:
        logger.exception("Chat generation request failed: %s", e)
        raise RuntimeError(
            f"Failed to call Ollama chat API at {OLLAMA_BASE_URL}. "
            f"Ensure Ollama is running and the 'llama3' model is available. Original error: {e}"
        )


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
    global chroma_client, _tenant_collections
    
    logger.info("Resetting ChromaDB collection for tenant: %s", tenant_id)
    
    # Clear cache first
    clear_embedding_cache()
    
    # Delete tenant-specific collection
    if chroma_client is not None:
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
async def index_files(tenant_id: str, chunks: List[str], source_filename: str) -> int:
    """
    Convenience wrapper for add_documents with tenant support.
    """
    return await add_documents(tenant_id, chunks, source_filename)


async def answer_question(tenant_id: str, question: str, top_k: int = 4) -> Tuple[AsyncGenerator[str, None], List[str]]:
    """
    Convenience wrapper for query_collection with tenant support.
    """
    return await query_collection(tenant_id, question, top_k)

