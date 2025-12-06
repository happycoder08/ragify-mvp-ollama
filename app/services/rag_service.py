from typing import List, Tuple, AsyncGenerator
import json
import httpx
import chromadb
from chromadb.config import Settings

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

# Persistent async HTTP client for connection pooling
_http_client: httpx.AsyncClient = None


def is_mock_mode() -> bool:
    return os.getenv("RAGIFY_MOCK", "0") == "1"


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create persistent async HTTP client for pooled connections."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _http_client


def _get_collection():
    """Ensure Chroma client/collection are initialized and return collection."""
    global chroma_client, collection
    if chroma_client is None or collection is None:
        chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=VECTOR_DIR)
        )
        collection = chroma_client.get_or_create_collection("documents")
    return collection


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using a local Ollama embedding model.
    Model: nomic-embed-text (run `ollama pull nomic-embed-text` first).
    Uses persistent async client for connection pooling.
    """
    logger.info("Embedding %d texts via Ollama (timeout=%ds) mock=%s", len(texts), REQUEST_TIMEOUT, is_mock_mode())
    # If mock mode is enabled, return deterministic small vectors to avoid Ollama.
    if is_mock_mode():
        emb = [0.0] * 8
        return [list(emb) for _ in texts]

    client = await _get_http_client()
    
    async def embed_one(idx: int, text: str) -> List[float]:
        """Embed a single text with error handling."""
        payload = {"model": "nomic-embed-text", "input": text}
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
    return embeddings


async def add_documents(chunks: List[str], source_filename: str) -> int:
    """
    Add chunks for a given source file into the Chroma collection.
    Returns number of chunks indexed.
    """
    if not chunks:
        return 0

    # If mock mode is enabled, skip actual Chroma operations to avoid external dependencies
    if is_mock_mode():
        logger.info("MOCK_MODE: skipping Chroma indexing for %s — returning %d chunks", source_filename, len(chunks))
        return len(chunks)

    # Lazy init Chroma client/collection here so the module import doesn't
    # attempt to open DB files when not needed.
    global chroma_client, collection
    if chroma_client is None or collection is None:
        chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=VECTOR_DIR)
        )
        collection = chroma_client.get_or_create_collection("documents")

    logger.info("Indexing %d chunks from %s", len(chunks), source_filename)
    embeddings = await embed_texts(chunks)
    base_id = source_filename.replace(" ", "_")
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source_file": source_filename, "chunk": i} for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    chroma_client.persist()
    return len(chunks)


async def query_collection(question: str, top_k: int = 4) -> Tuple[AsyncGenerator[str, None], List[str]]:
    """
    Perform a similarity search and answer the question using retrieved context.
    Returns (answer_generator, list_of_source_files) where answer_generator yields tokens.
    """
    logger.info("Query received: %s", question)

    # In mock mode, skip Chroma and Ollama chat and return deterministic answer.
    if is_mock_mode():
        async def mock_gen():
            yield "(mocked) This is a canned answer used for local UI testing."
        return mock_gen(), []

    # embed question
    q_embedding = (await embed_texts([question]))[0]

    results = _get_collection().query(
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


def reset_collection() -> None:
    """
    Reset the vector store by clearing all documents and reinitializing.
    Useful for testing or starting fresh with new documents.
    """
    global chroma_client, collection
    
    logger.info("Resetting ChromaDB collection...")
    
    # Close existing connections
    if chroma_client is not None:
        try:
            chroma_client.delete_collection("documents")
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.warning("Could not delete collection: %s", e)
        
        try:
            chroma_client = None
            collection = None
        except Exception as e:
            logger.warning("Error closing client: %s", e)
    
    # Delete persisted files
    import shutil
    if os.path.exists(VECTOR_DIR):
        try:
            shutil.rmtree(VECTOR_DIR)
            logger.info("Removed vector store directory: %s", VECTOR_DIR)
        except Exception as e:
            logger.warning("Could not remove vector store directory: %s", e)
            raise RuntimeError(f"Failed to clean vector store: {e}")
    
    # Reinitialize on next use via lazy loading
    logger.info("Collection reset complete. Will reinitialize on next index operation.")

