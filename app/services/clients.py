"""
Centralized client management for ChromaDB and HTTP connections.
Clients are initialized once on startup and reused across requests.
"""
import httpx
import chromadb
from chromadb.config import Settings
import logging
import os

from ..config import VECTOR_DIR

logger = logging.getLogger(__name__)

# Global client instances (initialized on startup)
chroma_client = None
http_client = None

# Timeout for Ollama requests
REQUEST_TIMEOUT = int(os.getenv("RAGIFY_OLLAMA_TIMEOUT", "300"))


def initialize_chroma_client():
    """Initialize ChromaDB client with persistent storage."""
    global chroma_client
    if chroma_client is None:
        logger.info("Initializing ChromaDB client (persistent storage: %s)", VECTOR_DIR)
        chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=VECTOR_DIR,
                anonymized_telemetry=False
            )
        )
        logger.info("ChromaDB client initialized successfully")
    return chroma_client


async def initialize_http_client():
    """Initialize async HTTP client for Ollama connections."""
    global http_client
    if http_client is None or http_client.is_closed:
        logger.info("Initializing async HTTP client (timeout=%ds)", REQUEST_TIMEOUT)
        http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        logger.info("HTTP client initialized successfully")
    return http_client


def get_chroma_client():
    """Get the initialized ChromaDB client."""
    if chroma_client is None:
        raise RuntimeError("ChromaDB client not initialized. Call initialize_chroma_client() first.")
    return chroma_client



def get_http_client():
    """Get the initialized async HTTP client."""
    if http_client is None or http_client.is_closed:
        raise RuntimeError("HTTP client not initialized. Call initialize_http_client() first.")
    return http_client


async def close_http_client():
    """Close the global async HTTP client and set to None."""
    global http_client
    if http_client and not http_client.is_closed:
        await http_client.aclose()
    http_client = None


async def shutdown_clients():
    """Cleanup and close all clients on shutdown."""
    global http_client, chroma_client

    if http_client and not http_client.is_closed:
        logger.info("Closing HTTP client...")
        await http_client.aclose()
        logger.info("HTTP client closed")
    http_client = None

    # ✅ In CI/test mode, don't persist (avoids logging on closed streams)
    is_ci_mode = (
        os.getenv("CI", "").lower() in ("true", "1", "yes") or
        os.getenv("APP_MODE", "").lower() == "ci"
    )

    if chroma_client:
        if not is_ci_mode:
            logger.info("Persisting ChromaDB...")
            try:
                chroma_client.persist()
                logger.info("ChromaDB persisted successfully")
            except Exception as e:
                logger.warning("Failed to persist ChromaDB: %s", e)
        chroma_client = None
