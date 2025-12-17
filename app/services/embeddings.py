"""
Embedding interfaces and implementations for RAGify.

Provides a clean abstraction for text embedding with multiple backends:
- RealEmbedder: Uses HTTP client to call Ollama/OpenAI embedding APIs
- MockEmbedder: Deterministic hash-based vectors for testing (no network required)
"""

import hashlib
import logging
import time
from typing import List, Protocol
from abc import abstractmethod
import httpx

logger = logging.getLogger(__name__)

# Standard embedding dimensions for different models
OLLAMA_EMBEDDING_DIM = 768  # nomic-embed-text
OPENAI_EMBEDDING_DIM = 1536  # text-embedding-3-small
MOCK_EMBEDDING_DIM = 384  # Smaller for efficiency in tests


class Embedder(Protocol):
    """
    Protocol for text embedding providers.
    
    All embedders must implement embed_texts to convert text chunks
    into dense vector representations.
    """
    
    @abstractmethod
    async def embed_texts(self, texts: List[str], tenant_id: str) -> List[List[float]]:
        """
        Embed a list of text strings into dense vectors.
        
        Args:
            texts: List of text strings to embed
            tenant_id: Tenant identifier for logging/telemetry
            
        Returns:
            List of embedding vectors (one per input text)
        """
        ...


class MockEmbedder:
    """
    Deterministic mock embedder for testing without network dependencies.
    
    Generates pseudo-random but deterministic embedding vectors using SHA-256 hashing.
    Same text always produces the same vector, different texts produce different vectors.
    Vectors are normalized to unit length for realistic similarity behavior.
    
    Key properties:
    - No HTTP client required (no network calls)
    - Deterministic: hash(text) -> same vector every time
    - Normalized: ||vector|| = 1.0 (unit length)
    - Fixed dimension: 384 (configurable)
    """
    
    def __init__(self, dimension: int = MOCK_EMBEDDING_DIM):
        """
        Initialize mock embedder.
        
        Args:
            dimension: Embedding vector dimension (default: 384)
        """
        self.dimension = dimension
        logger.info("MockEmbedder initialized with dimension=%d", dimension)
    
    async def embed_texts(self, texts: List[str], tenant_id: str) -> List[List[float]]:
        """
        Generate deterministic embeddings using SHA-256 hashing.
        
        Algorithm:
        1. Hash text with SHA-256
        2. Use hash bytes to generate pseudo-random floats
        3. Normalize to unit length
        
        Args:
            texts: List of text strings to embed
            tenant_id: Tenant identifier (for logging)
            
        Returns:
            List of normalized embedding vectors
        """
        logger.debug("MockEmbedder generating %d embeddings for tenant=%s", len(texts), tenant_id)
        
        embeddings = []
        for text in texts:
            # Generate deterministic vector from text hash
            vector = self._hash_to_vector(text)
            embeddings.append(vector)
        
        return embeddings
    
    def _hash_to_vector(self, text: str) -> List[float]:
        """
        Convert text to deterministic normalized vector using SHA-256.
        
        Process:
        1. Hash text with SHA-256 (32 bytes)
        2. Extend hash by re-hashing to get enough bytes for dimension
        3. Convert bytes to floats in range [-1, 1]
        4. Normalize to unit length
        
        Args:
            text: Input text string
            
        Returns:
            Normalized embedding vector of length self.dimension
        """
        # Start with SHA-256 hash of text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()  # 32 bytes
        
        # Extend hash by repeatedly hashing to get enough bytes
        # We need 4 bytes per float, so dimension * 4 bytes total
        bytes_needed = self.dimension * 4
        extended_bytes = bytearray()
        
        while len(extended_bytes) < bytes_needed:
            extended_bytes.extend(hash_bytes)
            # Re-hash to generate more bytes
            hash_obj = hashlib.sha256(hash_bytes)
            hash_bytes = hash_obj.digest()
        
        # Convert bytes to floats in range [-1, 1]
        vector = []
        for i in range(self.dimension):
            # Take 4 bytes and convert to int
            byte_chunk = extended_bytes[i*4:(i+1)*4]
            int_val = int.from_bytes(byte_chunk, byteorder='big', signed=False)
            # Map to [-1, 1] range
            float_val = (int_val / (2**32 - 1)) * 2 - 1
            vector.append(float_val)
        
        # Normalize to unit length
        norm = sum(x*x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector


class RealEmbedder:
    """
    Production embedder using HTTP client to call embedding APIs.
    
    Supports:
    - Ollama (nomic-embed-text model)
    - OpenAI (text-embedding-3-small model)
    
    Requires initialized HTTP client for API calls.
    """
    
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        provider_type: str = "ollama",
        model: str = None,
        base_url: str = None
    ):
        """
        Initialize real embedder with HTTP client.
        
        Args:
            http_client: Async HTTP client for API calls
            provider_type: "ollama" or "openai"
            model: Embedding model name (defaults based on provider)
            base_url: API base URL (defaults based on provider)
        """
        self.http_client = http_client
        self.provider_type = provider_type.lower()
        
        # Set defaults based on provider
        if self.provider_type == "ollama":
            self.model = model or "nomic-embed-text"
            self.base_url = base_url or "http://localhost:11434"
            self.dimension = OLLAMA_EMBEDDING_DIM
        elif self.provider_type == "openai":
            self.model = model or "text-embedding-3-small"
            self.base_url = base_url or "https://api.openai.com/v1"
            self.dimension = OPENAI_EMBEDDING_DIM
        else:
            raise ValueError(f"Unsupported provider: {provider_type}. Use 'ollama' or 'openai'")
        
        logger.info(
            "RealEmbedder initialized: provider=%s, model=%s, base_url=%s, dimension=%d",
            self.provider_type, self.model, self.base_url, self.dimension
        )
    
    async def embed_texts(self, texts: List[str], tenant_id: str) -> List[List[float]]:
        """
        Get embeddings from real API (Ollama or OpenAI).
        
        Args:
            texts: List of text strings to embed
            tenant_id: Tenant identifier (for logging)
            
        Returns:
            List of embedding vectors from API
        """
        logger.info(
            "RealEmbedder generating %d embeddings via %s (tenant=%s)",
            len(texts), self.provider_type, tenant_id
        )
        
        t_start = time.time()
        
        if self.provider_type == "ollama":
            embeddings = await self._embed_ollama(texts)
        elif self.provider_type == "openai":
            embeddings = await self._embed_openai(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_type}")
        
        duration = time.time() - t_start
        avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 0
        
        logger.info(
            "Embedding complete: provider=%s, num_texts=%d, duration_ms=%.2f, avg_ms_per_text=%.2f",
            self.provider_type, len(texts), duration * 1000,
            (duration * 1000) / len(texts) if texts else 0
        )
        
        return embeddings
    
    async def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from Ollama API.
        
        Uses parallel requests for efficiency with connection pooling.
        """
        import asyncio
        
        async def embed_one(idx: int, text: str) -> List[float]:
            """Embed single text with error handling."""
            payload = {"model": self.model, "prompt": text}
            
            try:
                resp = await self.http_client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=30.0
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.exception("Embedding request failed for text %d: %s", idx, e)
                raise RuntimeError(
                    f"Failed to get embeddings from Ollama at {self.base_url}. "
                    f"Check that Ollama is running and model '{self.model}' is pulled. "
                    f"Original error: {e}"
                )
            
            data = resp.json()
            emb = data.get("embedding")
            if emb is None:
                raise RuntimeError("No embedding returned from Ollama.")
            
            return emb
        
        # Parallel embedding requests
        embeddings = await asyncio.gather(*[embed_one(i, t) for i, t in enumerate(texts)])
        return embeddings
    
    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from OpenAI API.
        
        Uses batch API call for efficiency.
        """
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            resp = await self.http_client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.exception("OpenAI embedding request failed: %s", e)
            raise RuntimeError(
                f"Failed to get embeddings from OpenAI. "
                f"Check API key and network connectivity. "
                f"Original error: {e}"
            )
        
        data = resp.json()
        
        # Extract embeddings from response
        embeddings = [item["embedding"] for item in data["data"]]
        
        return embeddings


def create_embedder(
    provider_type: str = "ollama",
    http_client: httpx.AsyncClient = None
) -> Embedder:
    """
    Factory function to create appropriate embedder based on provider type.
    
    Args:
        provider_type: "ollama", "openai", or "mock"
        http_client: HTTP client for real embedders (not needed for mock)
        
    Returns:
        Configured embedder instance
    """
    provider_type = provider_type.lower()
    
    if provider_type == "mock":
        return MockEmbedder()
    elif provider_type in ("ollama", "openai"):
        if http_client is None:
            raise ValueError(f"HTTP client required for {provider_type} embedder")
        return RealEmbedder(
            http_client=http_client,
            provider_type=provider_type
        )
    else:
        raise ValueError(
            f"Unknown embedder provider: {provider_type}. "
            f"Supported: 'ollama', 'openai', 'mock'"
        )
