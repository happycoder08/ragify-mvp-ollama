"""
Reranker Provider abstraction for RAGify.
Supports multiple reranking backends via a common interface.
Rerankers improve relevance by reordering retrieved chunks based on query-document similarity.
"""

from typing import List, Dict, Any, Protocol, Tuple
import os
import logging
import httpx
import time

logger = logging.getLogger(__name__)


class RerankerProvider(Protocol):
    """Protocol for reranker providers that reorder documents by relevance."""
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
        metadata: List[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Number of top documents to return (None = return all)
            metadata: Optional metadata for each document
            
        Returns:
            Tuple of (indices, scores) where:
            - indices: List of original document indices in ranked order
            - scores: List of relevance scores (higher = more relevant)
        """
        ...


class NoneReranker:
    """No-op reranker that preserves original order."""
    
    def __init__(self):
        logger.info("Initialized NoneReranker (pass-through, no reranking)")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
        metadata: List[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float]]:
        """Return documents in original order with dummy scores."""
        indices = list(range(len(documents)))
        # Dummy scores: descending from 1.0 to simulate original ranking
        scores = [1.0 - (i * 0.01) for i in range(len(documents))]
        
        if top_n is not None:
            indices = indices[:top_n]
            scores = scores[:top_n]
        
        logger.debug(f"NoneReranker: Preserved {len(indices)} documents in original order")
        return indices, scores


class JinaReranker:
    """Jina AI reranker using jina-reranker-v2-base-multilingual model."""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model = model or os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
        self.base_url = base_url or os.getenv("JINA_BASE_URL", "https://api.jina.ai/v1/rerank")
        
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required for JinaReranker")
        
        logger.info(f"Initialized JinaReranker (model={self.model})")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
        metadata: List[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float]]:
        """Rerank documents using Jina AI API."""
        if not documents:
            return [], []
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents)
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                
            duration_ms = (time.time() - start_time) * 1000
            
            result = response.json()
            
            # Jina returns: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            results = result.get("results", [])
            
            indices = [r["index"] for r in results]
            scores = [r["relevance_score"] for r in results]
            
            logger.info(f"JinaReranker: Reranked {len(documents)} docs → {len(indices)} results in {duration_ms:.2f}ms")
            
            return indices, scores
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Jina API error: {e.response.status_code} - {e.response.text}")
            # Fallback to original order on error
            indices = list(range(min(top_n or len(documents), len(documents))))
            scores = [1.0] * len(indices)
            return indices, scores
            
        except Exception as e:
            logger.exception(f"JinaReranker failed: {e}")
            # Fallback to original order on error
            indices = list(range(min(top_n or len(documents), len(documents))))
            scores = [1.0] * len(indices)
            return indices, scores


class CohereReranker:
    """Cohere reranker using rerank-english-v3.0 or rerank-multilingual-v3.0."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model or os.getenv("COHERE_RERANKER_MODEL", "rerank-english-v3.0")
        self.base_url = "https://api.cohere.ai/v1/rerank"
        
        if not self.api_key:
            raise ValueError("COHERE_API_KEY environment variable is required for CohereReranker")
        
        logger.info(f"Initialized CohereReranker (model={self.model})")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
        metadata: List[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float]]:
        """Rerank documents using Cohere API."""
        if not documents:
            return [], []
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents),
            "return_documents": False  # We only need indices and scores
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                
            duration_ms = (time.time() - start_time) * 1000
            
            result = response.json()
            
            # Cohere returns: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            results = result.get("results", [])
            
            indices = [r["index"] for r in results]
            scores = [r["relevance_score"] for r in results]
            
            logger.info(f"CohereReranker: Reranked {len(documents)} docs → {len(indices)} results in {duration_ms:.2f}ms")
            
            return indices, scores
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Cohere API error: {e.response.status_code} - {e.response.text}")
            # Fallback to original order on error
            indices = list(range(min(top_n or len(documents), len(documents))))
            scores = [1.0] * len(indices)
            return indices, scores
            
        except Exception as e:
            logger.exception(f"CohereReranker failed: {e}")
            # Fallback to original order on error
            indices = list(range(min(top_n or len(documents), len(documents))))
            scores = [1.0] * len(indices)
            return indices, scores


def create_reranker_provider() -> RerankerProvider:
    """
    Factory function to create the appropriate reranker provider based on configuration.
    
    Environment variables:
        RERANKER_PROVIDER: "none" (default), "jina", or "cohere"
        JINA_API_KEY: Jina AI API key (required for jina provider)
        JINA_RERANKER_MODEL: Jina model (default: jina-reranker-v2-base-multilingual)
        COHERE_API_KEY: Cohere API key (required for cohere provider)
        COHERE_RERANKER_MODEL: Cohere model (default: rerank-english-v3.0)
    
    Returns:
        Configured reranker provider instance
    """
    from app.config import RERANKER_PROVIDER
    
    provider_type = os.getenv("RERANKER_PROVIDER", RERANKER_PROVIDER).lower()
    
    if provider_type == "jina":
        return JinaReranker()
    elif provider_type == "cohere":
        return CohereReranker()
    elif provider_type == "none":
        return NoneReranker()
    else:
        logger.warning(f"Unknown RERANKER_PROVIDER: {provider_type}. Falling back to NoneReranker")
        return NoneReranker()
