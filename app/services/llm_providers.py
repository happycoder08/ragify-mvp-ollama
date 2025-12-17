"""
LLM Provider abstraction for RAGify.
Supports multiple LLM backends via a common interface.
"""

from typing import AsyncGenerator, Protocol
import os
import json
import time
import logging
import httpx
import hashlib

from app.config import LLM_PROVIDER as DEFAULT_PROVIDER, LLM_MODEL as DEFAULT_MODEL

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers that support streaming text generation."""
    
    async def generate_stream(
        self, 
        prompt: str, 
        tenant_id: str,
        max_tokens: int = None,
        on_first_token: callable = None,
        timeout: int = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate text from prompt with streaming.
        
        Args:
            prompt: The full prompt to send to the LLM
            tenant_id: Tenant identifier for logging
            max_tokens: Maximum tokens to generate (None for no limit)
            on_first_token: Optional callback(duration_ms) called when first token arrives
            timeout: Request timeout in seconds (None for default)
            
        Yields:
            Text tokens as they arrive
        """
        ...


class OllamaLLMProvider:
    """Ollama local LLM provider."""
    
    def __init__(self, base_url: str = None, model: str = None, http_client: httpx.AsyncClient = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or DEFAULT_MODEL
        self.http_client = http_client
        logger.info(f"Initialized OllamaLLMProvider (url={self.base_url}, model={self.model})")
    
    async def generate_stream(
        self, 
        prompt: str, 
        tenant_id: str,
        max_tokens: int = None,
        on_first_token: callable = None,
        timeout: int = None
    ) -> AsyncGenerator[str, None]:
        """Generate text using Ollama streaming API."""
        logger.info("Calling Ollama model %s for tenant %s (prompt_len=%d, max_tokens=%s, timeout=%s)", 
                   self.model, tenant_id, len(prompt), max_tokens, timeout)
        
        t_start = time.time()
        first_token_sent = False
        token_count = 0
        
        # Use provided timeout or default to 300 seconds
        request_timeout = timeout or 300
        
        try:
            # Build request payload
            payload = {"model": self.model, "prompt": prompt, "stream": True}
            if max_tokens is not None:
                payload["options"] = {"num_predict": max_tokens}
            
            async with self.http_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=request_timeout,
            ) as resp:
                resp.raise_for_status()
                
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        
                        if token:
                            if not first_token_sent and on_first_token:
                                on_first_token(time.time() - t_start)
                                first_token_sent = True
                            token_count += 1
                            yield token
                            
                            # Stop if we've reached max_tokens
                            if max_tokens and token_count >= max_tokens:
                                break
                        
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Ollama response line: %s", line)
                        continue
                        
        except httpx.HTTPError as e:
            logger.exception("Ollama generation request failed: %s", e)
            raise RuntimeError(
                f"Failed to call Ollama at {self.base_url}. "
                f"Ensure Ollama is running and model '{self.model}' is available. Error: {e}"
            )


class OpenAILLMProvider:
    """OpenAI LLM provider (GPT-3.5, GPT-4, etc.)."""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        logger.info(f"Initialized OpenAILLMProvider (model={self.model})")
    
    async def generate_stream(
        self, 
        prompt: str, 
        tenant_id: str,
        max_tokens: int = None,
        on_first_token: callable = None
    ) -> AsyncGenerator[str, None]:
        """Generate text using OpenAI streaming API."""
        logger.info("Calling OpenAI model %s for tenant %s (prompt_len=%d, max_tokens=%s)", 
                   self.model, tenant_id, len(prompt), max_tokens)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        t_start = time.time()
        first_token_sent = False
        
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    resp.raise_for_status()
                    
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        
                        # OpenAI streaming format: "data: {json}\n\n"
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                        
                        if line == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(line)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            
                            if token:
                                if not first_token_sent and on_first_token:
                                    on_first_token(time.time() - t_start)
                                    first_token_sent = True
                                yield token
                                
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse OpenAI response line: %s", line)
                            continue
                            
            except httpx.HTTPError as e:
                logger.exception("OpenAI generation request failed: %s", e)
                raise RuntimeError(
                    f"Failed to call OpenAI API at {self.base_url}. "
                    f"Check API key and model '{self.model}' availability. Error: {e}"
                )


class MockLLMProvider:
    """
    Mock LLM provider for testing and development.
    
    Returns deterministic responses based on prompt hash.
    Can be configured to return ungrounded answers for validation testing.
    
    Enable via: LLM_PROVIDER=mock
    
    Environment variables:
        MOCK_UNGROUNDED: Set to "true" to return intentionally ungrounded answers
    """
    
    def __init__(self):
        self.ungrounded_mode = os.getenv("MOCK_UNGROUNDED", "false").lower() == "true"
        logger.info(f"Initialized MockLLMProvider (ungrounded_mode={self.ungrounded_mode})")
        
        # Predefined responses based on question keywords
        self.grounded_responses = {
            "vacation": "The vacation policy allows 15 days per year.",
            "sick": "The document does not specify this.",
            "onboarding": "New employees should arrive at 8:00 AM on the 3rd floor.",
            "arrive": "New employees should arrive at 8:00 AM on the 3rd floor.",
            "benefits": "The company provides health insurance and 401k matching.",
            "policy": "Please refer to the employee handbook for policy details.",
        }
        
        # Ungrounded responses (for testing validation rejection)
        self.ungrounded_responses = {
            "vacation": "Employees receive 30 days of vacation per year.",  # Hallucinated number
            "sick": "Unlimited sick leave is provided.",  # Hallucinated policy
            "onboarding": "Arrive at 9:30 AM on the 5th floor.",  # Wrong time/floor
            "benefits": "Free lunch and gym membership included.",  # Not in docs
            "policy": "All policies are available on the intranet.",  # Hallucinated location
        }
    
    async def generate_stream(
        self,
        prompt: str,
        tenant_id: str,
        max_tokens: int = None,
        on_first_token: callable = None,
        timeout: int = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate deterministic mock response based on prompt content.
        
        Simulates token-by-token streaming for realistic behavior.
        """
        t_start = time.time()
        
        # Trigger first token callback immediately
        if on_first_token:
            on_first_token(0.001)
        
        # Determine response based on prompt keywords
        prompt_lower = prompt.lower()
        
        # Select response set based on mode
        response_set = self.ungrounded_responses if self.ungrounded_mode else self.grounded_responses
        
        # Find matching response
        response = None
        for keyword, text in response_set.items():
            if keyword in prompt_lower:
                response = text
                break
        
        # Default response if no keyword match
        if response is None:
            response = "The document does not specify this."
        
        # Generate deterministic variation based on prompt hash (for uniqueness)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
        
        # Log the mock generation
        logger.info(
            "MockLLM generating response for tenant %s (prompt_hash=%s, ungrounded=%s, response_len=%d)",
            tenant_id, prompt_hash, self.ungrounded_mode, len(response)
        )
        
        # Stream response character by character (simulating token streaming)
        for char in response:
            yield char
            # Small delay to simulate realistic streaming (optional, can be removed for speed)
            # await asyncio.sleep(0.001)


def create_llm_provider(http_client: httpx.AsyncClient = None) -> LLMProvider:
    """
    Factory function to create the appropriate LLM provider based on configuration.
    
    Uses centralized config from app.config (RAGIFY_MODE sets defaults).
    
    Environment variables (override config):
        LLM_PROVIDER: "ollama", "openai", or "mock" (overrides config default)
        LLM_MODEL: Model name (overrides config default)
        OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
        OPENAI_API_KEY: OpenAI API key (required for OpenAI provider)
        OPENAI_BASE_URL: OpenAI API base URL (default: https://api.openai.com/v1)
        MOCK_UNGROUNDED: Set to "true" for mock provider to return ungrounded answers
    
    Returns:
        Configured LLM provider instance
    """
    provider_type = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    
    if provider_type == "mock":
        return MockLLMProvider()
    elif provider_type == "openai":
        return OpenAILLMProvider()
    elif provider_type == "ollama":
        return OllamaLLMProvider(http_client=http_client)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}. Supported: 'ollama', 'openai', 'mock'")


def create_embedding_provider(http_client: httpx.AsyncClient = None):
    """
    Factory function to create the appropriate embedding provider.
    
    For now, returns the same provider as LLM (providers handle both).
    In future, could support separate embedding-only providers.
    
    Environment variables:
        EMBEDDING_PROVIDER: "ollama", "openai", or "mock" (defaults to LLM_PROVIDER)
        EMBEDDING_MODEL: Model name for embeddings (provider-specific defaults)
    
    Returns:
        Configured embedding provider instance
    """
    # Use EMBEDDING_PROVIDER if set, otherwise fall back to LLM_PROVIDER
    provider_type = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    
    if provider_type == "mock":
        return MockLLMProvider()  # Mock provider handles embeddings too
    elif provider_type == "openai":
        return OpenAILLMProvider()
    elif provider_type == "ollama":
        return OllamaLLMProvider(http_client=http_client)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider_type}. Supported: 'ollama', 'openai', 'mock'")
