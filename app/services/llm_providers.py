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


def create_llm_provider(http_client: httpx.AsyncClient = None) -> LLMProvider:
    """
    Factory function to create the appropriate LLM provider based on configuration.
    
    Uses centralized config from app.config (RAGIFY_MODE sets defaults).
    
    Environment variables (override config):
        LLM_PROVIDER: "ollama" or "openai" (overrides config default)
        LLM_MODEL: Model name (overrides config default)
        OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
        OPENAI_API_KEY: OpenAI API key (required for OpenAI provider)
        OPENAI_BASE_URL: OpenAI API base URL (default: https://api.openai.com/v1)
    
    Returns:
        Configured LLM provider instance
    """
    provider_type = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    
    if provider_type == "openai":
        return OpenAILLMProvider()
    elif provider_type == "ollama":
        return OllamaLLMProvider(http_client=http_client)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}. Supported: 'ollama', 'openai'")
