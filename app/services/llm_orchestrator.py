"""
LLM orchestration module.

Handles buffered streaming with validation, timing logs, and refusal logic.
Provides a clean abstraction for LLM interaction with post-generation validation.
"""

import time
import re
import logging
from typing import AsyncGenerator, Callable, Optional

from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


def log_timing_rag(event: str, duration: float, tenant_id: str, **extra):
    """
    Log timing information for RAG pipeline events.
    Import from rag_service to maintain consistent logging.
    """
    from .rag_service import log_timing_rag as _log_timing_rag
    _log_timing_rag(event, duration, tenant_id, **extra)


async def generate_answer_stream(
    prompt: str,
    tenant_id: str,
    provider: LLMProvider,
    max_tokens: int,
    timeout: float,
    validate_fn: Optional[Callable[[str, str], bool]] = None,
    evidence_text: str = "",
    refusal_text: str = "The document does not specify this.",
    request_id: Optional[str] = None,
    chunk_size: int = 75
) -> AsyncGenerator[str, None]:
    """
    Generate and stream an answer from LLM with optional validation.
    
    Buffered streaming mode (when validate_fn is provided):
    1. Collect all tokens from LLM into full answer
    2. Log first token and completion timing
    3. Validate full answer against evidence using validate_fn
    4. If validation fails, replace with refusal_text
    5. Stream the (validated or refused) answer in chunks
    
    Direct streaming mode (when validate_fn is None):
    1. Stream tokens directly as they arrive
    2. Log timing events
    
    Args:
        prompt: The full prompt to send to LLM
        tenant_id: Tenant identifier for logging/config
        provider: LLM provider instance to use for generation
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        validate_fn: Optional validation function(answer, evidence) -> bool
        evidence_text: Evidence text for validation (required if validate_fn provided)
        refusal_text: Text to return if validation fails
        request_id: Request ID for tracing/logging
        chunk_size: Size of chunks for simulated streaming (default: 75 chars)
    
    Yields:
        Answer text chunks
    
    Raises:
        Exception: If LLM generation fails
    """
    t_llm = time.time()
    first_token_logged = False
    
    def _strip_technical_markers(text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(r"\bCHUNK_ID\b\s*=?\s*\S*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bVALIDATION_FAILED\b|\bVALIDATION\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned

    def on_first_token(duration: float):
        """Callback when first token arrives."""
        nonlocal first_token_logged
        if not first_token_logged:
            log_timing_rag("llm_first_token", duration, tenant_id, prompt_length=len(prompt))
            first_token_logged = True
    
    try:
        if validate_fn is not None:
            # BUFFERED MODE: Collect all tokens, validate, then stream
            full_answer = ""
            async for token in provider.generate_stream(
                prompt,
                tenant_id,
                max_tokens=max_tokens,
                on_first_token=on_first_token,
                timeout=timeout
            ):
                full_answer += token
            
            # Log completion timing
            log_timing_rag("llm_generation_complete", time.time() - t_llm, tenant_id)
            
            # Validate answer against evidence
            t_validate = time.time()
            is_supported = validate_fn(full_answer, evidence_text)
            validation_duration = time.time() - t_validate
            log_timing_rag(
                "answer_validation",
                validation_duration,
                tenant_id,
                is_supported=is_supported,
                answer_length=len(full_answer),
                request_id=request_id or "no-request-id"
            )
            
            if not is_supported:
                logger.warning(
                    "[%s] Answer validation REJECTED. Replacing with refusal. Original: %s",
                    request_id or "no-request-id",
                    full_answer[:200]
                )
                full_answer = refusal_text
            else:
                logger.info("[%s] Answer validation PASSED", request_id or "no-request-id")

            full_answer = _strip_technical_markers(full_answer)

            # Stream the (validated) answer in chunks to simulate streaming
            for i in range(0, len(full_answer), chunk_size):
                chunk = full_answer[i:i+chunk_size]
                yield chunk
        else:
            # DIRECT MODE: Stream tokens as they arrive (original behavior)
            async for token in provider.generate_stream(
                prompt,
                tenant_id,
                max_tokens=max_tokens,
                on_first_token=on_first_token,
                timeout=timeout
            ):
                yield token
            
            # Log completion timing
            log_timing_rag("llm_generation_complete", time.time() - t_llm, tenant_id)
    
    except Exception as e:
        logger.exception("LLM generation request failed: %s", e)
        raise
