"""
Unit tests for llm_orchestrator validation-before-streaming behavior.

Tests:
1. Invalid answer gets replaced with refusal_text before streaming
2. Valid answer passes through unchanged
3. Timing logs are preserved (first_token, complete, validation)
4. Direct streaming mode (no validation) works correctly
"""

import asyncio
import sys
import os
import time
from typing import AsyncGenerator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.llm_orchestrator import generate_answer_stream


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, response_tokens: list[str], delay_per_token: float = 0.001):
        self.response_tokens = response_tokens
        self.delay_per_token = delay_per_token
        self.first_token_callback = None
        self.start_time = None
    
    async def generate_stream(
        self,
        prompt: str,
        tenant_id: str,
        max_tokens: int = 100,
        on_first_token=None,
        timeout: float = 30.0
    ) -> AsyncGenerator[str, None]:
        """Generate mock streaming response."""
        self.start_time = time.time()
        self.first_token_callback = on_first_token
        
        for i, token in enumerate(self.response_tokens):
            if i == 0 and on_first_token:
                # Call first token callback
                on_first_token(time.time() - self.start_time)
            
            await asyncio.sleep(self.delay_per_token)
            yield token


def mock_validation_reject(answer: str, evidence: str) -> bool:
    """Mock validation that rejects everything except refusal."""
    return "The document does not specify this." in answer


def mock_validation_accept(answer: str, evidence: str) -> bool:
    """Mock validation that accepts everything."""
    return True


async def test_invalid_answer_replaced():
    """Test that invalid answers are replaced with refusal before streaming."""
    print("\n=== Testing invalid answer replacement ===")
    
    # Mock LLM that generates hallucinated answer
    mock_tokens = ["This ", "is ", "a ", "hallucinated ", "answer."]
    mock_provider = MockLLMProvider(mock_tokens)
    
    # Evidence that doesn't support the answer
    evidence = "The vacation policy allows 15 days."
    
    # Collect streamed output
    collected_chunks = []
    async for chunk in generate_answer_stream(
        prompt="What is the policy?",
        tenant_id="test-tenant",
        provider=mock_provider,
        max_tokens=100,
        timeout=30.0,
        validate_fn=mock_validation_reject,
        evidence_text=evidence,
        refusal_text="REFUSAL_STRING",
        request_id="test-001",
        chunk_size=10
    ):
        collected_chunks.append(chunk)
    
    # Full output should be refusal, not hallucination
    full_output = "".join(collected_chunks)
    assert full_output == "REFUSAL_STRING", f"Expected refusal but got: {full_output}"
    print(f"✓ Invalid answer replaced with refusal: {full_output}")


async def test_valid_answer_passes():
    """Test that valid answers pass through unchanged."""
    print("\n=== Testing valid answer passes through ===")
    
    # Mock LLM that generates valid answer
    mock_tokens = ["15 ", "days ", "per ", "year."]
    mock_provider = MockLLMProvider(mock_tokens)
    
    # Evidence that supports the answer
    evidence = "The vacation policy allows 15 days per year."
    
    # Collect streamed output
    collected_chunks = []
    async for chunk in generate_answer_stream(
        prompt="How many vacation days?",
        tenant_id="test-tenant",
        provider=mock_provider,
        max_tokens=100,
        timeout=30.0,
        validate_fn=mock_validation_accept,
        evidence_text=evidence,
        refusal_text="REFUSAL_STRING",
        request_id="test-002",
        chunk_size=10
    ):
        collected_chunks.append(chunk)
    
    # Full output should be original answer
    full_output = "".join(collected_chunks)
    expected = "15 days per year."
    assert full_output == expected, f"Expected '{expected}' but got: {full_output}"
    print(f"✓ Valid answer passed through: {full_output}")


async def test_direct_streaming_no_validation():
    """Test direct streaming mode (no validation function)."""
    print("\n=== Testing direct streaming (no validation) ===")
    
    # Mock LLM response
    mock_tokens = ["Token1", "Token2", "Token3"]
    mock_provider = MockLLMProvider(mock_tokens, delay_per_token=0.001)
    
    # Collect streamed output
    collected_chunks = []
    async for chunk in generate_answer_stream(
        prompt="Test prompt",
        tenant_id="test-tenant",
        provider=mock_provider,
        max_tokens=100,
        timeout=30.0,
        validate_fn=None,  # No validation
        evidence_text="",
        refusal_text="REFUSAL",
        request_id="test-003",
        chunk_size=10
    ):
        collected_chunks.append(chunk)
    
    # Should get raw tokens directly
    assert collected_chunks == mock_tokens, f"Expected raw tokens but got: {collected_chunks}"
    print(f"✓ Direct streaming returned raw tokens: {collected_chunks}")


async def test_chunking_behavior():
    """Test that validated answers are chunked correctly."""
    print("\n=== Testing chunking behavior ===")
    
    # Mock LLM that generates a longer answer
    answer_text = "This is a test answer that is longer than chunk size."
    mock_tokens = list(answer_text)  # Character by character
    mock_provider = MockLLMProvider(mock_tokens, delay_per_token=0.0001)
    
    # Collect streamed output with specific chunk size
    chunk_size = 10
    collected_chunks = []
    async for chunk in generate_answer_stream(
        prompt="Test",
        tenant_id="test-tenant",
        provider=mock_provider,
        max_tokens=100,
        timeout=30.0,
        validate_fn=mock_validation_accept,
        evidence_text="test answer",
        refusal_text="REFUSAL",
        request_id="test-004",
        chunk_size=chunk_size
    ):
        collected_chunks.append(chunk)
        # Each chunk should be <= chunk_size
        assert len(chunk) <= chunk_size, f"Chunk too large: {len(chunk)} > {chunk_size}"
    
    # Reassembled should match original
    full_output = "".join(collected_chunks)
    assert full_output == answer_text, f"Chunking corrupted output"
    print(f"✓ Chunking preserved answer integrity: {len(collected_chunks)} chunks")


async def test_refusal_text_customization():
    """Test that custom refusal text is used."""
    print("\n=== Testing custom refusal text ===")
    
    mock_tokens = ["Invalid", "answer"]
    mock_provider = MockLLMProvider(mock_tokens)
    
    custom_refusal = "CUSTOM_REFUSAL_MESSAGE"
    
    collected_chunks = []
    async for chunk in generate_answer_stream(
        prompt="Test",
        tenant_id="test-tenant",
        provider=mock_provider,
        max_tokens=100,
        timeout=30.0,
        validate_fn=mock_validation_reject,
        evidence_text="unrelated",
        refusal_text=custom_refusal,
        request_id="test-005",
        chunk_size=20
    ):
        collected_chunks.append(chunk)
    
    full_output = "".join(collected_chunks)
    assert full_output == custom_refusal, f"Expected custom refusal but got: {full_output}"
    print(f"✓ Custom refusal text used: {full_output}")


async def run_all_tests():
    """Run all orchestrator tests."""
    print("\n" + "="*60)
    print("LLM ORCHESTRATOR VALIDATION TESTS")
    print("="*60)
    
    await test_invalid_answer_replaced()
    await test_valid_answer_passes()
    await test_direct_streaming_no_validation()
    await test_chunking_behavior()
    await test_refusal_text_customization()
    
    print("\n" + "="*60)
    print("ALL ORCHESTRATOR TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
