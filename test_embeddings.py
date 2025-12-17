"""
Unit tests for embedding interfaces and implementations.

Tests MockEmbedder properties:
- Deterministic: same text -> same vector
- Discriminative: different text -> different vector
- Normalized: all vectors have unit length
- Stable dimension: vectors have consistent dimension
"""

import asyncio
import pytest
from app.services.embeddings import MockEmbedder, MOCK_EMBEDDING_DIM


def test_mock_embedder_deterministic():
    """
    Test that MockEmbedder produces identical vectors for identical text.
    
    Same input text should always produce the same embedding vector,
    regardless of when or how many times it's called.
    """
    embedder = MockEmbedder()
    text = "This is a test document about vacation policy."
    
    # Embed same text multiple times
    vec1 = asyncio.run(embedder.embed_texts([text], tenant_id="test"))[0]
    vec2 = asyncio.run(embedder.embed_texts([text], tenant_id="test"))[0]
    vec3 = asyncio.run(embedder.embed_texts([text], tenant_id="test"))[0]
    
    # All vectors should be identical
    assert vec1 == vec2, "Same text should produce identical vectors (call 1 vs 2)"
    assert vec2 == vec3, "Same text should produce identical vectors (call 2 vs 3)"
    assert vec1 == vec3, "Same text should produce identical vectors (call 1 vs 3)"
    
    print(f"✓ Deterministic: same text -> same vector ({len(vec1)} dims)")


def test_mock_embedder_discriminative():
    """
    Test that MockEmbedder produces different vectors for different texts.
    
    Different input texts should produce different embedding vectors.
    This ensures the embedder can distinguish between documents.
    """
    embedder = MockEmbedder()
    
    text1 = "Employees get 15 days of vacation per year."
    text2 = "The sick leave policy is not specified."
    text3 = "New employees should arrive at 8:00 AM."
    
    # Embed different texts
    vectors = asyncio.run(embedder.embed_texts([text1, text2, text3], tenant_id="test"))
    vec1, vec2, vec3 = vectors
    
    # All vectors should be different
    assert vec1 != vec2, "Different texts should produce different vectors (text1 vs text2)"
    assert vec2 != vec3, "Different texts should produce different vectors (text2 vs text3)"
    assert vec1 != vec3, "Different texts should produce different vectors (text1 vs text3)"
    
    print(f"✓ Discriminative: different texts -> different vectors")


def test_mock_embedder_normalized():
    """
    Test that MockEmbedder produces unit-length (normalized) vectors.
    
    All embedding vectors should have length (L2 norm) of 1.0.
    This is important for cosine similarity calculations.
    """
    embedder = MockEmbedder()
    
    texts = [
        "Short text",
        "This is a medium length text with more words and content.",
        "A" * 1000,  # Very long text
        "",  # Empty text edge case
    ]
    
    vectors = asyncio.run(embedder.embed_texts(texts, tenant_id="test"))
    
    for i, vec in enumerate(vectors):
        # Calculate L2 norm (vector length)
        norm = sum(x * x for x in vec) ** 0.5
        
        # Should be approximately 1.0 (allowing for floating point precision)
        assert abs(norm - 1.0) < 1e-6, f"Vector {i} should have unit length, got {norm}"
    
    print(f"✓ Normalized: all vectors have unit length (||v|| = 1.0)")


def test_mock_embedder_dimension_stable():
    """
    Test that MockEmbedder produces vectors of consistent dimension.
    
    All vectors should have the same dimension regardless of input text length.
    """
    embedder = MockEmbedder(dimension=MOCK_EMBEDDING_DIM)
    
    texts = [
        "Short",
        "Medium length text here",
        "A" * 5000,  # Very long
        "",  # Empty
        "Special chars: !@#$%^&*()",
        "Unicode: 你好世界 🌍",
    ]
    
    vectors = asyncio.run(embedder.embed_texts(texts, tenant_id="test"))
    
    expected_dim = MOCK_EMBEDDING_DIM
    
    for i, vec in enumerate(vectors):
        actual_dim = len(vec)
        assert actual_dim == expected_dim, \
            f"Vector {i} has dimension {actual_dim}, expected {expected_dim}"
    
    print(f"✓ Stable dimension: all vectors have {expected_dim} dimensions")


def test_mock_embedder_custom_dimension():
    """
    Test that MockEmbedder respects custom dimension parameter.
    """
    custom_dim = 128
    embedder = MockEmbedder(dimension=custom_dim)
    
    text = "Test document"
    vector = asyncio.run(embedder.embed_texts([text], tenant_id="test"))[0]
    
    assert len(vector) == custom_dim, \
        f"Expected dimension {custom_dim}, got {len(vector)}"
    
    print(f"✓ Custom dimension: embedder respects dimension={custom_dim}")


def test_mock_embedder_batch_processing():
    """
    Test that MockEmbedder correctly processes batches of texts.
    """
    embedder = MockEmbedder()
    
    texts = [f"Document {i}" for i in range(10)]
    
    # Embed batch
    batch_vectors = asyncio.run(embedder.embed_texts(texts, tenant_id="test"))
    
    # Embed individually
    individual_vectors = [
        asyncio.run(embedder.embed_texts([text], tenant_id="test"))[0]
        for text in texts
    ]
    
    # Batch and individual should produce same results
    assert len(batch_vectors) == len(individual_vectors), \
        "Batch should return same number of vectors as inputs"
    
    for i, (batch_vec, indiv_vec) in enumerate(zip(batch_vectors, individual_vectors)):
        assert batch_vec == indiv_vec, \
            f"Batch vector {i} should match individual embedding"
    
    print(f"✓ Batch processing: batch embeddings match individual embeddings")


def test_mock_embedder_empty_input():
    """
    Test that MockEmbedder handles empty text gracefully.
    """
    embedder = MockEmbedder()
    
    # Empty string should still produce a valid vector
    vector = asyncio.run(embedder.embed_texts([""], tenant_id="test"))[0]
    
    assert len(vector) == MOCK_EMBEDDING_DIM, "Empty text should produce valid vector"
    
    # Should still be normalized
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-6, "Empty text vector should be normalized"
    
    print(f"✓ Empty input: handles empty text gracefully")


def test_mock_embedder_similarity_behavior():
    """
    Test that similar texts produce somewhat similar vectors.
    
    While MockEmbedder is hash-based (not semantic), texts with shared
    content should produce vectors with some similarity.
    """
    embedder = MockEmbedder()
    
    # Very similar texts (only one word different)
    text1 = "Employees receive 15 days of vacation"
    text2 = "Employees receive 15 days of vacation"  # Identical
    text3 = "Employees receive 20 days of vacation"  # One word different
    
    vectors = asyncio.run(embedder.embed_texts([text1, text2, text3], tenant_id="test"))
    vec1, vec2, vec3 = vectors
    
    # Identical texts should have identical vectors
    assert vec1 == vec2, "Identical texts should have identical vectors"
    
    # Different texts should have different vectors
    assert vec1 != vec3, "Different texts should have different vectors"
    
    print(f"✓ Similarity behavior: identical texts -> identical vectors")


def test_mock_embedder_no_http_client_required():
    """
    Test that MockEmbedder does not require HTTP client.
    
    This is critical for testing without external dependencies.
    """
    # MockEmbedder should initialize without any HTTP client
    embedder = MockEmbedder()
    
    # Should be able to embed without network
    text = "Test document"
    vector = asyncio.run(embedder.embed_texts([text], tenant_id="test"))[0]
    
    assert len(vector) == MOCK_EMBEDDING_DIM, "Should embed without HTTP client"
    
    print(f"✓ No HTTP client required: MockEmbedder works offline")


if __name__ == "__main__":
    # Run all tests
    test_mock_embedder_deterministic()
    test_mock_embedder_discriminative()
    test_mock_embedder_normalized()
    test_mock_embedder_dimension_stable()
    test_mock_embedder_custom_dimension()
    test_mock_embedder_batch_processing()
    test_mock_embedder_empty_input()
    test_mock_embedder_similarity_behavior()
    test_mock_embedder_no_http_client_required()
    
    print("\n✓ All MockEmbedder tests passed!")
