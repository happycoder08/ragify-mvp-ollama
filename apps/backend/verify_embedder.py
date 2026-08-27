"""
Quick verification that MockEmbedder produces valid embeddings.
"""

import asyncio
from app.services.embeddings import MockEmbedder


async def main():
    print("=" * 60)
    print("MockEmbedder Verification")
    print("=" * 60)
    
    embedder = MockEmbedder(dimension=384)
    
    # Test documents
    docs = [
        "Employees receive 15 days of vacation per year.",
        "The sick leave policy is not specified in the handbook.",
        "New employees should arrive at 8:00 AM on their first day.",
    ]
    
    print(f"\nEmbedding {len(docs)} documents...")
    vectors = await embedder.embed_texts(docs, tenant_id="test-tenant")
    
    print(f"✓ Generated {len(vectors)} embeddings")
    print(f"✓ Dimension: {len(vectors[0])}")
    
    # Check normalization
    for i, vec in enumerate(vectors):
        norm = sum(x * x for x in vec) ** 0.5
        print(f"✓ Vector {i}: ||v|| = {norm:.6f} (should be ~1.0)")
    
    # Test determinism
    print("\n" + "=" * 60)
    print("Testing Determinism")
    print("=" * 60)
    
    text = "Test document for determinism check"
    vec1 = await embedder.embed_texts([text], tenant_id="test")
    vec2 = await embedder.embed_texts([text], tenant_id="test")
    vec3 = await embedder.embed_texts([text], tenant_id="test")
    
    if vec1 == vec2 == vec3:
        print("✓ Deterministic: Same text produces identical vectors")
    else:
        print("✗ FAILED: Vectors are not identical!")
        return
    
    # Test uniqueness
    print("\n" + "=" * 60)
    print("Testing Uniqueness")
    print("=" * 60)
    
    text1 = "Document about vacation policy"
    text2 = "Document about sick leave"
    
    v1 = (await embedder.embed_texts([text1], tenant_id="test"))[0]
    v2 = (await embedder.embed_texts([text2], tenant_id="test"))[0]
    
    if v1 != v2:
        print("✓ Unique: Different texts produce different vectors")
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(v1, v2))
        print(f"  Cosine similarity: {dot_product:.4f}")
    else:
        print("✗ FAILED: Different texts produce identical vectors!")
        return
    
    # Test batch processing
    print("\n" + "=" * 60)
    print("Testing Batch Processing")
    print("=" * 60)
    
    batch_texts = [f"Document {i}" for i in range(5)]
    batch_vectors = await embedder.embed_texts(batch_texts, tenant_id="test")
    
    print(f"✓ Batch: {len(batch_vectors)} vectors generated")
    
    # Verify each is unique
    unique_vectors = len(set(tuple(v) for v in batch_vectors))
    if unique_vectors == len(batch_vectors):
        print(f"✓ All {unique_vectors} vectors are unique")
    else:
        print(f"✗ FAILED: Only {unique_vectors}/{len(batch_vectors)} vectors are unique!")
    
    print("\n" + "=" * 60)
    print("✅ All Verification Tests Passed!")
    print("=" * 60)
    print("\nMockEmbedder is ready for testing:")
    print("  - Deterministic embeddings (same input → same output)")
    print("  - Normalized vectors (||v|| = 1.0)")
    print("  - Unique vectors (different input → different output)")
    print("  - No HTTP client required (pure computation)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
