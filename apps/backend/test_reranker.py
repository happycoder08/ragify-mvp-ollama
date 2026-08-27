"""
Test script for reranker providers.
Tests NoneReranker, JinaReranker, and CohereReranker.
"""

import os
import sys

# Test documents
query = "What are the remote work policies?"
documents = [
    "Employees can work from home up to 2 days per week.",
    "The company cafeteria serves lunch from 11am to 2pm.",
    "Remote work requires manager approval and stable internet connection.",
    "Annual leave policies allow 15 days of paid vacation.",
    "Video conferencing tools are available for remote collaboration.",
]

print("=" * 60)
print("Testing Reranker Providers")
print("=" * 60)

# Test 1: NoneReranker (always available)
print("\n[1] Testing NoneReranker (pass-through)...")
from app.services.reranker_providers import NoneReranker

reranker = NoneReranker()
indices, scores = reranker.rerank(query, documents, top_n=3)

print(f"Query: {query}")
print(f"Top {len(indices)} results:")
for idx, (i, score) in enumerate(zip(indices, scores)):
    print(f"  {idx+1}. [Score: {score:.4f}] {documents[i]}")

# Test 2: JinaReranker (if API key available)
print("\n[2] Testing JinaReranker...")
if os.getenv("JINA_API_KEY"):
    try:
        from app.services.reranker_providers import JinaReranker
        
        reranker = JinaReranker()
        indices, scores = reranker.rerank(query, documents, top_n=3)
        
        print(f"Query: {query}")
        print(f"Top {len(indices)} results:")
        for idx, (i, score) in enumerate(zip(indices, scores)):
            print(f"  {idx+1}. [Score: {score:.4f}] {documents[i]}")
    except Exception as e:
        print(f"❌ JinaReranker failed: {e}")
else:
    print("⚠️  Skipped (JINA_API_KEY not set)")
    print("   Set JINA_API_KEY to test: export JINA_API_KEY=your_key")

# Test 3: CohereReranker (if API key available)
print("\n[3] Testing CohereReranker...")
if os.getenv("COHERE_API_KEY"):
    try:
        from app.services.reranker_providers import CohereReranker
        
        reranker = CohereReranker()
        indices, scores = reranker.rerank(query, documents, top_n=3)
        
        print(f"Query: {query}")
        print(f"Top {len(indices)} results:")
        for idx, (i, score) in enumerate(zip(indices, scores)):
            print(f"  {idx+1}. [Score: {score:.4f}] {documents[i]}")
    except Exception as e:
        print(f"❌ CohereReranker failed: {e}")
else:
    print("⚠️  Skipped (COHERE_API_KEY not set)")
    print("   Set COHERE_API_KEY to test: export COHERE_API_KEY=your_key")

# Test 4: Factory function
print("\n[4] Testing create_reranker_provider() factory...")
from app.services.reranker_providers import create_reranker_provider

# Test with RERANKER_PROVIDER=none
os.environ["RERANKER_PROVIDER"] = "none"
reranker = create_reranker_provider()
print(f"RERANKER_PROVIDER=none → {type(reranker).__name__}")

# Test with RERANKER_PROVIDER=invalid (should fall back to NoneReranker)
os.environ["RERANKER_PROVIDER"] = "invalid"
reranker = create_reranker_provider()
print(f"RERANKER_PROVIDER=invalid → {type(reranker).__name__} (fallback)")

print("\n" + "=" * 60)
print("✅ Reranker provider tests complete!")
print("=" * 60)
