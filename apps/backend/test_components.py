#!/usr/bin/env python3
"""Isolated component testing"""

import sys
import os

# Test 1: PostgreSQL connection
print("\n=== TEST 1: PostgreSQL Connection ===")
try:
    from app.database import test_connection
    result = test_connection()
    print(f"✓ PostgreSQL: {result}")
except Exception as e:
    print(f"✗ PostgreSQL failed: {e}")
    sys.exit(1)

# Test 2: ChromaDB access
print("\n=== TEST 2: ChromaDB Access ===")
try:
    from app.services import clients
    # Initialize ChromaDB first!
    clients.initialize_chroma_client()
    chroma_client = clients.get_chroma_client()
    collections = chroma_client.list_collections()
    print(f"✓ ChromaDB: Found {len(collections)} collections")
    for c in collections:
        print(f"  - {c.name}: {c.count()} documents")
except Exception as e:
    print(f"✗ ChromaDB failed: {e}")
    sys.exit(1)

# Test 3: Ollama embedding (WITH TIMEOUT)
print("\n=== TEST 3: Ollama Embedding (with timeout) ===")
try:
    import asyncio
    from app.services.rag_service import embed_texts
    
    # Use asyncio with timeout
    async def test_embed():
        test_texts = ["What time should I arrive?"]
        result = await asyncio.wait_for(embed_texts(test_texts), timeout=10)
        return result
    
    embeddings = asyncio.run(test_embed())
    print(f"✓ Ollama embedding: Got {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
except asyncio.TimeoutError:
    print(f"✗ Ollama embedding TIMEOUT - Check if Ollama server is running on localhost:11434")
except Exception as e:
    print(f"✗ Ollama embedding failed: {e}")

# Test 4: File ingestion
print("\n=== TEST 4: File Ingestion ===")
try:
    from app.services import ingestion
    test_file = "demo_docs/Employee_Onboarding_Guide.txt"
    if not os.path.exists(test_file):
        print(f"✗ Demo file not found: {test_file}")
    else:
        text = ingestion.load_file_to_text(test_file)
        chunks = ingestion.chunk_text(text)
        print(f"✓ File ingestion: {len(text)} chars → {len(chunks)} chunks")
        print(f"  Sample chunk: {chunks[0][:100]}...")
except Exception as e:
    print(f"✗ File ingestion failed: {e}")

print("\n=== ALL TESTS COMPLETE ===")
