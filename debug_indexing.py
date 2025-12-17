#!/usr/bin/env python3
"""Diagnostic script to debug the indexing pipeline"""

import asyncio
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Check files exist
    upload_dir = Path("./app/uploads")
    print(f"\n=== UPLOADS DIRECTORY ===")
    print(f"Path: {upload_dir.absolute()}")
    print(f"Exists: {upload_dir.exists()}")
    
    if upload_dir.exists():
        files = list(upload_dir.glob("*"))
        print(f"Files: {len(files)}")
        for f in files[:5]:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    
    # Check ChromaDB
    print(f"\n=== CHROMADB ===")
    import chromadb
    from app.services import clients
    
    chroma_client = clients.get_chroma_client()
    collections = chroma_client.list_collections()
    print(f"Total collections: {len(collections)}")
    
    for c in collections:
        print(f"\nCollection: {c.name}")
        count = c.count()
        print(f"  Documents: {count}")
        if count > 0:
            # Get first 3 chunks
            results = c.get(limit=3)
            for i, (doc_id, doc, meta) in enumerate(zip(
                results['ids'],
                results['documents'], 
                results['metadatas']
            )):
                print(f"\n  Chunk {i+1}:")
                print(f"    ID: {doc_id}")
                print(f"    Metadata: {meta}")
                print(f"    Content: {doc[:80]}...")
    
    # Test embedding
    print(f"\n=== OLLAMA EMBEDDING TEST ===")
    try:
        from app.services.rag_service import embed_texts
        test_text = ["What time should I arrive?"]
        embeddings = await embed_texts(test_text)
        print(f"✓ Embedding works: {len(embeddings[0])} dimensions")
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
    
    # Check app config
    print(f"\n=== APP CONFIG ===")
    from app.config import RAGIFY_MODE, MOCK_MODE
    print(f"RAGIFY_MODE: {RAGIFY_MODE}")
    print(f"MOCK_MODE: {MOCK_MODE}")

if __name__ == "__main__":
    asyncio.run(main())
