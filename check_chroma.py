#!/usr/bin/env python3
"""Check ChromaDB contents directly"""
from app.services.clients import get_chroma_client

client = get_chroma_client()
collections = client.list_collections()
print(f"Collections: {[c.name for c in collections]}")

try:
    coll = client.get_collection("tenant_default")
    count = coll.count()
    print(f"Tenant_default count: {count}")
    
    if count > 0:
        results = coll.get(include=["documents", "metadatas"])
        print(f"\nChunks in collection:")
        for i, (doc_id, meta, doc) in enumerate(zip(results["ids"], results["metadatas"], results["documents"])):
            print(f"[{i}] {meta.get('source_file')} chunk {meta.get('chunk')}: {doc[:80]}...")
except Exception as e:
    print(f"Error: {e}")
