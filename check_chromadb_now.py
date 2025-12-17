#!/usr/bin/env python3
"""
Check what's actually in ChromaDB right now
"""
import asyncio
from app import clients

async def check_chromadb():
    print("Checking ChromaDB contents...\n")
    
    # Get ChromaDB client
    chroma_client = clients.get_chroma_client()
    
    # List all collections
    collections = chroma_client.list_collections()
    print(f"Total collections: {len(collections)}")
    for coll in collections:
        print(f"  - {coll.name}")
    
    # Check tenant_default collection
    try:
        collection = chroma_client.get_collection("tenant_default")
        count = collection.count()
        print(f"\nCollection 'tenant_default': {count} chunks")
        
        if count > 0:
            # Get all items
            results = collection.get(include=["documents", "metadatas"])
            print(f"\nChunks in collection:")
            for i, (doc_id, meta) in enumerate(zip(results["ids"], results["metadatas"])):
                print(f"  [{i}] id={doc_id} | source={meta.get('source_file')} | chunk={meta.get('chunk')}")
                
            # Check if our target document is there
            target_chunks = [meta for meta in results["metadatas"] if "Onboarding" in meta.get("source_file", "")]
            print(f"\nTarget chunks (from Employee_Onboarding_Guide.txt): {len(target_chunks)}")
            if target_chunks:
                for meta in target_chunks[:3]:
                    print(f"  - {meta}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_chromadb())
