#!/usr/bin/env python3
"""Inspect what's in ChromaDB for Employee_Onboarding_Guide"""

import asyncio
from app.services import clients

async def main():
    clients.initialize_chroma_client()
    
    collection = clients.get_chroma_client().get_or_create_collection("documents_default")
    
    print(f"Total chunks in collection: {collection.count()}\n")
    
    # Get all chunks
    results = collection.get()
    
    # Filter for Employee_Onboarding_Guide chunks
    onboarding_chunks = []
    for i, (doc_id, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        if 'Employee_Onboarding_Guide' in meta.get('source_file', ''):
            onboarding_chunks.append((doc_id, doc, meta))
    
    print(f"Chunks from Employee_Onboarding_Guide.txt: {len(onboarding_chunks)}\n")
    
    # Look for "8:00 AM" specifically
    print("="*70)
    print("ALL chunks searching for '8:00':")
    print("="*70)
    found_exact = False
    for doc_id, doc, meta in onboarding_chunks:
        if '8:00' in doc:
            print(f"\n✓ FOUND '8:00'!")
            print(f"  Chunk ID: {doc_id}")
            print(f"  Full content:\n{doc}\n")
            found_exact = True
    
    if not found_exact:
        print("\n⚠️  '8:00' not found in any chunk!")
        print("\nSearching for 'ARRIVE'...")
        for doc_id, doc, meta in onboarding_chunks:
            if 'ARRIVE' in doc or 'arrival' in doc.lower():
                print(f"\n✓ Found 'ARRIVE' mention:")
                print(f"  Chunk ID: {doc_id}")
                print(f"  Text: {doc[:300]}...\n")
    
    # Show ALL chunks
    print("\n" + "="*70)
    print(f"ALL {len(onboarding_chunks)} chunks from Employee_Onboarding_Guide:")
    print("="*70)
    for i, (doc_id, doc, meta) in enumerate(onboarding_chunks):
        print(f"\n--- Chunk {i+1} (ID: {doc_id}) ---")
        print(f"{doc[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())

