#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect what's in ChromaDB for Employee_Onboarding_Guide"""

import asyncio
from app.services import clients

async def main():
    clients.initialize_chroma_client()
    
    collection = clients.get_chroma_client().get_or_create_collection("documents_default")
    
    # Get all chunks
    results = collection.get()
    
    # Filter for Employee_Onboarding_Guide chunks
    onboarding_chunks = []
    for i, (doc_id, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        if 'Employee_Onboarding_Guide' in meta.get('source_file', ''):
            onboarding_chunks.append((doc_id, doc, meta))
    
    print(f"Total chunks from Employee_Onboarding_Guide.txt: {len(onboarding_chunks)}\n")
    
    # Look for "8:00 AM" specifically
    print("SEARCHING FOR '8:00'...")
    found_800 = False
    for doc_id, doc, meta in onboarding_chunks:
        if '8:00' in doc:
            print(f"FOUND '8:00' in chunk: {doc_id}")
            print(f"Content:\n{doc}\n")
            found_800 = True
    
    if not found_800:
        print("'8:00' NOT FOUND IN ANY CHUNK!\n")
        print("Checking if it's in the source document...")
        with open("demo_docs/Employee_Onboarding_Guide.txt", "r") as f:
            content = f.read()
            if "8:00" in content:
                print("YES, '8:00' IS IN THE SOURCE DOCUMENT!")
                idx = content.find("8:00")
                print(f"Context around '8:00': ...{content[max(0, idx-100):idx+200]}...")
            else:
                print("NO, '8:00' IS NOT IN THE SOURCE DOCUMENT!")

if __name__ == "__main__":
    asyncio.run(main())
