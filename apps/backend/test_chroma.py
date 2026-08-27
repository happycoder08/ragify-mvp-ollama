#!/usr/bin/env python
"""Check ChromaDB contents"""
import sys
sys.path.insert(0, '.')
from app.services import clients

# Initialize ChromaDB client
clients.initialize_chroma_client()

# Get ChromaDB client
client = clients.get_chroma_client()

# Get the default tenant collection
collection = client.get_or_create_collection(name='documents_default')

# Get all documents
results = collection.get()

print(f'Total chunks: {len(results.get("ids", []))}')
print(f'Unique files:')

filenames = {}
for metadata in results.get('metadatas', []):
    filename = metadata.get('filename', 'unknown')
    if filename not in filenames:
        filenames[filename] = True
        doc_id = metadata.get('doc_id', 'N/A')
        print(f'  - {filename} (doc_id={doc_id})')

print(f'\nAll metadata entries:')
for i, meta in enumerate(results.get('metadatas', [])[:5]):
    print(f'  {i}: {meta}')
