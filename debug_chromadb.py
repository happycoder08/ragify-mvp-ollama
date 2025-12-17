#!/usr/bin/env python3
"""Debug script to inspect ChromaDB content"""

import chromadb
from pathlib import Path

# Initialize ChromaDB client
db_path = Path('vectorstore')
print(f'ChromaDB path exists: {db_path.exists()}')
if db_path.exists():
    print(f'Contents of vectorstore/: {list(db_path.glob("*"))}')

client = chromadb.PersistentClient(path=str(db_path))
collections = client.list_collections()
print(f'\nTotal collections: {len(collections)}')

for c in collections:
    print(f'\n{"="*60}')
    print(f'Collection: {c.name}')
    print(f'{"="*60}')
    count = c.count()
    print(f'Document count: {count}')
    
    if count > 0:
        # Get all documents to see what's stored
        results = c.get()
        print(f'\nTotal chunks retrieved: {len(results["documents"])}')
        
        # Show first few chunks with metadata
        for i in range(min(5, len(results['documents']))):
            doc = results['documents'][i]
            metadata = results['metadatas'][i] if 'metadatas' in results else {}
            doc_id = results['ids'][i] if 'ids' in results else 'N/A'
            
            print(f'\n--- Chunk {i+1} ---')
            print(f'ID: {doc_id}')
            print(f'Metadata: {metadata}')
            print(f'Content (first 150 chars): {doc[:150]}...')
    else:
        print('\nNo documents in collection!')

print('\n' + '='*60)
print('DEBUG: Check if upload added documents to correct collection')
print('='*60)
