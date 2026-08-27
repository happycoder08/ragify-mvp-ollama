import sys
import os
import asyncio
import glob

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services import ingestion, rag_service, clients

async def ingest_demo_files():
    # Initialize HTTP client for embeddings
    await clients.initialize_http_client()
    clients.initialize_chroma_client()
    
    try:
        tenant_id = "default"
        demo_dir = os.path.join(project_root, "demo", "policy_pack")
        files = glob.glob(os.path.join(demo_dir, "*.md"))
        
        print(f"Found {len(files)} files in {demo_dir}")
        
        for i, file_path in enumerate(files):
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...")
            
            # 1. Load text
            text = ingestion.load_file_to_text(file_path)
            
            # 2. Chunk
            # Using section chunking as per app convention
            chunks = ingestion.chunk_text_sections(text)
            print(f"  - Generated {len(chunks)} chunks")
            
            # 3. Index
            doc_id = 9000 + i 
            await rag_service.index_files(tenant_id, chunks, filename, doc_id=doc_id)
            print(f"  - Indexed to {tenant_id}")
            
    finally:
        await clients.close_http_client()

if __name__ == "__main__":
    # Ensure mode is demo
    os.environ["RAGIFY_MODE"] = "demo"
    asyncio.run(ingest_demo_files())
