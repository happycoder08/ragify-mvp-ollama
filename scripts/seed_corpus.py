import asyncio
import os
import sys
import logging
from pathlib import Path

# Fix imports
sys.path.append(os.getcwd())

from app.services import ingestion, rag_service, clients

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed():
    logger.info("Initializing clients...")
    clients.initialize_chroma_client()
    await clients.initialize_http_client()
    
    stress_dir = Path("uploads_stress")
    if not stress_dir.exists():
        logger.error(f"Directory {stress_dir} not found!")
        return

    files = list(stress_dir.glob("*.*"))
    logger.info(f"Found {len(files)} files in {stress_dir}")

    tenant_id = "default"

    for f_path in files:
        logger.info(f"Processing {f_path.name}...")
        try:
            # 1. Load text
            text = ingestion.load_file_to_text(str(f_path))
            if not text:
                logger.warning(f"File {f_path.name} is empty or unreadable.")
                continue

            # 2. Chunk
            # Use section-aware chunking if possible
            chunks = ingestion.chunk_text_sections(text) 
            if not chunks:
                 chunks = ingestion.chunk_text(text)
            
            logger.info(f"Generated {len(chunks)} chunks for {f_path.name}")

            # 3. Index
            # index_files(tenant_id, chunks, source_filename, doc_id=None)
            await rag_service.index_files(
                tenant_id=tenant_id,
                chunks=chunks,
                source_filename=f_path.name
            )
            logger.info(f"Indexed {f_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {f_path.name}: {e}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(seed())
