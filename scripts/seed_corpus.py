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


def _enable_mock_indexing_if_needed() -> None:
    """Allow corpus seeding in CI/mock mode where indexing is otherwise skipped."""
    is_mock_or_ci = (
        os.getenv("RAGIFY_MOCK", "0") == "1"
        or os.getenv("LLM_PROVIDER", "").lower() == "mock"
        or os.getenv("APP_MODE", "").lower() == "ci"
        or os.getenv("CI", "").lower() in {"1", "true", "yes"}
    )
    if is_mock_or_ci:
        os.environ["ALLOW_CHROMA_INDEXING_IN_MOCK"] = "true"
        logger.info("Mock/CI mode detected; enabling Chroma indexing for eval corpus seeding.")


def _iter_seed_files() -> list[Path]:
    """Return the eval corpus files, preferring the richer onboarding docs used by the demo/product flow."""
    candidates = [Path("uploads_stress"), Path("demo_docs"), Path("tests/testdata/docs")]
    files: list[Path] = []
    seen: set[str] = set()
    for base in candidates:
        if not base.exists():
            continue
        for f_path in sorted(base.glob("*.*")):
            resolved = str(f_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(f_path)
    return files


async def seed():
    _enable_mock_indexing_if_needed()
    logger.info("Initializing clients...")
    clients.initialize_chroma_client()
    await clients.initialize_http_client()

    files = _iter_seed_files()
    if not files:
        logger.error("No seed corpus files found under uploads_stress/ or tests/testdata/docs/")
        return

    logger.info(f"Found {len(files)} files across seed corpus directories")

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
