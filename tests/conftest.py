import os
import sys
import shutil
import logging
import json
import pytest
import httpx
from pathlib import Path
from asgi_lifespan import LifespanManager

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# ✅ Set CI mode + VECTOR_DIR BEFORE importing app modules that read config
VECTOR_DIR_TEST = os.path.join(REPO_ROOT, "vectorstore_test")
os.environ["APP_MODE"] = "ci"
os.environ["CI"] = "true"
os.environ["VECTOR_DIR"] = VECTOR_DIR_TEST
os.environ["ALLOW_CHROMA_INDEXING_IN_MOCK"] = "true"

# ✅ Silence Chroma shutdown chatter early
null = logging.NullHandler()
for name in ("chromadb", "chromadb.db", "chromadb.db.duckdb"):
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.addHandler(null)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False

from main import app
import main
from app.services import clients, rag_service
from app.services.ingestion import load_file_to_text, chunk_text


@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def seed_vectorstore():
    # Reset cached providers so env vars apply cleanly
    if hasattr(rag_service, "reset_embedding_provider_for_tests"):
        rag_service.reset_embedding_provider_for_tests()
    if hasattr(rag_service, "reset_llm_provider_for_tests"):
        rag_service.reset_llm_provider_for_tests()

    # Ensure CI/mock mode + allow chroma indexing during seeding
    os.environ["APP_MODE"] = "ci"
    os.environ["CI"] = "true"
    os.environ["ALLOW_CHROMA_INDEXING_IN_MOCK"] = "true"

    # Clean slate
    if os.path.exists(VECTOR_DIR_TEST):
        shutil.rmtree(VECTOR_DIR_TEST)
    os.makedirs(VECTOR_DIR_TEST, exist_ok=True)

    # Init Chroma
    clients.initialize_chroma_client()

    docs_dir = Path(REPO_ROOT) / "tests" / "testdata" / "docs"
    manifest_path = docs_dir / "manifest.json"
    assert docs_dir.exists(), f"Missing docs folder: {docs_dir}"
    assert manifest_path.exists(), f"Missing manifest.json: {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_file = {m["file"]: m for m in manifest}

    # Deterministic ingestion order
    allowed_ext = {".txt", ".md", ".pdf", ".docx"}
    paths = sorted([p for p in docs_dir.iterdir() if p.suffix.lower() in allowed_ext])

    total_chunks_indexed = 0
    per_file_counts = {}

    for path in paths:
        filename = path.name

        # Extract text using your production ingestion code
        text = load_file_to_text(str(path))

        # Fail fast: ensure extraction is sane (manifest anchors must exist)
        spec = manifest_by_file.get(filename)
        assert spec is not None, f"{filename} missing from manifest.json"
        for needle in spec.get("must_contain", []):
            assert needle in text, f"[{filename}] extraction missing required token: {needle}"

        # Chunk using production chunker
        chunks = chunk_text(text)
        assert chunks, f"[{filename}] produced 0 chunks"

        # Seed via production add_documents
        n = await rag_service.add_documents("default", chunks, filename)
        per_file_counts[filename] = n
        total_chunks_indexed += n

        # Sanity: if seeding returns 0, something is wrong (unless your add_documents is skipping)
        assert n > 0, f"[{filename}] add_documents returned 0 chunks indexed"

    # Assert each file contributed chunks
    for filename, count in per_file_counts.items():
        assert count > 0, f"File {filename} contributed 0 chunks"

    # Assert seeded using same collection accessor as retrieval
    col = await rag_service.get_collection_async("default")
    assert col.count() > 0, "Chroma collection not seeded!"
    assert total_chunks_indexed > 0, "No chunks indexed across all docs!"

    # Optional: print counts (useful when tuning chunking)
    # print("Seeded chunks per file:", per_file_counts)

    yield

    # Cleanup after session
    if os.path.exists(VECTOR_DIR_TEST):
        shutil.rmtree(VECTOR_DIR_TEST)


@pytest.fixture(scope="session")
async def asgi_client(seed_vectorstore):
    # Override auth + db
    from app.auth import get_current_user
    from app.database import get_db

    async def _fake_current_user():
        return {"tenant_id": "default", "username": "golden-test"}

    def _fake_db():
        yield None

    app.dependency_overrides[get_current_user] = _fake_current_user
    app.dependency_overrides[get_db] = _fake_db

    # Disable rate limiter deterministically
    class _NoopLimiter:
        def check_rate_limit(self, tenant_id, upload_size_mb=0):
            return True, None
        def record_request(self, tenant_id, upload_size_mb=0):
            return None

    old_get_rate_limiter = getattr(main, "get_rate_limiter", None)
    main.get_rate_limiter = lambda: _NoopLimiter()

    try:
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
    finally:
        app.dependency_overrides.clear()
        if old_get_rate_limiter is not None:
            main.get_rate_limiter = old_get_rate_limiter
        else:
            delattr(main, "get_rate_limiter")
