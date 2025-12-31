import os
import sys
import shutil
import logging
import pytest
import httpx
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

    # Clean slate
    if os.path.exists(VECTOR_DIR_TEST):
        shutil.rmtree(VECTOR_DIR_TEST)
    os.makedirs(VECTOR_DIR_TEST, exist_ok=True)

    # Init Chroma
    clients.initialize_chroma_client()

    # Load onboarding doc
    doc_path = os.path.join(REPO_ROOT, "testdata", "onboarding", "onboarding_guide.txt")
    text = load_file_to_text(doc_path)
    chunks = chunk_text(text)

    # Seed using the real pipeline (async)
    await rag_service.add_documents("default", chunks, "onboarding_guide.txt")

    # Assert using the same collection accessor as retrieval
    col = await rag_service.get_collection_async("default")
    assert col.count() > 0, "Chroma collection not seeded!"

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
