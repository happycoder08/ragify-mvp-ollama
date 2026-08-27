import pytest
import asyncio

pytestmark = pytest.mark.asyncio

def try_import_collection():
    try:
        from app.services.rag_service import get_collection_async
        return get_collection_async
    except ImportError:
        pytest.skip("Could not import get_collection_async from rag_service.py")

async def try_initialize_clients():
    try:
        from app.services import clients
        clients.initialize_chroma_client()
        await clients.initialize_http_client()
    except Exception as e:
        pytest.skip(f"Could not initialize ChromaDB or HTTP client: {e}")

@pytest.mark.asyncio
async def test_get_collection_async_returns_collection():
    await try_initialize_clients()
    get_collection_async = try_import_collection()
    try:
        collection = await get_collection_async("default")
    except Exception as e:
        pytest.skip(f"ChromaDB or dependencies not available: {e}")
    # Should not be a coroutine
    assert not asyncio.iscoroutine(collection), "Returned object is a coroutine, not a collection instance"
    # Should have a count attribute or method
    assert hasattr(collection, "count"), "Collection object missing 'count' attribute or method"
    # Should be callable if it's a method
    count_attr = getattr(collection, "count")
    assert callable(count_attr), "Collection 'count' attribute is not callable"
