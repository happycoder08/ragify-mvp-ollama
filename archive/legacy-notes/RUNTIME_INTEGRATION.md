# AppRuntime Integration Examples

This document shows how to integrate AppRuntime into the RAGify application incrementally.

## Production Usage (main.py)

```python
from fastapi import FastAPI
from app.runtime import build_runtime_from_env

app = FastAPI()

# Global runtime instance
runtime = None

@app.on_event("startup")
async def startup():
    global runtime
    # Initialize runtime with production dependencies
    runtime = build_runtime_from_env()
    logger.info("Application runtime initialized")

@app.post("/api/query")
async def query(request: QueryRequest, user: dict = Depends(get_current_user)):
    """Use runtime dependencies instead of global singletons."""
    
    # Get LLM provider from runtime
    llm_provider = runtime.llm_provider
    
    # Generate response using runtime provider
    async for token in llm_provider.generate_stream(
        prompt=prompt,
        tenant_id=user["tenant_id"],
        max_tokens=500
    ):
        yield token
```

## Testing Usage

```python
from app.runtime import build_test_runtime
from starlette.testclient import TestClient

def test_query_with_mock_runtime():
    # Build test runtime with mock providers
    test_runtime = build_test_runtime()
    
    # Override app runtime
    app.dependency_overrides[get_runtime] = lambda: test_runtime
    
    # Create test client
    client = TestClient(app)
    
    # Test query endpoint
    response = client.post("/api/query", json={"question": "test"})
    
    # Test uses mock provider - no external dependencies needed
    assert response.status_code == 200
```

## Incremental Migration Strategy

### Phase 1: Add Runtime Dependency (Done ✅)
- Created `app/runtime.py` with AppRuntime container
- Created factory functions: `build_runtime_from_env()`, `build_test_runtime()`
- Added unit tests verifying CI compatibility

### Phase 2: Initialize Runtime in main.py (Future)
```python
# In main.py startup:
from app.runtime import build_runtime_from_env

@app.on_event("startup")
async def startup():
    global runtime
    runtime = build_runtime_from_env()
```

### Phase 3: Add FastAPI Dependency (Future)
```python
# Create dependency injector
def get_runtime() -> AppRuntime:
    return runtime

# Use in endpoints
@app.post("/api/query")
async def query(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
    rt: AppRuntime = Depends(get_runtime)  # Inject runtime
):
    # Use rt.llm_provider instead of global rag_service
    ...
```

### Phase 4: Refactor Endpoints One at a Time (Future)
- Start with `/api/query` - inject runtime, use rt.llm_provider
- Then `/api/upload` - use rt.task_runner for background indexing
- Finally other endpoints as needed

### Phase 5: Update Tests (Future)
```python
@pytest.fixture
def test_runtime():
    return build_test_runtime()

@pytest.fixture
def client(test_runtime):
    app.dependency_overrides[get_runtime] = lambda: test_runtime
    yield TestClient(app)
    app.dependency_overrides.clear()
```

## Benefits

### For Testing
- ✅ **No External Dependencies**: Tests run with `db_enabled=False`, `http_client=None`
- ✅ **Mock Providers**: `LLM_PROVIDER=mock` provides deterministic responses
- ✅ **Synchronous Tasks**: `sync_task_runner` makes tests predictable
- ✅ **Easy Fixtures**: `build_test_runtime()` creates isolated test environment

### For Production
- ✅ **Clear Dependencies**: All dependencies in one place
- ✅ **Environment-Based Config**: `build_runtime_from_env()` reads config
- ✅ **Swappable Providers**: Easy to switch between Ollama/OpenAI/Mock
- ✅ **Background Tasks**: `background_task_runner` for async operations

### For Development
- ✅ **Incremental Migration**: Can adopt gradually without big-bang refactor
- ✅ **Type Safety**: AppRuntime is a dataclass with clear types
- ✅ **Testability**: Easy to mock individual components
- ✅ **No Breaking Changes**: Existing code continues to work

## Current Status

**Implemented:**
- ✅ `app/runtime.py` - AppRuntime container
- ✅ `test_runtime.py` - Unit tests (5/5 passing)
- ✅ `build_runtime_from_env()` - Production factory
- ✅ `build_test_runtime()` - Test factory
- ✅ Mock provider support verified

**Not Implemented (Future Work):**
- Integration into main.py startup
- FastAPI dependency injection
- Endpoint refactoring
- Integration test migration

**Design Decisions:**
- Minimal footprint - no immediate refactoring required
- Compatible with existing global singletons
- Can be adopted incrementally per endpoint
- Test-first design - tests work without DB/HTTP client
