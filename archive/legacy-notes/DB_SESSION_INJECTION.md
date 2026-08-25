# Database Session Injection for Ingestion Tasks

## Overview
Refactored `process_document_background` to use dependency-injected database sessions instead of global `SessionLocal`, enabling proper testing and eliminating global state.

## Changes Made

### 1. Updated `process_document_background` Function ([main.py](main.py))

**Before:**
```python
async def process_document_background(doc_id: int, tenant_id: str, file_path: str, filename: str):
    from app.database import SessionLocal
    db = SessionLocal()  # Global session factory
    try:
        # ... process document ...
        db.commit()
    finally:
        db.close()
```

**After:**
```python
async def process_document_background(
    doc_id: int,
    tenant_id: str,
    file_path: str,
    filename: str,
    db_session_factory  # Injected dependency
):
    # Create database session using injected factory
    db = None
    if db_session_factory is not None:
        db = db_session_factory()  # No global import
    
    try:
        # ... process document ...
        if db is not None and doc_id != -1:
            db.commit()
    finally:
        if db is not None:
            db.close()
```

**Key Changes:**
- ✅ Added `db_session_factory` parameter
- ✅ Removed global `SessionLocal` import
- ✅ Create session using injected factory
- ✅ Handle case when `db_session_factory` is `None` (no database)
- ✅ Handle case when `doc_id` is `-1` (no DB record)

### 2. Updated Task Submission ([main.py](main.py))

**Before:**
```python
task_runner.submit(
    process_document_background,
    doc_id,
    tenant_id,
    saved_path,
    file.filename
)
```

**After:**
```python
task_runner.submit(
    process_document_background,
    doc_id=doc_id,
    tenant_id=tenant_id,
    file_path=saved_path,
    filename=file.filename,
    db_session_factory=runtime.get_db_session  # Explicit injection
)
```

**Key Changes:**
- ✅ Use keyword arguments for clarity
- ✅ Pass `runtime.get_db_session` as `db_session_factory`
- ✅ All parameters explicitly passed (no globals)

### 3. Created Comprehensive Tests ([test_ingestion_task.py](test_ingestion_task.py))

**Tests Created:**
1. `test_process_document_background_with_session_factory` - Verifies DB session injection works
2. `test_process_document_background_handles_errors` - Verifies error handling updates status to "failed"
3. `test_process_document_background_without_db` - Verifies task works without database
4. `test_process_document_background_explicit_parameters` - Verifies all params passed explicitly

**Test Architecture:**
```python
@pytest.fixture
def test_session_factory(test_db_engine):
    """Create a session factory for testing."""
    TestingSessionLocal = sessionmaker(bind=test_db_engine)
    return TestingSessionLocal

def test_process_document_background_with_session_factory(
    test_session_factory,
    sample_document_file
):
    # Create document with "pending" status
    db = test_session_factory()
    doc = Document(tenant_id="test-tenant", status="pending", ...)
    db.add(doc)
    db.commit()
    doc_id = doc.id
    db.close()
    
    # Run task with injected session factory
    asyncio.run(process_document_background(
        doc_id=doc_id,
        tenant_id="test-tenant",
        file_path=sample_document_file,
        filename="test_document.txt",
        db_session_factory=test_session_factory  # Injected!
    ))
    
    # Verify status was updated (proves DB injection worked)
    db = test_session_factory()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    assert doc.status != "pending"  # Changed from pending
    db.close()
```

## Benefits

### 1. **Testability**
- ✅ No global database state
- ✅ Each test creates its own isolated database
- ✅ Tests use temporary SQLite databases
- ✅ Can verify DB updates happen correctly

### 2. **Explicit Dependencies**
- ✅ All parameters passed explicitly: `tenant_id`, `doc_id`, `db_session_factory`
- ✅ No hidden globals or imports inside functions
- ✅ Clear function signature shows all dependencies

### 3. **Flexibility**
- ✅ Production: Uses `runtime.get_db_session` (PostgreSQL)
- ✅ Testing: Uses test session factory (SQLite)
- ✅ No DB mode: Pass `None` as `db_session_factory`

### 4. **Clean Architecture**
- ✅ Functions don't import global state
- ✅ Dependencies injected at call site
- ✅ Easier to reason about and maintain

## Test Results

```
test_ingestion_task.py::test_process_document_background_with_session_factory PASSED
test_ingestion_task.py::test_process_document_background_handles_errors PASSED
test_ingestion_task.py::test_process_document_background_without_db PASSED
test_ingestion_task.py::test_process_document_background_explicit_parameters PASSED

✅ 4/4 ingestion tests passing
✅ 32/32 total unit tests passing (runtime, task_runner, embeddings, ingestion)
```

## Usage Examples

### Production (main.py)
```python
# Runtime holds database session factory
runtime = build_runtime_from_env()

# Upload endpoint submits task with injected factory
task_runner.submit(
    process_document_background,
    doc_id=doc.id,
    tenant_id=tenant_id,
    file_path=saved_path,
    filename=file.filename,
    db_session_factory=runtime.get_db_session  # Production DB
)
```

### Testing
```python
# Create test session factory
test_session_factory = sessionmaker(bind=test_engine)

# Run task with test factory
asyncio.run(process_document_background(
    doc_id=123,
    tenant_id="test-tenant",
    file_path="/path/to/file",
    filename="test.txt",
    db_session_factory=test_session_factory  # Test DB
))

# Verify DB was updated
db = test_session_factory()
doc = db.query(Document).filter(Document.id == 123).first()
assert doc.status == "indexed"
db.close()
```

### No Database Mode
```python
# Run without database (useful for testing document processing only)
asyncio.run(process_document_background(
    doc_id=-1,  # -1 indicates no DB record
    tenant_id="test-tenant",
    file_path="/path/to/file",
    filename="test.txt",
    db_session_factory=None  # No database
))
# Document is processed but no DB updates happen
```

## Migration Notes

### Key Constraint: Explicit Parameters
All task parameters must be passed explicitly:
- ✅ `tenant_id` - Always passed, never inferred
- ✅ `doc_id` - Always passed, use `-1` if no DB record
- ✅ `db_session_factory` - Always passed, use `None` if no DB

### Handling Async vs Sync Session Factories
The code handles both sync and async context managers:
```python
if inspect.isasyncgenfunction(db_session_factory):
    # Async context manager (for test fixtures)
    async with db_session_factory() as session:
        db = session
else:
    # Sync callable (SessionLocal)
    db = db_session_factory()
```

This allows flexibility in testing while maintaining production compatibility.

## Future Improvements

### Potential Enhancements
1. **Pass AppRuntime Instead:** Instead of just `db_session_factory`, pass entire `runtime` object
   - Would provide access to `http_client`, `llm_provider`, etc.
   - Single dependency instead of multiple parameters
   
2. **Dependency Injection Container:** Use proper DI framework
   - More formal dependency management
   - Automatic injection based on type hints

3. **Session Scoping:** Use context managers for automatic cleanup
   ```python
   async with db_session_factory() as db:
       # ... do work ...
       # Automatic commit/rollback and close
   ```

## Summary

Successfully eliminated global database state from ingestion tasks by:
- ✅ Adding `db_session_factory` parameter to `process_document_background`
- ✅ Passing `runtime.get_db_session` from upload endpoint
- ✅ Creating 4 comprehensive tests with isolated test databases
- ✅ Ensuring all parameters (`tenant_id`, `doc_id`, `db_session_factory`) are explicit
- ✅ All 32 unit tests passing

The ingestion pipeline is now fully testable with no global state! 🎉
