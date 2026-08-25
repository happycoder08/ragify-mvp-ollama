# Synchronous Upload in CI Mode

## Overview

The `/api/upload` endpoint behaves differently depending on the execution environment:

- **Production Mode** (BackgroundTaskRunner): Upload returns immediately, indexing happens asynchronously in background
- **CI/Test Mode** (InlineTaskRunner): Upload waits for indexing to complete, documents are immediately queryable

This ensures predictable, deterministic behavior in tests while maintaining non-blocking performance in production.

## Behavior

### Production Mode (Default)

```bash
# Normal production environment
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
```

**Flow:**
1. POST /api/upload → returns immediately with `status="pending"`
2. Indexing happens in background task (async)
3. Client polls GET /api/documents/{id}/status until `status="indexed"`
4. POST /api/query → returns evidence

**Response:**
```json
{
  "status": "ok",
  "message": "2 file(s) uploaded. Processing in background.",
  "documents": [
    {
      "id": 123,
      "filename": "policy.txt",
      "status": "pending",
      "created_at": "2025-12-17T10:30:00"
    }
  ]
}
```

### CI/Test Mode (InlineTaskRunner)

```bash
# CI environment or test execution
CI=true
# OR
APP_MODE=ci
```

**Flow:**
1. POST /api/upload → **blocks until indexing completes**
2. Indexing happens synchronously (inline)
3. Returns with `status="indexed"` immediately
4. POST /api/query → returns evidence (no polling needed)

**Response:**
```json
{
  "status": "ok",
  "message": "2 file(s) uploaded and indexed successfully.",
  "documents": [
    {
      "id": 123,
      "filename": "policy.txt",
      "status": "indexed",
      "created_at": "2025-12-17T10:30:00"
    }
  ]
}
```

## Implementation Details

### Detection Logic

In [main.py](main.py):

```python
# Detect if we're using InlineTaskRunner (CI mode or testing)
is_inline_mode = hasattr(runtime.task_runner, 'submit') and not callable(runtime.task_runner)
if is_inline_mode:
    logger.info("InlineTaskRunner detected - indexing will complete synchronously")
```

### Execution

```python
# Submit indexing task
task_runner.submit(
    process_document_background,
    doc_id=doc_id,
    tenant_id=tenant_id,
    file_path=saved_path,
    filename=file.filename,
    db_session_factory=runtime.get_db_session
)

# With InlineTaskRunner, task is complete now - refresh document status
if is_inline_mode and doc_record and db:
    db.refresh(doc_record)  # Get updated status="indexed"
    # Update response with fresh status
    for doc_dict in uploaded_docs:
        if doc_dict.get("id") == doc_record.id:
            doc_dict["status"] = doc_record.status
            logger.info(f"Document {doc_record.id} indexed synchronously: status={doc_record.status}")
            break
```

### Response Message

```python
# Build appropriate response message
if is_inline_mode:
    message = f"{len(files)} file(s) uploaded and indexed successfully."
else:
    message = f"{len(files)} file(s) uploaded. Processing in background."
```

## Testing

### Integration Test

See [test_integration.py](test_integration.py):

```python
def test_immediate_queryability_after_upload(client: TestClient, sample_document: str):
    """
    Test that documents are immediately queryable after upload in CI/inline mode.
    
    Verifies:
    - Upload completes synchronously (InlineTaskRunner)
    - Document status is "indexed" immediately after upload
    - Chunks are persisted to vector store before upload returns
    - Query immediately after upload returns evidence (evidence_count > 0)
    - No polling or waiting required between upload and query
    """
    # Upload document (should complete indexing synchronously)
    upload_response = client.post("/api/upload", files=files, headers=headers)
    upload_data = upload_response.json()
    
    # Verify synchronous completion
    assert "indexed successfully" in upload_data["message"]
    assert upload_data["documents"][0]["status"] == "indexed"
    
    # Query IMMEDIATELY after upload (no time.sleep needed)
    query_response = client.post("/api/query", json={"question": "..."}, headers=headers)
    
    # Verify evidence was retrieved
    debug_data = parse_sse_events(query_response.text)[0]["data"]
    assert debug_data["evidence_count"] > 0  # Evidence available immediately
```

### Test Output

```bash
$ pytest test_integration.py::test_immediate_queryability_after_upload -v -s

test_integration.py::test_immediate_queryability_after_upload 
✓ Immediate queryability test passed
  - Document indexed synchronously: policy.txt
  - Evidence count: 1
  - Retrieved chunks: 5
  - Selected chunks: 5
PASSED
```

## Benefits

### For Testing

1. **Deterministic**: No race conditions or timing issues
2. **Fast**: No polling loops or arbitrary sleep delays
3. **Reliable**: Upload + query in same test case works consistently
4. **Simple**: No async polling logic needed in tests

### For Production

1. **Non-blocking**: HTTP response returns immediately
2. **Scalable**: Background tasks don't block request threads
3. **Resilient**: Failed indexing doesn't block upload response
4. **Monitorable**: Can track indexing progress via status endpoint

## Configuration

### Enable Synchronous Mode

**Via Environment Variables:**
```bash
# Any of these will trigger InlineTaskRunner
export CI=true
export CI=1
export CI=yes
export APP_MODE=ci
```

**Via AppRuntime:**
```python
from app.runtime import build_test_runtime

# Test runtime uses InlineTaskRunner
runtime = build_test_runtime()
app.runtime = runtime
```

### Disable Synchronous Mode (Production)

**Default behavior:**
```bash
# No CI environment variable
# OR
export CI=false
export APP_MODE=production
```

**Via AppRuntime:**
```python
from app.runtime import build_runtime_from_env

# Production runtime uses BackgroundTaskRunner factory
runtime = build_runtime_from_env()
```

## Related Documentation

- [CI_MODE.md](CI_MODE.md) - CI mode auto-configuration
- [TASKRUNNER_IMPLEMENTATION.md](TASKRUNNER_IMPLEMENTATION.md) - Task runner abstraction
- [TEST_ISOLATION.md](TEST_ISOLATION.md) - Test isolation strategy
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing guide

## Migration Notes

### For Existing Tests

**Before** (with polling):
```python
# Upload
upload_resp = client.post("/api/upload", ...)
doc_id = upload_resp.json()["documents"][0]["id"]

# Poll until indexed
for _ in range(10):
    status_resp = client.get(f"/api/documents/{doc_id}/status", ...)
    if status_resp.json()["status"] == "indexed":
        break
    time.sleep(0.5)

# Query
query_resp = client.post("/api/query", ...)
```

**After** (synchronous):
```python
# Upload (blocks until indexed)
upload_resp = client.post("/api/upload", ...)
assert upload_resp.json()["documents"][0]["status"] == "indexed"

# Query immediately (no polling needed)
query_resp = client.post("/api/query", ...)
```

### For Production Code

No changes needed - production mode continues to use background tasks automatically.

## FAQ

**Q: How do I know which mode is active?**

A: Check the upload response message:
- Synchronous: `"X file(s) uploaded and indexed successfully."`
- Async: `"X file(s) uploaded. Processing in background."`

**Q: Can I force synchronous mode in production?**

A: Not recommended. Set `CI=true` only in test environments. For production, use the status polling pattern.

**Q: What happens if indexing fails in synchronous mode?**

A: The exception propagates to the upload endpoint, returning 500 error with details. In async mode, the error is logged and document status is set to `"failed"`.

**Q: Does synchronous mode affect performance?**

A: Yes - upload requests take longer (blocks until indexing completes). This is acceptable in tests but not recommended for production.

**Q: How long does synchronous indexing take?**

A: Typically < 1 second for small documents in test mode (mock providers). Real embeddings/LLMs would take longer.
