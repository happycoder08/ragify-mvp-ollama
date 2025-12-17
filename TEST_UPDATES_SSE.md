# Test Updates for SSE Response Contract

## Summary
Updated all integration tests to use the new Server-Sent Events (SSE) response format from `/api/query` endpoint. Tests now parse SSE events and validate the `final` event against minimal contract requirements.

## Changes Made

### 1. Added SSE Parser Import
- All tests now import and use `parse_sse_events()` helper function
- Parses SSE text format into structured events list

### 2. Updated Response Parsing Pattern

**Old Pattern (NDJSON)**:
```python
response_lines = query_response.text.strip().split("\n")
answer_line = json.loads(response_lines[1])
metadata_line = json.loads(response_lines[2])
```

**New Pattern (SSE)**:
```python
events = parse_sse_events(query_response.text)
final_events = [e for e in events if e["event"] == "final"]
final_data = final_events[0]["data"]
```

### 3. Minimal Assertions (Per Requirements)

All tests now assert minimally on the `final` event:

**For Any Response**:
- ✅ `final.answer` exists (string)
- ✅ `final.refused` is boolean

**For Success (refused=false)**:
- ✅ `final.evidence` is non-empty list
- ✅ Evidence items have required fields (via Pydantic)

**For Refusal (refused=true)**:
- ✅ `final.answer == "The document does not specify this."` (canonical message)
- ✅ `final.refusal_reason == "NOT_FOUND"`
- ✅ `final.evidence` is empty list

### 4. Fixed KeyError Issues
- Changed `upload_data["uploaded"]` → `upload_data["documents"]`
- Matches actual API response structure

### 5. Removed Document Content Hardcoding
- Tests no longer hardcode specific onboarding document content
- Use minimal assertions on structure, not content
- Example: Instead of checking for "15 vacation days", just check `len(evidence) > 0`

### 6. Removed Polling/Sleep Calls
- Removed `time.sleep(0.5)` calls (unnecessary with InlineTaskRunner)
- InlineTaskRunner executes immediately, no background delay

### 7. Added Missing Import
- Added `import time` for any tests that needed it (test_tenant_isolation)

## Updated Tests

### test_integration.py
1. ✅ `test_full_workflow` - Already updated (reference implementation)
2. ✅ `test_unrelated_query_refusal` - SSE parsing, minimal assertions
3. ✅ `test_tenant_isolation` - SSE parsing, fixed KeyError, removed sleep
4. ✅ `test_empty_collection_query` - SSE parsing, canonical refusal check
5. ✅ `test_multiple_documents` - SSE parsing, fixed KeyError, removed sleep
6. ✅ `test_authentication_failures` - No changes needed (doesn't parse response)
7. ✅ `test_ungrounded_answer_validation` - SSE parsing, minimal validation

## Test Results

**Before Updates**: 5 failed, 2 passed
- JSONDecodeError (SSE format not recognized)
- KeyError: 'uploaded' (wrong field name)
- NameError: 'time' not imported

**After Updates**: 7 passed, 0 failed ✅

**Full Suite**: 19/19 passing in ~10s
- 7 schema validation tests
- 7 integration tests  
- 5 SSE format tests

## Benefits

1. **Contract Compliance**: Tests validate SSE format and Pydantic schemas
2. **Minimal Coupling**: No hardcoded document content, flexible assertions
3. **Fast & Reliable**: No polling, deterministic with InlineTaskRunner
4. **Future-Proof**: Tests validate structure, not specific content

## Example Test Pattern

```python
def test_example(client, sample_document):
    # Login
    login_response = client.post("/api/login", json={"username": "alice", "password": "alice123"})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Upload
    files = {"files": ("doc.txt", sample_document.encode(), "text/plain")}
    upload_response = client.post("/api/upload", files=files, headers=headers)
    assert upload_response.json()["documents"][0]["status"] == "indexed"
    
    # Query
    query_response = client.post("/api/query", json={"question": "Test?"}, headers=headers)
    
    # Parse SSE
    events = parse_sse_events(query_response.text)
    final_events = [e for e in events if e["event"] == "final"]
    final_data = final_events[0]["data"]
    
    # Minimal assertions
    assert isinstance(final_data["refused"], bool)
    
    if not final_data["refused"]:
        assert len(final_data["evidence"]) > 0  # Has evidence
    else:
        assert final_data["answer"] == "The document does not specify this."  # Canonical refusal
        assert final_data["refusal_reason"] == "NOT_FOUND"
```

## Migration Checklist for Other Tests

If updating other test files that use `/api/query`:

- [ ] Import `parse_sse_events` from `test_integration`
- [ ] Replace NDJSON line parsing with `parse_sse_events(response.text)`
- [ ] Extract `final` event: `[e for e in events if e["event"] == "final"][0]["data"]`
- [ ] Update assertions to use `final_data["answer"]`, `final_data["refused"]`, etc.
- [ ] Change `upload_data["uploaded"]` → `upload_data["documents"]`
- [ ] Remove hardcoded content checks (e.g., specific vacation day numbers)
- [ ] Assert on structure (fields exist, types correct) not content
- [ ] Remove `time.sleep()` calls (InlineTaskRunner is synchronous)
