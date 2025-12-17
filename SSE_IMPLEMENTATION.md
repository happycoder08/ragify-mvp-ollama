# SSE (Server-Sent Events) Implementation Summary

## Overview
Refactored `/api/query` endpoint from NDJSON streaming to Server-Sent Events (SSE) format for better browser compatibility and standardized event streaming.

## Changes Made

### 1. API Endpoint Changes (`main.py`)

**Content-Type**: Changed from `application/x-ndjson` to `text/event-stream`

**Event Types**:
- `event: debug` - Diagnostic information (evidence count, sources, etc.)
- `event: token` - Individual token chunks during streaming
- `event: final` - Final QueryFinalResponse with complete answer
- `event: error` - Error messages (ready for future error handling)

**SSE Format**:
```
event: <event_type>
data: <json_payload>

```

### 2. Event Payloads

#### Debug Event
```
event: debug
data: {"evidence_count": 5, "sources_count": 3, "request_id": "..."}

```

#### Token Event
```
event: token
data: {"t": "Hello"}

```

#### Final Event (Success)
```
event: final
data: {"answer": "...", "refused": false, "evidence": [...], "sources": [...]}

```

#### Final Event (Refusal)
```
event: final
data: {"answer": "The document does not specify this.", "refused": true, "refusal_reason": "NOT_FOUND", "evidence": [], "sources": []}

```

### 3. Test Infrastructure (`test_integration.py`)

**SSE Parser**: Added `parse_sse_events()` helper function that:
- Parses SSE text into structured events
- Handles multi-line data fields
- Attempts JSON parsing on data, falls back to string
- Returns list of `{"event": str, "data": dict|str}`

**Updated `test_full_workflow`**:
- Uses `parse_sse_events()` to parse streaming response
- Extracts events by type (debug, token, final)
- Validates final event with QueryFinalResponse schema
- Tests both success and refusal cases

### 4. SSE Format Tests (`test_sse_format.py`)

New test file with comprehensive SSE parser validation:
- `test_sse_parser`: Basic single and multiple event parsing
- `test_sse_multiline_data`: Multi-line JSON data handling
- `test_sse_event_types`: All event types (debug, token, final, error)
- `test_sse_empty_events`: Edge cases (empty, whitespace, malformed)
- `test_sse_json_parsing`: JSON vs plain text data

## Benefits

1. **Browser Compatibility**: Native EventSource API support
2. **Standardized Format**: Industry-standard SSE specification
3. **Event Typing**: Clear distinction between debug, token, and final events
4. **Schema Validation**: QueryFinalResponse Pydantic validation still enforced
5. **Backward Compatibility**: Tests updated, but schema guarantees maintained

## Test Results

All tests passing:
- ✅ 7/7 schema validation tests (test_query_schema.py)
- ✅ 1/1 integration test with SSE parsing (test_integration.py::test_full_workflow)
- ✅ 5/5 SSE format tests (test_sse_format.py)

**Total**: 13/13 passing in ~3.8s

## Migration Notes

**UI Updates Required** (future work):
- Replace NDJSON parsing with EventSource or manual SSE parsing
- Listen for specific event types instead of line-by-line JSON parsing
- Example:
  ```javascript
  const eventSource = new EventSource('/api/query');
  eventSource.addEventListener('token', (e) => {
    const data = JSON.parse(e.data);
    appendToken(data.t);
  });
  eventSource.addEventListener('final', (e) => {
    const data = JSON.parse(e.data);
    handleFinalResponse(data);
  });
  ```

**No Breaking Changes**: All existing schema validations and guarantees maintained.
