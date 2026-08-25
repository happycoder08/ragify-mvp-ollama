# Debug Mode Implementation Summary

## Changes Made

Added `debug` parameter to `/api/query` endpoint that provides detailed retrieval diagnostics.

### API Changes

**Request**: Added optional `debug` parameter to `QueryRequest` model
```python
{
  "question": "What time should I arrive?",
  "debug": 1  // 0 = off (default), 1 = detailed diagnostics
}
```

**Response**: Enhanced debug object in NDJSON stream

**debug=0 (default, legacy mode)**:
```json
{
  "debug": {
    "evidence_count": 3,
    "sources_count": 5,
    "selected_chunks": [...],  // basic chunk info
    "context": "full context text"
  }
}
```

**debug=1 (detailed diagnostics)**:
```json
{
  "debug": {
    "evidence_count": 3,
    "sources_count": 5,
    "retrieved_count": 20,    // NEW: total chunks retrieved from ChromaDB
    "selected_count": 5,      // NEW: chunks selected after reranking/filtering
    "selected_chunks": [      // NEW: detailed chunk diagnostics
      {
        "id": "Employee_Onboarding_Guide.txt_30",
        "header": "1. ARRIVE AT THE OFFICE (8:00 AM)",
        "snippet": "1. ARRIVE AT THE OFFICE (8:00 AM)    - Report to the main reception...",
        "distance": 382.6118
      },
      // ... more chunks
    ]
  }
}
```

### Code Changes

1. **main.py**:
   - Added `debug: int = 0` to `QueryRequest` model (line ~205)
   - Pass debug flag to `answer_question()` (line ~590)
   - Enhanced debug object construction in `stream_response()` (lines ~600-615)

2. **app/services/rag_service.py**:
   - Updated `query_collection()` signature to accept `debug` parameter (line ~397)
   - Changed return type from `List[Dict[str, Any]]` to `Dict[str, Any]` for debug info
   - Build detailed chunk diagnostics when `debug >= 1` (lines ~750-770)
   - Return enhanced debug object with `retrieved_count`, `selected_count`, and chunk details
   - Updated `answer_question()` wrapper to pass debug flag (line ~930)

3. **eval/run_eval.py**:
   - Enable debug mode in queries: `"debug": 1` (line ~53)
   - Display retrieval diagnostics in output (lines ~175-178)

### Testing

Created `test_debug_mode.py` to verify:
- ✅ debug=0 works (legacy mode, no retrieved/selected counts)
- ✅ debug=1 works (detailed diagnostics included)
- ✅ Chunk details include id, header, snippet, distance

Test output:
```
[2] Testing with debug=1 (detailed diagnostics)...
  Debug object keys (debug=1): ['evidence_count', 'sources_count', 'retrieved_count', 'selected_count', 'selected_chunks']
  - evidence_count: 3
  - sources_count: 5
  - retrieved_count: 20   <-- Total retrieved from ChromaDB
  - selected_count: 5     <-- After hybrid reranking
  
  First chunk details:
    - id: Employee_Onboarding_Guide.txt_30
    - header: 1. ARRIVE AT THE OFFICE (8:00 AM)
    - snippet: 1. ARRIVE AT THE OFFICE (8:00 AM)    - Report to the main reception on the 3rd floor...
    - distance: 382.6118
```

### Usage

**API Request**:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What documents do I need?",
    "debug": 1
  }'
```

**Eval Harness**:
The eval script now automatically uses `debug=1` and displays retrieval diagnostics:
```
   [1/15] What time should I arrive on my first day?
   FAIL | Keywords: ['8:00']/['8:00', '8 am'] | Evidence: 3 snippets
   Retrieval: 20 retrieved -> 5 selected
```

### Benefits

1. **Retrieval Transparency**: See exactly how many chunks were retrieved and selected
2. **Distance Debugging**: View similarity distances to understand relevance filtering
3. **Content Inspection**: Preview chunk headers and snippets without full context dump
4. **Pipeline Validation**: Verify retrieval → reranking → selection pipeline working
5. **Non-Breaking**: debug=0 maintains backward compatibility with existing clients

### Notes

- Debug mode works even when LLM is skipped (empty results, grounding refusal)
- Retrieval diagnostics help identify if failures are due to:
  - **No retrieval**: `retrieved_count=0` → indexing issue
  - **Poor reranking**: `retrieved_count=20, selected_count=0` → reranking too strict
  - **Wrong content**: High `distance` values → embedding/chunking issues
- Chunk snippets are truncated to 200 chars to keep response size manageable
