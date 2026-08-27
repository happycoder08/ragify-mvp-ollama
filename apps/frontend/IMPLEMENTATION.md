# RAGify Frontend - Backend API Integration

## Summary

Successfully wired the frontend UI to the backend API using the exact contract specifications from:
- `src/contracts/openapi.json`
- `src/contracts/types.ts`
- `src/contracts/sse.md`

## Files Created

### Core API Implementation

1. **src/api.ts** - API client for REST endpoints
   - `login()` - POST /api/login
   - `listDocuments()` - GET /api/documents
   - `uploadDocuments()` - POST /api/upload
   - Uses `VITE_API_URL` environment variable
   - Implements proper error handling with ErrorResponse type

2. **src/sse.ts** - SSE streaming client for /api/query
   - `queryWithSSE()` - Implements exact SSE parsing from sse.md
   - Parses event/data lines exactly as specified
   - Event types: debug → token → final
   - Includes 30-second timeout protection
   - `useSSEQuery()` - React hook for SSE streaming

### UI Pages

3. **src/pages/Login.tsx** + Login.css
   - Login form using contract LoginRequest/LoginResponse types
   - Stores JWT token via auth utilities
   - Navigates to /docs after successful login

4. **src/pages/Docs.tsx** + Docs.css (updated)
   - Document table with columns: filename, status, uploaded_at, error
   - File upload with drag-and-drop support
   - Smart polling: polls every 2s ONLY when status="pending"
   - Stops polling when all docs are terminal OR after 60s
   - Top banner: "Indexing in progress..." or "Ready to query"
   - Uses contract types: DocumentRecord, UploadResponse, DocumentsListResponse

5. **src/pages/Query.tsx** + Query.css
   - RAG query interface with SSE streaming
   - Shows streaming answer token-by-token
   - Displays evidence, sources, and debug info
   - Handles refusals properly
   - Uses contract types: QueryRequest, QueryFinalResponse, DebugInfo

### Updated Files

6. **src/App.tsx** + App.css
   - React Router with protected routes
   - Navigation bar with Query/Documents links
   - Logout functionality
   - Routes: /login, /query, /docs

7. **src/utils/auth.ts** (verified compatible)
   - Already matches LoginResponse structure
   - JWT token storage in localStorage

## Contract Compliance

✅ **API Fields**: Only uses fields defined in types.ts
✅ **SSE Parsing**: Exact implementation from sse.md
✅ **Document Status**: Only "pending" | "indexed" | "failed" (no "indexing")
✅ **Environment**: Uses VITE_API_URL for base URL
✅ **No Infinite Loops**: Polling stops after 60s or when all docs terminal
✅ **Contract Files**: Not modified

## Polling Logic

The Docs page implements safe polling:
1. Polls every 2 seconds ONLY when `status === "pending"`
2. Stops when all documents reach terminal state (indexed/failed)
3. 60-second timeout with message display
4. Cleanup on component unmount (clearInterval)

## SSE Implementation

The SSE parser follows sse.md exactly:
1. Parses "event: <type>" and "data: <json>" lines
2. Event flow: debug → token* → final
3. Token events have shape `{ t: string }`
4. Final event has QueryFinalResponse shape
5. 30-second timeout protection
6. Proper error handling

## Environment Configuration

**.env**
```
VITE_API_URL=http://localhost:8000
```

## Routes

- `/login` - Public login page
- `/query` - Protected RAG query interface
- `/docs` - Protected document management
- `/` - Redirects to /query

## Usage

1. Start the dev server: `npm run dev`
2. Login with backend credentials
3. Upload documents at /docs
4. Query documents at /query
5. Watch SSE streaming in real-time

## Type Safety

All API calls use exact contract types:
- LoginRequest/LoginResponse
- DocumentRecord/DocumentsListResponse/UploadResponse
- QueryRequest/QueryFinalResponse
- DebugInfo/EvidenceItem/SourceItem
- ErrorResponse

No guessing or assumptions about API fields.
