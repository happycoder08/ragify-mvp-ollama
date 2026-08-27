# API Contract Audit

Date: 2026-08-27

## Scope

Compared the FastAPI routes in `apps/backend/main.py` and schemas in `apps/backend/app/` with the active frontend client in `apps/frontend/src/api.ts`, contracts in `apps/frontend/src/contracts/types.ts`, and the SSE parser in `apps/frontend/src/sse.ts`.

## Findings and Fixes

### Authentication

`POST /api/login` matches. The request uses `username` and `password`; the response contains `access_token`, `token_type`, and `tenant_id`.

### Upload and documents

`POST /api/upload` matches the multipart field name `files` and returns `status`, `message`, `documents`, `files_processed`, and `files_with_db_record`.

`GET /api/documents` returns `{ documents: [...] }`. Document records include the fields used by the frontend. The TypeScript status union now includes the backend's transient `indexing` state.

The existing purge, status, and reindex response types remain compatible. The active API client exposes purge; status and reindex helpers are not currently needed by the UI.

### Query

The request fields match, including `question`, `top_k`, `mode`, `conversation_id`, `conversation_history`, `doc_ids`, `debug`, and `stream`.

The backend response includes fields that were missing from the frontend contracts:

- `pipeline_marker`
- `needs_clarification`
- `clarification`
- `EvidenceItem.anchor_detected`
- Extended debug diagnostics such as `context_length`, `retrieved_chunks_top20`, `debug_trace`, refusal metadata, and invariant flags

These fields were added to `apps/frontend/src/contracts/types.ts`. `QueryResponse` is now an alias of the canonical `QueryFinalResponse`.

### SSE

The backend emits the following frames:

- `event: debug` with a JSON `DebugInfo` payload
- `event: token` with `{ "t": "..." }`
- `event: final` with a JSON `QueryFinalResponse` payload

`queryWithSSE` parses event blocks, supports CRLF and multiple `data:` lines, dispatches all three event types, and stops on `final`. The wire format is compatible.

The client previously defaulted to `http://localhost:8000`, which bypassed the Vite proxy in local development. It now defaults to a relative URL and uses `VITE_API_URL` only when explicitly configured.

### Conversations

The backend supports create, list, fetch, add-message, and delete routes. The request fields match the frontend definitions. Backend conversation responses also include `message_count`, and conversation titles can be null; both are now represented in TypeScript.

The active API client did not expose conversation or guardrail helpers. Added typed helpers for all conversation operations and `GET /api/guardrails`.

### Guardrails

The frontend previously used incorrect names such as `max_files_per_upload`, `rate_limit_requests_per_hour`, and `rate_limit_mb_per_hour`. The backend returns `max_files_per_request`, `max_requests_per_minute`, `max_requests_per_hour`, `max_upload_mb_per_hour`, `llm_timeout_seconds`, and `upload_timeout_seconds`. The TypeScript contract now matches the backend.

## Remaining Notes

`apps/frontend/src/utils/api.ts` and `apps/frontend/src/types/api.ts` contain an older duplicate client/type layer. The active application imports `src/api.ts`, `src/sse.ts`, and `src/contracts/types.ts`; the duplicate layer is not imported by the current pages. It should either be removed or migrated in a later cleanup to avoid two contract sources.

## Validation

- Frontend unit tests: 10 passed in `Query.test.tsx`
- Frontend production build: passed with the existing Vite dynamic-import chunk warning
- Backend OpenAPI inspection: audited route registrations and `QueryRequest` schema
