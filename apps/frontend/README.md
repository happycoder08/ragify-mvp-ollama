# RAGify Frontend (React + TypeScript + Vite)

This app contains the RAGify frontend built with React, TypeScript, and Vite. It is part of the root monorepo and proxies `/api` and `/health` requests to the FastAPI backend at `http://localhost:8000` during development.

Quick start
```
# from the repository root
Push-Location apps/frontend
npm install

# start dev server with HMR
npm run dev

# build production bundle
npm run build

# preview production build locally
npm run preview

# run unit tests (Vitest)
npx vitest run
Pop-Location

# Or run the frontend from the root
npm --prefix apps/frontend run dev
```

Environment flags
- `VITE_DEMO_MODE=true` — enable demo mode UI (hides dev-only diagnostic banners but keeps Evidence/Sources visible). Can be set in a `.env.local` or exported before `npm run dev`.
- `VITE_SHOW_DEVTOOLS=true` — enable dev tooling banners and debug counters. Also can be toggled via `?debug=1` URL param during development.

Frontend architecture & behavior
- Pages: `src/pages/` contains the main views. `Query.tsx` is the primary interactive page for asking questions.
- SSE: `src/sse.ts` provides `queryWithSSE` used by `Query.tsx` to stream tokens and receive final responses.
- Types: API types are in `src/contracts/types.ts` and used across components.

Key UI controls added
- Demo Control Bar (in `Query.tsx`, above the question input):
  - Radio toggle: "All docs" vs "Selected docs".
  - When "Selected docs" is chosen a multi-select appears listing uploaded documents (populated from `listDocuments`).
  - On submit the request includes `doc_ids` only when "Selected docs" is enabled and at least one document is selected.

- Evidence display (in `Query.tsx` + `EvidencePanel`):
  - Default shows only the top evidence chunk.
  - Small toggle link "Show all evidence (N)" reveals all evidence items (no internal changes to `EvidencePanel`).

- Answer Mode badge (in `Query.tsx` header):
  - `NOT FOUND` when `refused === true`.
  - `EXTRACTED` when `debugInfo.pipeline_marker` starts with `EXTRACTOR_`.
  - `CITED` otherwise.

- Copy answer (in `Query.tsx` below the answer):
  - "Copy answer" button copies `answerText`, appends source filenames and evidence headings, using `navigator.clipboard` and shows a transient "Copied!" indicator.

Developer UX and gating
- Dev-only banners (invariant checks, mismatch warnings, and debug counters) are gated by the `VITE_DEMO_MODE` flag — demo mode hides these while preserving the underlying dev checks and logic.

Testing
- Unit tests live under `src/pages/__tests__/`. Run them with `npx vitest run`.

Notes
- Styling for new controls uses existing theme classes in `src/pages/Query.css` to keep visual consistency (e.g. `.clear-button`, `.inline-link-btn`).
- No dev code was removed — rendering is gated only, so toggles and flags control visibility.

If you'd like, I can add a short "Frontend quick reference" section listing component entry points and data flow diagrams.
