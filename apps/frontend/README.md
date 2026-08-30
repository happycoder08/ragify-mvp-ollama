# RAGify Frontend

This frontend is the user-facing layer for the RAGify monorepo. It is built with React 18, TypeScript, and Vite, and it connects to the FastAPI backend over the local development API at `http://localhost:8000`.

## Architecture

```text
User Browser
    │
    ▼
React 18 + TypeScript App (apps/frontend)
    │
    ├─ Query page / chat flow
    ├─ Docs page / selected document filtering
    ├─ Evidence panel / citation review
    ├─ Demo UI / readiness checks
    └─ SSE streaming client
    │
    ▼
FastAPI backend
    ├─ /api/query
    ├─ /api/upload
    ├─ /api/documents
    └─ /api/system/config
```

## Stack

- React 18
- TypeScript
- Vite for local development and build output
- Vitest for frontend unit testing
- SSE streaming client (`src/sse.ts`) for token-by-token answers

## Application Structure

```text
apps/frontend/
├── src/
│   ├── api.ts               # API helper functions
│   ├── sse.ts               # SSE query streaming client
│   ├── contracts/           # API response types
│   ├── pages/               # main pages and UI panels
│   ├── utils/               # answer mode, selection, auth helpers
│   ├── App.tsx              # route composition
│   └── main.tsx             # app bootstrap
├── package.json             # frontend scripts
├── vite.config.ts           # Vite config
├── vitest.config.ts         # Vitest config
└── README.md                # this file
```

## Quick Start

### Install and run from the repo root

```bash
npm install
npm run dev
```

### Run frontend only

```bash
npm --prefix apps/frontend run dev
```

### Production build

```bash
npm run build
```

### Preview production bundle

```bash
npm --prefix apps/frontend run preview
```

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_DEMO_MODE` | `false` | Enables demo-mode UI gating and hides dev-only banners while keeping core evidence features visible |
| `VITE_SHOW_DEVTOOLS` | `false` | Enables dev/debug UI overlays and counters |
| `VITE_DEMO_Q1` | empty | Demo question 1 for demo mode |
| `VITE_DEMO_Q2` | empty | Demo question 2 for demo mode |

## Feature Summary

### Query experience
- Question input and streaming answer display
- SSE-backed answer rendering for token-by-token output
- Final answer summary with evidence and sources

### UI controls
- Selected Docs filter: radio toggle for All docs vs Selected docs
- Multi-select of uploaded documents when Selected docs is active
- Evidence Panel toggle to show only the top evidence item or all evidence items
- Answer Mode badge display: `EXTRACTED`, `CITED`, and `NOT FOUND`
- Copy answer utility that copies the answer plus source and evidence context

### Demo and debug behavior
- Demo mode allows curated question quick actions for presentations
- Dev-only banners and mismatch warnings are hidden in demo mode but still remain in the logic
- `?debug=1` can enable tooltips and diagnostic output in development builds

## Key UI Logic

The main interactive page lives in `src/pages/Query.tsx` and includes:

- selected document filtering
- stream handling and answer accumulation
- evidence display and source rendering
- answer mode classification
- copy-to-clipboard interaction

The answer mode logic is computed in `src/utils/computeAnswerMode.ts`:
- `NOT_FOUND` when the system refuses or no support is found
- `EXTRACTED` when the pipeline marker starts with `EXTRACTOR_`
- `CITED` for normal citations-backed answers

## Testing Matrix

Run the frontend suite:

```bash
npm --prefix apps/frontend run test
```

Run it in one-shot CI mode:

```bash
npm --prefix apps/frontend run test -- --run
```

## Development Notes

- The frontend is expected to run alongside the backend via the root `npm run dev` command.
- The app uses the backend’s `POST /api/query` flow and streams final results via SSE.
- When `VITE_DEMO_MODE=true`, the UI is tuned for presentation without removing core evidence logic.

For backend setup and retrieval behavior, see [apps/backend/README.md](apps/backend/README.md). For the monorepo-wide workflow, see the root [README.md](../../README.md).
