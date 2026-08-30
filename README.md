# RAGify AI Monorepo

RAGify is a full-stack document intelligence platform for grounded, multi-tenant question answering over uploaded knowledge bases. The repository combines a FastAPI backend for ingestion, retrieval, grounding, and orchestration with a React 18 + TypeScript + Vite frontend for interactive Q&A, evidence browsing, and result review.

## Monorepo Structure

```text
.
├── package.json              # root orchestration and shared scripts
├── README.md                 # project overview and developer workflow
├── apps/
│   ├── backend/              # FastAPI + Python service, ChromaDB, Postgres, pytest
│   │   ├── app/              # runtime, auth, schema, config, retrieval, guardrails
│   │   ├── main.py           # API entrypoint and server startup
│   │   ├── requirements.txt  # backend dependencies
│   │   ├── tests/            # pytest suite
│   │   └── ...               # ingestion scripts, diagnostics, demo assets
│   └── frontend/             # React 18 + TypeScript + Vite client
│       ├── src/              # UI pages, components, API client, SSE integration
│       ├── package.json      # frontend scripts and dependencies
│       └── ...               # Vitest tests, CSS, Vite config
└── docker-compose.yml        # optional local infrastructure definition
```

### Backend architecture (`apps/backend`)
- FastAPI application with authenticated APIs, upload pipeline, and query orchestration
- PostgreSQL for metadata, auth, tenant config, conversation records, and document status
- ChromaDB for vector persistence under `apps/backend/vectorstore/`
- Pytest-based validation for unit, integration, ground-truth, and retrieval tests
- Ollama and optional OpenAI/mock providers for embeddings and chat responses

### Frontend architecture (`apps/frontend`)
- React 18 application with TypeScript and Vite
- Vitest suite for UI and utility validation
- SSE client for streaming answer tokens from the backend
- Interactive evidence display, document selection, mode badges, and copy support

## Product Capabilities

### Backend capabilities
- Multi-tenancy with tenant-scoped document access and configuration
- Grounding Gate that refuses or constrains answers when evidence is insufficient
- MMR (Maximal Marginal Relevance) selection to diversify retrieved chunks
- Deterministic extractors for structured facts and common operational questions
- Question classification for broad vs. narrow queries and clarifications
- PDF, DOCX, and TXT ingestion with chunking, indexing, and evidence tracking
- Streaming query responses with sources, evidence snippets, and debug metadata

### Frontend controls and UX
- Selected Docs filter: All docs vs Selected docs radio toggle
- Document multi-select appears when Selected docs is enabled
- Evidence Panel toggles to show just the top chunk or all evidence items
- Answer Mode badges: `EXTRACTED`, `CITED`, and `NOT FOUND`
- Copy answer button for the final response, sources, and evidence details
- Demo-mode UI controls and dev-only diagnostics gated by environment flags

## Local Development Workflow

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- PostgreSQL available at `localhost:5432`
- Ollama running locally with `nomic-embed-text` and a chat model such as `llama3` or `llama3.2:1b`

### Windows PowerShell

```powershell
# from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r apps/backend/requirements.txt
npm install

# optional: start PostgreSQL with Docker
# docker run --name ragify-postgres -e POSTGRES_USER=ragify -e POSTGRES_PASSWORD=ragify -e POSTGRES_DB=ragify_db -p 5432:5432 -d postgres:15

# start backend + frontend together
npm run dev
```

### Unix/macOS

```bash
# from the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/backend/requirements.txt
npm install

# optional: start PostgreSQL with Docker
# docker run --name ragify-postgres -e POSTGRES_USER=ragify -e POSTGRES_PASSWORD=ragify -e POSTGRES_DB=ragify_db -p 5432:5432 -d postgres:15

# start backend + frontend together
npm run dev
```

### Root-level execution model
The root script in [package.json](package.json) runs both services together with `concurrently`:

```bash
npm run dev
```

This starts:
- Backend: `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Frontend: `npm --prefix apps/frontend run dev`

Local endpoints:
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

## Environment Variables and Runtime Modes

### Primary environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAGIFY_MODE` | `demo` | App execution mode: `dev`, `demo`, or `prod` |
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama`, `openai`, or `mock` |
| `DATABASE_URL` | `postgresql://ragify:ragify@localhost:5432/ragify_db` | PostgreSQL connection string |
| `VITE_DEMO_MODE` | `false` | Enables demo-mode UI behavior in the frontend |
| `LLM_MODEL` | `llama3.2:1b` | Chat model for the active provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `JWT_SECRET_KEY` | `your-secret-key-change-in-production` | Token signing secret |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRY_HOURS` | `24` | Token lifetime |

### Execution modes
- `dev`: full diagnostic visibility and more permissive development tuning
- `demo`: optimized for safe, fast demonstrations with constrained output and retrieval behavior
- `prod`: production-oriented defaults with stricter operational tuning

Examples:

```bash
# local development mode
RAGIFY_MODE=dev npm run dev

# demo mode
RAGIFY_MODE=demo npm run dev

# production mode
RAGIFY_MODE=prod npm run dev
```

Frontend demo mode example:

```bash
VITE_DEMO_MODE=true npm run dev
```

## Degraded State Behavior

When PostgreSQL is disconnected or `DATABASE_URL` is invalid, the backend enters a degraded but non-fatal state. Startup logs warn that the app will continue running, and initialization proceeds with a warning if the database is unavailable.

Operational implications:
- App startup continues rather than crashing the process
- DB-backed metadata, document status, and conversation persistence may be unavailable
- ChromaDB and the configured LLM provider can still keep the RAG pipeline functional if they are healthy
- Access patterns that rely on database-backed tenant state may return partial data or `503` responses

This behavior is intentional for local development and demo environments where vector search and LLM access remain usable while Postgres is temporarily offline.

## Testing Matrix

### Backend Pytest

```bash
# run the backend suite
pytest apps/backend -q

# deterministic mock-mode backend test run
LLM_PROVIDER=mock pytest apps/backend -q

# specific integration test with mock provider
LLM_PROVIDER=mock pytest apps/backend/test_integration.py -v
```

### Frontend Vitest

```bash
# run the frontend suite
npm --prefix apps/frontend run test

# one-shot CI-style run
npm --prefix apps/frontend run test -- --run
```

## Useful Commands

```bash
# backend only
npm run dev:backend

# frontend only
npm run dev:frontend

# frontend production build
npm run build

# backend tests
npm run test:backend

# frontend Vitest
npm run test:frontend
```

## Notes

- The standard developer workflow is the root `npm run dev` command.
- The frontend proxies API requests to the backend at `http://localhost:8000` during development.
- The backend owns ingestion, indexing, grounding, and retrieval; the frontend owns interaction design, evidence display, and user assistance.
- For deterministic local automation with no Ollama dependency, use `LLM_PROVIDER=mock`.

## 📚 Additional Documentation

- [apps/backend/README.md](apps/backend/README.md) - Backend architecture, APIs, runtime modes, and testing
- [apps/frontend/README.md](apps/frontend/README.md) - Frontend architecture, SSE flow, and UI controls
- [apps/backend/TESTING_GUIDE.md](apps/backend/TESTING_GUIDE.md) - Backend testing guide and validation workflow
- [apps/backend/SETUP_WITHOUT_DOCKER.md](apps/backend/SETUP_WITHOUT_DOCKER.md) - Alternative local setup instructions
- [apps/backend/archive/legacy-notes/PHASE1_SUMMARY.md](apps/backend/archive/legacy-notes/PHASE1_SUMMARY.md) - Historical implementation summary
- [apps/backend/archive/legacy-notes/CI_TESTING.md](apps/backend/archive/legacy-notes/CI_TESTING.md) - CI testing notes and mock-provider setup
- [apps/backend/archive/legacy-notes/GROUNDING_GATE_SUMMARY.md](apps/backend/archive/legacy-notes/GROUNDING_GATE_SUMMARY.md) - Grounding gate documentation
- [apps/backend/archive/legacy-notes/SSE_IMPLEMENTATION.md](apps/backend/archive/legacy-notes/SSE_IMPLEMENTATION.md) - Streaming implementation details

For implementation depth on the backend, see [apps/backend/README.md](apps/backend/README.md). For frontend UX, controls, and SSE flow, see [apps/frontend/README.md](apps/frontend/README.md).

