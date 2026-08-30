# RAGify Backend

This backend powers the monorepo’s retrieval and grounding system. It exposes a FastAPI API for tenant-aware document ingestion, semantic search, evidence-backed Q&A, and runtime diagnostics. The service is designed to work with PostgreSQL for metadata, ChromaDB for vector retrieval, and an Ollama/OpenAI/mock LLM provider stack.

## Architecture

```text
Browser / Frontend
      │
      ▼
FastAPI Backend (apps/backend/main.py)
      │
      ├─ Auth + tenancy
      ├─ Upload + ingestion pipeline
      ├─ Retrieval + grounding gate
      ├─ Query orchestration + SSE output
      └─ Runtime diagnostics / health checks
      │
      ├─ PostgreSQL (metadata, docs, conversations)
      ├─ ChromaDB (vectorstore / embeddings)
      ├─ Ollama (embeddings + LLM)
      └─ OpenAI or Mock provider (optional)
```

## Backend Responsibilities

- FastAPI API surface for login, upload, document listing, query, debug, and admin endpoints
- Multi-tenancy with tenant-scoped access boundaries and configuration
- Document ingestion for PDF, DOCX, and TXT files
- Chunking, metadata extraction, and vector indexing in ChromaDB
- Grounding validation before final answer generation
- MMR selection, deterministic extractors, and question classification
- Pytest suite covering retrieval quality, integration flow, and grounding behavior

## Key Components

- `app/config.py` — runtime config and mode selection (`dev`, `demo`, `prod`)
- `app/database.py` — SQLAlchemy/PostgreSQL session management
- `app/auth.py` — authentication and tenant resolution
- `app/services/ingestion.py` — file reading, chunking, and indexing pipeline
- `app/services/rag_service.py` — retrieval orchestration, MMR selection, evidence extraction
- `app/services/grounding.py` — grounding gate and evidence scoring
- `main.py` — HTTP routes and app lifecycle

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL running locally or via Docker
- Ollama with `nomic-embed-text` and a chat model such as `llama3` or `llama3.2:1b`

### Install dependencies

```bash
# from the repo root
pip install -r apps/backend/requirements.txt
```

### Configure environment

```bash
# default local Postgres connection
DATABASE_URL=postgresql://ragify:ragify@localhost:5432/ragify_db
RAGIFY_MODE=demo
LLM_PROVIDER=ollama
```

### Start backend only

```bash
npm run dev:backend
```

The backend listens on:
- http://localhost:8000

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAGIFY_MODE` | `demo` | Runtime mode: `dev`, `demo`, or `prod` |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `mock` |
| `DATABASE_URL` | `postgresql://ragify:ragify@localhost:5432/ragify_db` | PostgreSQL connection string |
| `LLM_MODEL` | `llama3.2:1b` | Active model used by the configured provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `JWT_SECRET_KEY` | `your-secret-key-change-in-production` | JWT signing secret |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRY_HOURS` | `24` | Token lifetime |

## Runtime Modes

- `dev`: verbose diagnostics and broader development-oriented settings
- `demo`: safe and fast default for demonstrations
- `prod`: production-tuned settings and stricter behavior

## Degraded State Behavior

When PostgreSQL is unavailable, the app does not fail immediately. Startup logs warn that the backend is operating in a degraded state, while the service continues if ChromaDB and the LLM provider remain reachable.

Expected behavior:
- DB-backed metadata and document tracking may be unavailable
- Document listing and upload persistence may fail or return partial data
- Retrieval and Q&A may still work if ChromaDB and Ollama remain healthy
- protected endpoints may return 503 or partial responses depending on which layer is unavailable

## Core Backend Features

- Multi-tenancy and tenant config isolation
- Grounding Gate to prevent unsupported claims
- MMR selection for diverse context retrieval
- Deterministic extractors for structured fact extraction
- Question classification for broad vs. specific requests
- Conversation persistence and debug metadata
- Rate limiting, guardrails, and dependency health checks

## API Overview

### Public endpoints
- `POST /api/login` — authenticate and receive a JWT
- `GET /health` — health status for the service
- `GET /api/system/config` — active runtime configuration summary
- `GET /` — serves the app shell

### Protected endpoints
- `GET /api/config` — tenant-specific configuration
- `GET /api/guardrails` — guardrail limits for the tenant
- `GET /api/rate-limit-status` — current tenant usage
- `GET /api/debug` — runtime details and collection counts
- `GET /api/health/deps` — dependency health check
- `POST /api/upload` — upload and start indexing
- `POST /api/query` — grounded query with evidence and sources
- `GET /api/documents` — list documents for the tenant
- `POST /api/documents/purge` — reset tenant document state
- `GET /api/conversations` — list conversation history
- `POST /api/conversations` — create a conversation

## Testing Matrix

Run the backend suite:

```bash
pytest apps/backend -q
```

Run the suite with mock provider for deterministic local testing:

```bash
LLM_PROVIDER=mock pytest apps/backend -q
```

Run a focused integration test:

```bash
LLM_PROVIDER=mock pytest apps/backend/test_integration.py -v
```

## Suggested Local Workflow

```bash
# start PostgreSQL if needed
# docker run --name ragify-postgres -e POSTGRES_USER=ragify -e POSTGRES_PASSWORD=ragify -e POSTGRES_DB=ragify_db -p 5432:5432 -d postgres:15

# ensure Ollama models are available
ollama pull nomic-embed-text
ollama pull llama3

# start the app
npm run dev:backend
```

For the full stack, see the root [README.md](../../README.md) and the frontend docs in [../frontend/README.md](../frontend/README.md).

# Run mock tests (no external dependencies)
LLM_PROVIDER=mock pytest test_integration.py -v
```

### Test Coverage Areas
- ✅ Document upload and processing (PDF, DOCX, TXT)
- ✅ Vector indexing and retrieval
- ✅ Broad vs. specific question handling
- ✅ Adjacent chunk expansion for comprehensive answers
- ✅ Evidence citation and grounding validation
- ✅ Streaming response functionality
- ✅ Multi-tenant isolation
- ✅ Authentication and authorization
- ✅ Error handling and edge cases

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGIFY_MODE` | `demo` | Global mode: `dev`, `demo`, or `prod` (see CONFIG_GUIDE.md) |
| `DATABASE_URL` | `postgresql://ragify:ragify@localhost:5432/ragify_db` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | `your-secret-key-change-in-production` | Secret key for JWT signing |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRY_HOURS` | `24` | Token expiry time |
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama`, `openai`, `mock`) |
| `LLM_MODEL` | `llama3.2:1b` | LLM model name for the active provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `RAGIFY_OLLAMA_TIMEOUT` | `300` | Timeout for Ollama requests |
| `OPENAI_API_KEY` | - | OpenAI API key (if using `openai` provider) |
| `MOCK_UNGROUNDED` | `false` | Enable ungrounded answers for mock provider |
| `RAGIFY_MOCK` | `0` | Enable mock mode (no Ollama/Chroma) |
| `VECTOR_DIR` | `vectorstore/` | Directory where ChromaDB collections are persisted |
| `RERANKER_PROVIDER` | `none` | Optional reranker backend (`none`, `jina`, `cohere`) |
| `APP_MODE` | - | Set to `ci` to force CI/test wiring (mock providers, inline tasks) |

### Modes (dev / demo / prod)

RAGify centralizes most tuning in `RAGIFY_MODE` (see [CONFIG_GUIDE.md](CONFIG_GUIDE.md)):
- **dev**: full features, verbose logging, generous limits (unlimited tokens, more chunks), best for development and debugging.
- **demo** (default): optimized for fast, safe responses (smaller token budget, fewer chunks, stricter thresholds) for live demos.
- **prod**: balanced quality/speed with tighter defaults, reranking enabled by default, suitable for production.

You can inspect the active config at runtime via `GET /api/system/config`, and still override specific settings with env vars (for example `LLM_PROVIDER`, `LLM_MODEL`, or `RAGIFY_OLLAMA_TIMEOUT`).

For a concise, up-to-date summary of each mode, usage examples, and best practices, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

### LLM Provider Options

RAGify supports multiple LLM backends via the `LLM_PROVIDER` environment variable:

- **`ollama`** (default): Local Ollama instance
  - Requires running Ollama server
  - Models: `nomic-embed-text` (embeddings), `llama3` (chat)
  - Best for: Local development, privacy-focused deployments

- **`openai`**: OpenAI API
  - Requires `OPENAI_API_KEY` environment variable
  - Models: `text-embedding-3-small` (embeddings), `gpt-4` (chat)
  - Best for: Production deployments, cloud hosting

- **`mock`**: Mock provider for testing
  - No external dependencies required
  - Returns deterministic keyword-based responses
  - Supports grounded and ungrounded modes (`MOCK_UNGROUNDED=true`)
  - Best for: CI/CD pipelines, integration tests
  - See [CI Testing Guide](CI_TESTING.md) for details

**Example:**
```bash
# Use Ollama (default)
LLM_PROVIDER=ollama uvicorn main:app --reload

# Use OpenAI
export OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai uvicorn main:app --reload

# Use mock provider for testing
LLM_PROVIDER=mock pytest test_integration.py -v
```


### Adding New Tenants

Edit `app/tenant_config.py`:

```python
TENANT_CONFIGS = {
    "new_tenant": {
        "tenant_id": "new_tenant",
        "title": "New Tenant Name",
        "primary_color": "#6366f1",
        "logo_url": "https://example.com/logo.png",
        "disclaimer": "Custom disclaimer text"
    }
}
```

Then add user in `app/auth.py`:

```python
USERS_DB = {
    "new_user": {
        "username": "new_user",
        "hashed_password": bcrypt.hashpw("password".encode(), bcrypt.gensalt(12)).decode(),
        "tenant_id": "new_tenant"
    }
}
```

## 🎯 Advanced Capabilities

### Intelligent Question Handling

RAGify automatically adapts its retrieval strategy based on question type:

**Broad Questions** (e.g., "What do I do on my first day?")
- Expands context with adjacent chunks (±2 indices)
- Includes "First Day" and "Checklist" sections automatically
- Ensures minimum 1500+ characters of context
- Provides numbered checklists with evidence citations

**Specific Questions** (e.g., "What is the wifi password?")
- Uses focused retrieval (6 chunks)
- Leverages deterministic extractors for common queries
- Provides direct, evidence-based answers

**Deterministic Extractors** for common HR/IT questions:
- **Badge Pickup**: "Where do I pick up my badge?" → "BADGE_PICKUP: Available at reception/security desk"
- **Manager Name**: "Who is my manager?" → "MANAGER_NAME: [Extracted Name]"
- **Reception Location**: "Where is reception?" → "RECEPTION_LOCATION: Main lobby/front desk"

### Evidence & Grounding

- **Citation Tracking**: Every response includes `(CHUNK_ID=<id>)` references
- **Anchor Detection**: Identifies time slots, numeric values, and key terms
- **Grounding Validation**: Ensures answers are based only on indexed documents
- **Source Attribution**: Lists all contributing documents with snippets

### Use Cases

**HR Onboarding**: Comprehensive first-day checklists with time-specific instructions
**IT Support**: Wifi passwords, software installation guides, troubleshooting steps
**Policy Documentation**: Company policies with specific rule extraction
**Training Materials**: Step-by-step procedures with prerequisite identification
**FAQ Systems**: Natural language answers with source document references

## 📊 Database Schema

### Document Model
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL,
    status VARCHAR NOT NULL,  -- 'indexing', 'indexed', 'failed'
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tenant_id ON documents(tenant_id);
CREATE INDEX idx_status ON documents(status);
```

## 🔐 Security Considerations

### Production Deployment Checklist
- [ ] Change all default passwords in `app/auth.py`
- [ ] Set strong `JWT_SECRET_KEY` (min 32 characters)
- [ ] Use environment variables for all secrets
- [ ] Enable HTTPS/TLS for all connections
- [ ] Set up proper PostgreSQL user permissions
- [ ] Configure CORS for specific domains only
- [ ] Implement rate limiting on API endpoints
- [ ] Set up log monitoring and alerting
- [ ] Regular security audits and updates
- [ ] Backup database regularly

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Test database connection
psql -h localhost -U ragify -d ragify_db

# Check if database is running
docker ps | grep postgres
```

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
# (depends on your installation method)
```

### Authentication Issues
- Clear browser localStorage and try logging in again
- Check JWT token expiry (default 24 hours)
- Verify credentials in `app/auth.py`

## 📝 Development

### Running in Mock Mode (No External Dependencies)
```bash
export RAGIFY_MOCK=1
uvicorn main:app --reload
```

### Database Migrations
```python
# In Python shell
from app.database import init_db
init_db()  # Creates all tables
```

## 📚 Additional Documentation

The project keeps detailed historical notes in the archive, while the active docs remain in the top-level backend folder.

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing manual with 50+ test cases
- [SETUP_WITHOUT_DOCKER.md](SETUP_WITHOUT_DOCKER.md) - Alternative setup options
- [archive/legacy-notes/PHASE1_SUMMARY.md](archive/legacy-notes/PHASE1_SUMMARY.md) - Complete Phase 1 implementation details
- [archive/legacy-notes/CI_TESTING.md](archive/legacy-notes/CI_TESTING.md) - CI/CD testing guide with mock provider
- [archive/legacy-notes/GROUNDING_GATE_SUMMARY.md](archive/legacy-notes/GROUNDING_GATE_SUMMARY.md) - Evidence validation system
- [archive/legacy-notes/SSE_IMPLEMENTATION.md](archive/legacy-notes/SSE_IMPLEMENTATION.md) - Streaming response implementation
- [archive/legacy-notes/ASYNC_INGESTION.md](archive/legacy-notes/ASYNC_INGESTION.md) - Document processing pipeline
- [archive/legacy-notes/EMBEDDER_IMPLEMENTATION.md](archive/legacy-notes/EMBEDDER_IMPLEMENTATION.md) - Embedding and retrieval system
- [archive/legacy-notes/BRANCH_STATUS.md](archive/legacy-notes/BRANCH_STATUS.md) - Current development status
- [archive/legacy-notes/CONVERSATION_SUPPORT.md](archive/legacy-notes/CONVERSATION_SUPPORT.md) - Multi-turn conversation features
- [archive/legacy-notes/GUARDRAILS.md](archive/legacy-notes/GUARDRAILS.md) - Safety and quality controls

## 🤝 Contributing

1. Create a feature branch from `master`
2. Make your changes
3. Test thoroughly (see TESTING_GUIDE.md)
4. Submit a pull request

> The repository keeps historical project notes in the archive folder for reference, while the default branch is kept focused on the clean, maintainable product experience.

## 📄 License

This project is part of the RAGify MVP and is intended for demonstration and educational purposes.

## 🔗 Tech Stack

- **Backend**: FastAPI 0.100+ with async streaming support
- **Database**: PostgreSQL 15+ with SQLAlchemy ORM
- **Vector DB**: ChromaDB <0.4.0 with tenant-scoped collections
- **AI Models**: Ollama (nomic-embed-text, llama3) + OpenAI API support
- **Auth**: JWT with bcrypt password hashing
- **Frontend**: Vanilla JavaScript with streaming response handling
- **Document Processing**: PyPDF2, python-docx, character-based chunking
- **Testing**: pytest with 50+ comprehensive test cases
- **Deployment**: Docker support with multi-provider LLM flexibility

### RAG Pipeline Features

#### Retrieval
- **Hybrid Scoring**: Lexical overlap + semantic similarity ranking
- **MMR Selection**: Maximal Marginal Relevance for diverse context
- **Adjacent Expansion**: Automatic chunk inclusion for key sections
- **Fallback Rewriting**: Query enhancement for better document matching

#### Generation
- **Evidence-Based**: Strict citation requirements with chunk ID tracking
- **Context-Aware**: Minimum length thresholds (1500+ chars for broad questions)
- **Streaming Output**: Real-time response generation
- **Grounding Validation**: Evidence verification with anchor type detection

#### Intelligence
- **Question Classification**: Broad vs. specific question detection
- **Deterministic Extractors**: Specialized handling for common queries
- **Anchor Detection**: Time and numeric pattern recognition
- **Coverage Gates**: Quality assurance for response completeness
