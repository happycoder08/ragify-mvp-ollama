# RAGify AI

RAGify is a full-stack document question-answering application. The repository is organized as a small monorepo with a FastAPI backend and a React/Vite frontend.

## Repository Structure

```text
apps/
  backend/       FastAPI API, RAG pipeline, tests, and legacy static files
  frontend/      React + TypeScript + Vite application
package.json     Root development orchestration scripts
```

The frontend development server proxies `/api` and `/health` to the backend at `http://localhost:8000`.

## Prerequisites

- Python 3.11+
- Node.js and npm
- PostgreSQL on `localhost:5432` (required for document uploads and document lists)
- Ollama with `nomic-embed-text` and `llama3.2:1b` available

To start PostgreSQL with Docker, start Docker Desktop and run:

```powershell
docker run --name ragify-postgres -e POSTGRES_USER=ragify -e POSTGRES_PASSWORD=ragify -e POSTGRES_DB=ragify_db -p 5432:5432 -d postgres:15
```

For native PostgreSQL setup, see [apps/backend/SETUP_WITHOUT_DOCKER.md](apps/backend/SETUP_WITHOUT_DOCKER.md).

## Install

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r apps/backend/requirements.txt
npm install
Push-Location apps/frontend
npm install
Pop-Location
```

Create `apps/backend/.env` when custom configuration is needed. Start with [apps/backend/.env.example](apps/backend/.env.example).

## Run Both Apps

From the repository root:

```powershell
npm run dev
```

This starts:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

The root scripts are defined in [package.json](package.json):

- `npm run dev:backend` starts Uvicorn from `apps/backend`, preserving relative storage paths.
- `npm run dev:frontend` starts Vite from `apps/frontend`.
- `npm run dev` runs both services with `concurrently`.

Open `http://localhost:5173` to use the React application. The backend also exposes its legacy static pages under `http://localhost:8000/static/`.

## Backend

The backend provides authentication, document ingestion, tenant isolation, ChromaDB retrieval, grounded answers, and streaming responses. See [apps/backend/README.md](apps/backend/README.md) for API capabilities and backend-specific details.

Run backend tests from the root with:

```powershell
pytest apps/backend
```

## Frontend

The frontend is a React, TypeScript, and Vite app. See [apps/frontend/README.md](apps/frontend/README.md) for UI behavior and testing details.

```powershell
npm --prefix apps/frontend run build
npm --prefix apps/frontend run test -- --run
```

## External Services

- PostgreSQL stores application metadata, authentication, and document records.
- ChromaDB persists embeddings under `apps/backend/vectorstore/`.
- Ollama provides embeddings and chat responses at `http://localhost:11434`.

## Reference Documentation
#### Document Ingestion (`ingestion.py`)
- **Format Support**: PDF, DOCX, TXT processing
- **Chunking Strategy**: Character-based with configurable overlap
- **Metadata Extraction**: Headers, source files, chunk indices
- **Error Handling**: Robust processing with detailed error reporting

#### Testing Suite
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline testing
- **Mock Provider**: Deterministic testing without external dependencies
- **CI/CD Support**: Automated testing for deployment pipelines

## 🧪 Testing

RAGify includes a comprehensive test suite with 50+ test cases covering all major functionality.

### Test Categories
- **Unit Tests**: Individual component validation (chunking, scoring, evidence extraction)
- **Integration Tests**: End-to-end pipeline testing (upload → index → query → response)
- **RAG Pipeline Tests**: Retrieval quality, context expansion, evidence validation
- **Grounding Tests**: Evidence verification and citation accuracy
- **Multi-tenant Tests**: Tenant isolation and access control
- **API Tests**: Endpoint validation and error handling

### Quick Test Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest test_rag_pipeline.py -v          # RAG pipeline validation
pytest test_grounding_gate.py -v        # Evidence verification
pytest test_integration.py -v           # End-to-end testing

# Run with coverage
pytest --cov=app --cov-report=html

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
When PostgreSQL is unavailable, the backend starts in degraded mode. Health checks and some existing Chroma queries may work, but document uploads and document listing will not.
