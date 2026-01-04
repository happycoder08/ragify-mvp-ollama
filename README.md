# RAGify AI – Multi-Tenant Document Intelligence Platform

A production-ready Retrieval-Augmented Generation (RAG) system with **multi-tenant support**, **JWT authentication**, and **document tracking**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Static HTML)                  │
│  • Login page with authentication                            │
│  • Tenant-specific branding and configuration                │
│  • Document upload and management                            │
│  • Query interface with streaming responses                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (main.py)                  │
│  • JWT authentication middleware                             │
│  • Protected API endpoints                                   │
│  • Tenant access verification                                │
│  • Streaming response handling                               │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐
│   PostgreSQL    │  │  ChromaDB       │  │   Ollama     │
│   (Metadata)    │  │  (Embeddings)   │  │  (AI Models) │
│                 │  │                 │  │              │
│ • Documents     │  │ • Tenant-scoped │  │ • Embeddings │
│ • Users         │  │   collections   │  │ • Chat (LLM) │
│ • Tenants       │  │ • Vector search │  │              │
└─────────────────┘  └─────────────────┘  └──────────────┘
```

## ✨ Features

### Multi-Tenancy
- **Tenant Isolation**: Each tenant has separate document collections in ChromaDB
- **Custom Branding**: Configurable colors, logos, and titles per tenant
- **Access Control**: Users can only access their tenant's documents

### Authentication & Security
- **JWT Tokens**: Secure authentication with 24-hour token expiry
- **Password Hashing**: bcrypt with 12 rounds for secure password storage
- **Protected Endpoints**: All document operations require authentication

### Advanced RAG Capabilities
- **Intelligent Question Classification**: Automatic detection of broad vs. specific questions
- **Context Expansion**: Adjacent chunk inclusion for comprehensive answers (e.g., "First Day" checklists)
- **Diverse Retrieval**: MMR (Maximal Marginal Relevance) selection for balanced context
- **Evidence-Based Responses**: Strict citation requirements with chunk ID references
- **Deterministic Extractors**: Specialized handling for common questions:
  - Badge pickup locations
  - Manager/supervisor names
  - Reception/front desk locations
- **Hybrid Scoring**: Combines lexical overlap and semantic similarity for optimal retrieval
- **Coverage Assurance**: Minimum context thresholds (1500+ chars for broad questions)
- **Fallback Query Rewriting**: Automatic query enhancement for better document matching

### Document Processing
- **Multi-Format Support**: PDF, DOCX, and TXT file processing
- **Smart Chunking**: Character-based segmentation with configurable overlap (800 chars + 200 overlap)
- **Metadata Preservation**: Header extraction, source tracking, and chunk indexing
- **Upload Tracking**: PostgreSQL records with indexing status and error handling

### AI & Response Features
- **Streaming Responses**: Real-time answer generation with source attribution
- **Multiple LLM Providers**: Ollama (local), OpenAI (cloud), Mock (testing)
- **Grounding Validation**: Evidence verification with time/numeric anchor detection
- **Context-Aware Generation**: Answers strictly based on indexed documents only
- **Debug Capabilities**: Detailed retrieval and processing information for troubleshooting

### Conversations & History
- **Conversation Threads**: Conversations persisted per tenant with titles and timestamps
- **Message History**: Both user and assistant messages stored with optional source metadata
- **Context Reuse**: `/api/query` can take a `conversation_id` and reuse the last N turns
- **Cleanup APIs**: Endpoints to list, delete, and inspect conversation history per tenant

### Guardrails & Rate Limiting
- **Upload Guardrails**: Limits on file size, count, and allowed extensions per tenant
- **Rate Limiting**: Per-tenant request and upload-size quotas with `/api/rate-limit-status`
- **Grounding Gate**: Configurable thresholds to require sufficient evidence before answering
- **Tenant-Specific Policies**: Guardrail config retrieved via `/api/guardrails`

### Quality Assurance
- **Comprehensive Testing**: 50+ test cases covering all major functionality
- **Mock Provider**: Deterministic responses for CI/CD pipelines
- **Integration Testing**: End-to-end validation of upload → query → response pipeline
- **Performance Validation**: Context length and retrieval quality assertions

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (or use without database - see [SETUP_WITHOUT_DOCKER.md](SETUP_WITHOUT_DOCKER.md))
- Ollama with `nomic-embed-text` and `llama3` models

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL (Optional but Recommended)

**Option A: Using Docker**
```bash
docker run --name ragify-postgres -e POSTGRES_USER=ragify -e POSTGRES_PASSWORD=ragify -e POSTGRES_DB=ragify_db -p 5432:5432 -d postgres:15
```

**Option B: Native Installation**
See [SETUP_WITHOUT_DOCKER.md](SETUP_WITHOUT_DOCKER.md) for detailed instructions.

### 3. Set Up Ollama

```bash
# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:1b  # or `llama3`

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### 4. Configure Environment (Optional)

Create `.env` file:
```bash
DATABASE_URL=postgresql://ragify:ragify@localhost:5432/ragify_db
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
```

### 5. Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access the Application

Open http://localhost:8000 in your browser.

## 👥 Default Login Credentials

| Tenant | Username | Password | Brand Color | Description |
|--------|----------|----------|-------------|-------------|
| Default | `demo` | `demo123` | Blue (#3b82f6) | Default tenant for general use |
| ACME Corp | `acme_admin` | `acme123` | Red (#ef4444) | ACME Corporation tenant |
| Finance Co | `finance_user` | `finance123` | Green (#10b981) | Finance Company tenant |

**⚠️ Security Warning**: Change these credentials in production! Update `app/auth.py` with secure passwords.

## 📡 API Endpoints

### Public Endpoints
- `POST /api/login` - Authenticate user and get JWT token
- `GET /health` - Basic health check (status, mock mode, active RAGIFY mode)
- `GET /api/system/config` - Active RAG configuration (mode, tokens, top_k, provider)
- `GET /` - Serves main SPA (static frontend)

### Protected Endpoints (Require JWT)
- `GET /api/config` - Get tenant-specific branding/config
- `GET /api/guardrails` - Get tenant-specific guardrail limits (upload and query constraints)
- `GET /api/rate-limit-status` - Current rate limit usage for the tenant
- `GET /api/debug` - High-level runtime debug info (providers, collection stats, recent docs)
- `GET /api/health/deps` - Dependency health (Ollama + Chroma) for the current tenant
- `POST /api/upload` - Upload and index documents (PDF, DOCX, TXT) with background processing
- `POST /api/query` - Advanced document querying with intelligent retrieval
  - **Broad Question Handling**: Automatic context expansion for comprehensive questions
  - **Evidence Citations**: Responses include chunk IDs and evidence snippets
  - **Streaming Response**: Server-sent events (`event: token` / `event: final`)
  - **Debug Mode**: Optional structured `debug_info` with retrieval and grounding details
- `GET /api/documents` - List all documents for current tenant with status tracking
- `GET /api/documents/{doc_id}/status` - Status of a single document (pending/indexed/failed)
- `POST /api/documents/{doc_id}/reindex` - Re-run indexing pipeline for one document
- `POST /api/documents/purge` - Delete all tenant documents (metadata + files) and reset vectors
- `POST /api/reset` - Reset the current tenant's vector store (destructive)
- **Conversations API**:
  - `POST /api/conversations` - Create a conversation
  - `GET /api/conversations` - List recent conversations for the tenant
  - `GET /api/conversations/{conversation_id}` - Get conversation with messages
  - `POST /api/conversations/{conversation_id}/messages` - Append a message
  - `DELETE /api/conversations/{conversation_id}` - Delete a conversation
- **Debug Utilities**:
  - `GET /api/debug/find_chunks` - Search indexed chunks for a substring (per-tenant)
- **Demo Utilities (demo mode only)**:
  - `POST /api/demo-verify` - Run a small suite of demo queries to verify evidence coverage

### Response Formats

#### Query Response
```json
{
  "answer": "Streaming text response with evidence-based information...",
  "refused": false,
  "refusal_reason": null,
  "sources": [
    {
      "doc_id": 1,
      "filename": "document1.pdf",
      "chunk_id": "doc1_chunk_5"
    }
  ],
  "evidence": [
    {
      "snippet": "Relevant text excerpt...",
      "chunk_id": "doc1_chunk_5",
      "heading": "First Day Checklist",
      "doc_id": 1,
      "anchor_type": "TIME"
    }
  ],
  "debug_info": {
    "retrieved_count": 25,
    "selected_count": 8,
    "context_length": 2100,
    "collection_name": "documents_default__MockEmbedder__384"
  }
}
```
 
### Other
- `GET /health` - Basic health check
- `GET /` - Main application UI

## 🗂️ Project Structure

```
ragify-mvp-ollama/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration and paths
│   ├── auth.py                # JWT authentication system
│   ├── database.py            # PostgreSQL connection
│   ├── models.py              # SQLAlchemy models
│   ├── tenant_config.py       # Tenant configurations
│   └── services/
│       ├── ingestion.py       # Document processing & chunking
│       └── rag_service.py     # Advanced multi-tenant RAG engine
│           ├── ChunkHit dataclass    # Document chunk representation
│           ├── MMR selection         # Diverse retrieval algorithm
│           ├── Broad question detection
│           ├── Evidence extraction   # Citation and grounding
│           ├── Deterministic extractors
│           └── Hybrid scoring        # Lexical + semantic ranking
├── static/
│   ├── login.html             # Authentication page
│   └── index.html             # Main application UI
├── main.py                    # FastAPI application with streaming
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Test configuration
├── test_*.py                  # 50+ comprehensive test files
├── *.md                       # Documentation files
└── CI_TESTING.md              # CI/CD testing guide
```

### Key Components

#### RAG Service (`rag_service.py`)
- **Question Classification**: Broad vs. specific question detection
- **Retrieval Pipeline**: Hybrid scoring with lexical overlap + embeddings
- **Selection Algorithms**: MMR for diversity, adjacent chunk expansion
- **Evidence Processing**: Citation tracking, anchor type detection
- **Response Generation**: Context-aware LLM prompting with strict evidence requirements

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

- [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) - Complete Phase 1 implementation details
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing manual with 50+ test cases
- [SETUP_WITHOUT_DOCKER.md](SETUP_WITHOUT_DOCKER.md) - Alternative setup options
- [CI_TESTING.md](CI_TESTING.md) - CI/CD testing guide with mock provider
- [GROUNDING_GATE_SUMMARY.md](GROUNDING_GATE_SUMMARY.md) - Evidence validation system
- [SSE_IMPLEMENTATION.md](SSE_IMPLEMENTATION.md) - Streaming response implementation
- [ASYNC_INGESTION.md](ASYNC_INGESTION.md) - Document processing pipeline
- [EMBEDDER_IMPLEMENTATION.md](EMBEDDER_IMPLEMENTATION.md) - Embedding and retrieval system
- [BRANCH_STATUS.md](BRANCH_STATUS.md) - Current development status
- [CONVERSATION_SUPPORT.md](CONVERSATION_SUPPORT.md) - Multi-turn conversation features
- [GUARDRAILS.md](GUARDRAILS.md) - Safety and quality controls

## 🤝 Contributing

1. Create a feature branch from `phase1-mvp`
2. Make your changes
3. Test thoroughly (see TESTING_GUIDE.md)
4. Submit a pull request

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
