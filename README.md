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

### Document Management
- **Upload Tracking**: PostgreSQL records for all uploaded documents
- **Status Monitoring**: Track indexing progress (indexing, indexed, failed)
- **Error Handling**: Detailed error messages for failed operations

### RAG Capabilities
- **Semantic Search**: Vector-based document retrieval using Ollama embeddings
- **Streaming Responses**: Real-time answer generation with source attribution
- **Context-Aware**: Answers strictly based on indexed documents

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
ollama pull llama3

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

### Protected Endpoints (Require JWT)
- `GET /api/config` - Get tenant-specific configuration
- `POST /api/upload` - Upload and index documents
- `POST /api/query` - Query documents with streaming response
- `GET /api/documents` - List all documents for current tenant
- `POST /api/reset` - Reset tenant's vector store

### Other
- `GET /health` - Health check endpoint
- `GET /` - Redirects to login page

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
│       ├── ingestion.py       # Document processing
│       └── rag_service.py     # Multi-tenant RAG engine
├── static/
│   ├── login.html             # Authentication page
│   └── index.html             # Main application UI
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── TESTING_GUIDE.md           # Comprehensive testing instructions
├── SETUP_WITHOUT_DOCKER.md    # Alternative setup options
└── PHASE1_SUMMARY.md          # Implementation documentation
```

## 🧪 Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions, including:
- Quick start guide
- Step-by-step testing procedures
- Tenant isolation verification
- API testing examples
- Troubleshooting tips

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://ragify:ragify@localhost:5432/ragify_db` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | `your-secret-key-change-in-production` | Secret key for JWT signing |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRY_HOURS` | `24` | Token expiry time |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `RAGIFY_OLLAMA_TIMEOUT` | `300` | Timeout for Ollama requests |
| `RAGIFY_MOCK` | `0` | Enable mock mode (no Ollama/Chroma) |

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
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing manual
- [SETUP_WITHOUT_DOCKER.md](SETUP_WITHOUT_DOCKER.md) - Alternative setup options
- [BRANCH_STATUS.md](BRANCH_STATUS.md) - Current development status

## 🤝 Contributing

1. Create a feature branch from `phase1-mvp`
2. Make your changes
3. Test thoroughly (see TESTING_GUIDE.md)
4. Submit a pull request

## 📄 License

This project is part of the RAGify MVP and is intended for demonstration and educational purposes.

## 🔗 Tech Stack

- **Backend**: FastAPI 0.100+
- **Database**: PostgreSQL 15+ with SQLAlchemy
- **Vector DB**: ChromaDB <0.4.0
- **AI Models**: Ollama (nomic-embed-text, llama3)
- **Auth**: JWT with bcrypt
- **Frontend**: Vanilla JavaScript (no framework)
- **Document Processing**: PyPDF, python-docx

---

**Built with ❤️ for intelligent document processing**
