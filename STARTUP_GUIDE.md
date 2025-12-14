# RAGify Startup & Shutdown Scripts

Quick scripts to manage all RAGify services with a single command.

## Files

- **`startup.ps1`** - Starts all required services
- **`shutdown.ps1`** - Stops all running services
- **`README.md`** - This file

## Prerequisites

Before running the startup script, ensure you have:

1. **Python 3.9+** - With FastAPI, uvicorn, and other dependencies installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama** - Download from https://ollama.ai
   - After installation, pull required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ```

3. **Docker** (Optional) - For PostgreSQL database
   - Download from https://www.docker.com/products/docker-desktop

## Usage

### Start All Services (Demo Mode)

```powershell
.\startup.ps1
```

This will start:
- ✅ **Ollama** - Embeddings and LLM inference
- ✅ **FastAPI** - Web server on `http://localhost:8000`
- ⚠️ **PostgreSQL** - Optional (requires `-WithPostgres` flag)

### Start with PostgreSQL

```powershell
.\startup.ps1 -WithPostgres
```

This additionally starts:
- 🐘 **PostgreSQL** - On `localhost:5432` (user: `ragify`, password: `ragify123`)

### Start with Different Mode

```powershell
.\startup.ps1 -Mode production
# or
.\startup.ps1 -Mode dev
# or
.\startup.ps1 -Mode demo  # default
```

Available modes:
- **demo** (default) - Fast responses, safe defaults (100 tokens, 4 chunks)
- **dev** - Full features, verbose logs, no limits
- **prod** - Balanced quality/speed (500 tokens, 20 chunks)

### Shutdown All Services

```powershell
.\shutdown.ps1
```

This will stop:
- ✅ FastAPI server
- ✅ Ollama
- ✅ PostgreSQL (if running)

## Service Details

### Ollama (Port 11434)

- **Purpose**: Embeddings and LLM inference
- **Models**: 
  - `nomic-embed-text` (embedding model)
  - `llama3.2:1b` (LLM)
- **Status**: Check with `ollama list`

### FastAPI (Port 8000)

- **Purpose**: Web API and UI
- **Access**: http://localhost:8000
- **Mode**: Controlled by `RAGIFY_MODE` environment variable

### PostgreSQL (Port 5432)

- **Purpose**: Document metadata storage (optional)
- **User**: `ragify`
- **Password**: `ragify123`
- **Database**: `ragify_db`
- **Container**: `ragify-postgres`

## Troubleshooting

### Ollama Won't Start
```powershell
# Check if already running
Get-Process -Name ollama

# Or start manually
ollama serve
```

### FastAPI Port 8000 Already in Use
```powershell
# Kill process using port 8000
Stop-Process -Name "python" -Force
```

### PostgreSQL Container Error
```powershell
# Check existing containers
docker ps -a

# Remove old container if needed
docker rm ragify-postgres

# Restart Docker Desktop if issues persist
```

### Models Not Available in Ollama
```bash
# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:1b

# List available models
ollama list
```

## Log Files

- **FastAPI logs**: Printed to console
- **Ollama logs**: Printed to console
- **PostgreSQL logs**: `docker logs ragify-postgres`

## Advanced

### Manual Service Start

**Ollama:**
```powershell
ollama serve
```

**FastAPI (demo mode):**
```powershell
$env:RAGIFY_MODE="demo"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**PostgreSQL:**
```powershell
docker run -d `
  --name ragify-postgres `
  -e POSTGRES_USER=ragify `
  -e POSTGRES_PASSWORD=ragify123 `
  -e POSTGRES_DB=ragify_db `
  -p 5432:5432 `
  postgres:15
```

### Check Service Status

```powershell
# Check running processes
Get-Process -Name ollama
Get-Process -Name python | Where-Object { $_.CommandLine -like "*uvicorn*" }

# Check Docker containers
docker ps

# Check Ollama models
ollama list
```

## Performance Notes

- **Demo mode**: Fast (~2-3 seconds), safe defaults
- **Dev mode**: Full features, slower, verbose logging
- **Prod mode**: Balanced, optimized for production

For best performance on limited VRAM:
- Use demo or prod mode
- Use smaller LLM (`llama2`, `neural-chat`)
- Reduce `top_k` in config.py

## Support

For issues, check:
1. `http://localhost:8000/api/config` - Current configuration
2. Console output - Service logs
3. `app/config.py` - Configuration settings

---

**RAGify MVP** - Local RAG with Ollama + ChromaDB + Free Hybrid Reranking
