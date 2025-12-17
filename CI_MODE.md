# CI Mode Configuration

## Overview
RAGify supports an explicit **CI mode** that automatically configures the application to run without external dependencies (Ollama, OpenAI, PostgreSQL). This enables fast, deterministic testing in continuous integration environments.

## Activation

Set either environment variable:

```bash
# Option 1: Standard CI flag
export CI=true

# Option 2: Explicit app mode
export APP_MODE=ci
```

Supported values for `CI`: `true`, `1`, `yes` (case-insensitive)

## Automatic Configuration

When CI mode is enabled, the application automatically configures:

| Component | Production Default | CI Mode Override |
|-----------|-------------------|------------------|
| **LLM Provider** | `ollama` or `openai` | `mock` |
| **Embedding Provider** | `ollama` or `openai` | `mock` |
| **Task Runner** | `background` (FastAPI BackgroundTasks) | `inline` (synchronous execution) |
| **HTTP Client** | Initialized for API calls | `None` (disabled) |
| **Database** | PostgreSQL (with fallback) | SQLite or no-op |

## Benefits

✅ **Zero External Dependencies**: No Ollama, OpenAI, or external services required  
✅ **Fast Execution**: Mock providers return instantly, inline task runner executes synchronously  
✅ **Deterministic**: Mock responses are predictable and reproducible  
✅ **Cost-Free**: No API calls to paid services  
✅ **CI/CD Ready**: Perfect for GitHub Actions, GitLab CI, Jenkins, etc.

## Startup Behavior

When the app boots in CI mode, you'll see a startup banner:

```
============================================================
CI MODE ENABLED
  - LLM Provider: mock (no Ollama/OpenAI required)
  - Embedding Provider: mock (deterministic vectors)
  - Task Runner: inline (synchronous execution)
  - HTTP Client: disabled
============================================================
```

## Health Check

The `/health` endpoint confirms CI mode:

```bash
curl http://localhost:8000/health
```

Response in CI mode:
```json
{
  "status": "ok",
  "mock_mode": true,
  "ragify_mode": "demo"
}
```

## Usage in CI/CD Pipelines

### GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      CI: true  # Automatically set by GitHub Actions
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest test_integration.py -v
      
      - name: Start app (smoke test)
        run: |
          uvicorn main:app --host 0.0.0.0 --port 8000 &
          sleep 5
          curl http://localhost:8000/health
```

### GitLab CI

```yaml
test:
  image: python:3.11
  variables:
    APP_MODE: ci
  script:
    - pip install -r requirements.txt
    - pytest test_integration.py -v
    - uvicorn main:app --host 0.0.0.0 --port 8000 &
    - sleep 5
    - curl http://localhost:8000/health
```

### Docker Compose (CI)

```yaml
services:
  ragify-ci:
    build: .
    environment:
      - CI=true
    ports:
      - "8000:8000"
    command: uvicorn main:app --host 0.0.0.0 --port 8000
```

## Testing

### Unit Tests

CI mode configuration is tested in [test_ci_mode.py](test_ci_mode.py):

```python
import os
os.environ["CI"] = "true"

from app.runtime import build_runtime_from_env

runtime = build_runtime_from_env()

assert runtime.http_client is None
assert type(runtime.llm_provider).__name__ == "MockLLMProvider"
assert type(runtime.embedding_provider).__name__ == "MockEmbedder"
```

### Integration Tests

App startup in CI mode is tested in [test_ci_mode_startup.py](test_ci_mode_startup.py):

```python
os.environ["CI"] = "true"

from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    response = client.get("/health")
    assert response.json()["mock_mode"] is True
```

## Manual Testing

### Start App in CI Mode

```bash
# Set CI mode
export CI=true

# Start the app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Configuration

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected output:
# {"status":"ok","mock_mode":true,"ragify_mode":"demo"}

# Upload a document (will use mock indexing)
curl -X POST http://localhost:8000/api/upload \
  -H "Authorization: Bearer <token>" \
  -F "files=@test.txt"

# Query (will use mock LLM)
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is in the document?"}'
```

## Production vs CI Mode

| Aspect | Production | CI Mode |
|--------|-----------|---------|
| **LLM Calls** | Real HTTP calls to Ollama/OpenAI | Mock responses (instant) |
| **Embeddings** | Vector generation via API | Deterministic SHA-256 vectors |
| **Indexing** | Background async task | Synchronous inline execution |
| **Database** | PostgreSQL with persistence | SQLite or test fixtures |
| **Response Time** | Depends on model/API | <100ms (deterministic) |
| **Cost** | API usage fees | $0 |

## Troubleshooting

### CI mode not detected

Check environment variables:
```bash
echo $CI
echo $APP_MODE
```

Verify startup logs show CI banner.

### Tests still trying to connect to Ollama

Ensure tests are not overriding `LLM_PROVIDER`:
```python
# Remove any explicit provider settings
if "LLM_PROVIDER" in os.environ:
    del os.environ["LLM_PROVIDER"]

os.environ["CI"] = "true"
```

### Database connection errors

CI mode gracefully handles database failures. Check logs for:
```
Database initialization failed: ... App will run without Postgres.
```

This is expected and non-fatal in CI mode.

## See Also

- [test_ci_mode.py](test_ci_mode.py) - CI mode configuration tests
- [test_ci_mode_startup.py](test_ci_mode_startup.py) - App startup tests
- [test_integration.py](test_integration.py) - Full workflow integration tests
- [app/runtime.py](app/runtime.py) - Runtime configuration logic
