# RAGify Configuration Guide

## Overview

RAGify now supports centralized configuration via `RAGIFY_MODE` environment variable. All settings (timeouts, token limits, chunk sizes, similarity thresholds) are managed in `app/config.py` with presets optimized for different use cases.

## RAGIFY_MODE Options

Set via environment variable: `RAGIFY_MODE=dev|demo|prod`

### 🧪 **DEV Mode** (Development & Testing)
**Purpose:** Full features, verbose logging, no limits

**Settings:**
- Default query mode: `full` (detailed answers)
- Fast mode: 3 chunks, unlimited tokens
- Full mode: 6 chunks, unlimited tokens
- Similarity threshold: 400 (lenient for testing)
- Request timeout: 600s (10 min for slow models)
- Chunk size: 800 chars, 200 overlap
- Log level: DEBUG
- Timing logs: Enabled

**Use when:**
- Developing new features
- Testing with experimental models
- Debugging RAG pipeline
- Need verbose logs and metrics

**Example:**
```bash
RAGIFY_MODE=dev uvicorn main:app --reload
```

---

### 🚀 **DEMO Mode** (Default - Fast & Safe)
**Purpose:** Fast responses, safe defaults, limited context

**Settings:**
- Default query mode: `fast` (brief answers)
- Fast mode: 2 chunks, 50 tokens max
- Full mode: 4 chunks, 150 tokens max
- Similarity threshold: 350 (strict relevance)
- Request timeout: 300s (5 min)
- Chunk size: 800 chars, 200 overlap
- Log level: INFO
- Timing logs: Enabled

**Use when:**
- Live demos to customers/stakeholders
- Need 2-3 second response times
- Showcasing RAG capabilities
- Limited compute resources

**Example:**
```bash
# Demo mode is default, no env var needed
uvicorn main:app --host 0.0.0.0 --port 8000

# Or explicitly:
RAGIFY_MODE=demo uvicorn main:app
```

---

### 🏭 **PROD Mode** (Production Ready)
**Purpose:** Balanced quality/speed, security, monitoring

**Settings:**
- Default query mode: `full` (detailed answers)
- Fast mode: 3 chunks, 100 tokens max
- Full mode: 5 chunks, 500 tokens max
- Similarity threshold: 350 (balanced)
- Request timeout: 300s (5 min)
- Chunk size: 800 chars, 200 overlap
- Log level: INFO
- Timing logs: Enabled
- Provider: Configurable via LLM_PROVIDER env var

**Use when:**
- Running in production environment
- Need balance of speed and quality
- Monitoring and observability required
- Multiple tenants

**Example:**
```bash
RAGIFY_MODE=prod LLM_PROVIDER=openai LLM_MODEL=gpt-3.5-turbo uvicorn main:app
```

---

## Configuration Details

### Mode Comparison Table

| Setting                  | DEV          | DEMO (Default) | PROD         |
|-------------------------|--------------|----------------|--------------|
| **Default query mode**  | full         | fast           | full         |
| **Fast: top_k**         | 3            | 2              | 3            |
| **Fast: max_tokens**    | unlimited    | 50             | 100          |
| **Full: top_k**         | 6            | 4              | 5            |
| **Full: max_tokens**    | unlimited    | 150            | 500          |
| **Similarity threshold**| 400          | 350            | 350          |
| **Request timeout**     | 600s         | 300s           | 300s         |
| **Log level**           | DEBUG        | INFO           | INFO         |

### Response Time Estimates

| Mode          | Fast Query | Full Query |
|---------------|------------|------------|
| **DEV**       | 3-5s       | 10-20s     |
| **DEMO**      | 2-3s       | 5-8s       |
| **PROD**      | 3-4s       | 8-12s      |

*Estimates based on llama3.2:1b on moderate hardware*

---

## Environment Variable Overrides

You can override specific settings without changing RAGIFY_MODE:

```bash
# Use demo mode but increase timeout for slow models
RAGIFY_MODE=demo RAGIFY_OLLAMA_TIMEOUT=600 uvicorn main:app

# Use prod mode but switch to OpenAI
RAGIFY_MODE=prod LLM_PROVIDER=openai LLM_MODEL=gpt-4 OPENAI_API_KEY=sk-... uvicorn main:app

# Use dev mode with custom Ollama instance
RAGIFY_MODE=dev OLLAMA_BASE_URL=http://remote-server:11434 uvicorn main:app
```

### Available Override Variables

- `LLM_PROVIDER`: "ollama" or "openai" (overrides mode default)
- `LLM_MODEL`: Model name (overrides mode default)
- `OLLAMA_BASE_URL`: Ollama server URL
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_BASE_URL`: OpenAI base URL

---

## API Changes

### New Endpoint: System Config

**GET /api/system/config** (public)

Returns active RAGify configuration:

```json
{
  "ragify_mode": "demo",
  "default_mode": "fast",
  "max_tokens_fast": 50,
  "max_tokens_full": 150,
  "top_k_fast": 2,
  "top_k_full": 4,
  "similarity_threshold": 350,
  "request_timeout": 300,
  "llm_provider": "ollama",
  "llm_model": "llama3.2:1b",
  "log_level": "INFO"
}
```

### Query Endpoint Updates

**POST /api/query**

The `mode` parameter now defaults to the value from RAGIFY_MODE:
- DEV mode: defaults to `mode=full`
- DEMO mode: defaults to `mode=fast`
- PROD mode: defaults to `mode=full`

You can still override per-request:
```json
{
  "question": "What is the policy?",
  "mode": "full"  // Override to full mode even in demo
}
```

---

## Testing Configuration

Run the test script to see all mode configurations:

```bash
python test_config.py
```

Output shows configuration for each mode:
```
============================================================
RAGIFY_MODE = DEV
============================================================
{
  "ragify_mode": "dev",
  "default_mode": "full",
  "max_tokens_fast": null,
  "max_tokens_full": null,
  ...
}

Key differences:
  - Default query mode: full
  - Fast mode: 3 chunks, None tokens
  - Full mode: 6 chunks, None tokens
  ...
```

---

## Migration Guide

### Before (Hardcoded Settings)
```python
# rag_service.py
SIMILARITY_THRESHOLD = 350
max_tokens = 50 if mode == "fast" else None

# main.py
top_k = 2 if mode == "fast" else payload.top_k
```

### After (Centralized Config)
```python
# All settings from config.py
from app.config import (
    SIMILARITY_THRESHOLD,
    MAX_TOKENS_FAST,
    MAX_TOKENS_FULL,
    TOP_K_FAST,
    TOP_K_FULL,
)
```

**Benefits:**
1. Single source of truth for all settings
2. Mode-specific presets (dev/demo/prod)
3. Easy environment variable overrides
4. Consistent behavior across services

---

## Best Practices

### 1. Development Workflow
```bash
# Start with dev mode for full features
RAGIFY_MODE=dev uvicorn main:app --reload

# Test with demo mode before deploying
RAGIFY_MODE=demo uvicorn main:app

# Deploy with prod mode
RAGIFY_MODE=prod uvicorn main:app
```

### 2. Demo Presentations
```bash
# Demo mode gives fast 2-3s responses
RAGIFY_MODE=demo uvicorn main:app --host 0.0.0.0

# Visit http://localhost:8000 - all queries default to fast mode
```

### 3. Production Deployment
```bash
# Prod mode with OpenAI for quality
RAGIFY_MODE=prod \
  LLM_PROVIDER=openai \
  LLM_MODEL=gpt-3.5-turbo \
  OPENAI_API_KEY=sk-... \
  uvicorn main:app --workers 4
```

### 4. Local Testing with Different Models
```bash
# Dev mode with larger model
RAGIFY_MODE=dev LLM_MODEL=llama3:8b uvicorn main:app
```

---

## Troubleshooting

### Slow Responses in Demo Mode
**Symptom:** Queries taking >5 seconds in demo mode

**Solutions:**
1. Check if model is loaded: `ollama list`
2. Preload model: `ollama pull llama3.2:1b`
3. Switch to smaller model: `LLM_MODEL=llama3.2:1b`
4. Increase timeout if needed: `RAGIFY_OLLAMA_TIMEOUT=600`

### Too Many Irrelevant Sources
**Symptom:** Wrong documents appearing in sources

**Solutions:**
1. Check threshold: Lower = stricter (demo/prod use 350)
2. Switch to dev mode to see all distances in logs
3. Adjust threshold in config.py for your use case
4. Clear vectorstore: `rm -rf vectorstore/`

### Want Longer Answers in Demo Mode
**Symptom:** Answers too brief (50 tokens max in demo)

**Solutions:**
1. Switch to prod mode: `RAGIFY_MODE=prod`
2. Override per-request: `{"mode": "full"}` in query payload
3. Edit config.py: Increase `MAX_TOKENS_FAST` in DEMO preset

---

## Summary

**RAGIFY_MODE** provides three battle-tested configurations:

- 🧪 **DEV**: Full features, verbose logs, no limits → for development
- 🚀 **DEMO**: Fast responses, safe defaults → for demos (default)
- 🏭 **PROD**: Balanced quality/speed → for production

Simply set `RAGIFY_MODE=<mode>` and all settings (tokens, chunks, thresholds, timeouts) are configured optimally for that use case.

Override specific settings via environment variables when needed. Check active config at `/api/system/config`.
