# RAGify Configuration Guide

## Overview

RAGify uses `RAGIFY_MODE` to switch between tuned presets defined in `app/config.py`.
Modes change token limits, retrieval depth, chunking, reranking, timeouts, and logging.

Set via environment variable:
```
RAGIFY_MODE=dev|demo|prod
```

## Quick Summary

| Setting                 | DEV                  | DEMO (default)         | PROD                 |
|-------------------------|----------------------|-------------------------|----------------------|
| Default query mode      | full                 | fast                    | full                 |
| Fast: max_tokens        | None (unlimited)     | 100                     | 100                  |
| Full: max_tokens        | None (unlimited)     | 150                     | 500                  |
| Fast: top_k             | 3                    | 20                      | 3                    |
| Full: top_k             | 6                    | 15                      | 5                    |
| Fast: top_n (post-rank) | -                    | 5                       | -                    |
| Full: top_n (post-rank) | -                    | 8                       | -                    |
| Similarity threshold    | 400                  | 400                     | 350                  |
| Chunk size / overlap    | 800 / 200            | 300 / 50                | 800 / 200            |
| Reranking enabled       | False                | False                   | True                 |
| Reranker top_n          | -                    | 8                       | 3                    |
| Context budget (chars)  | -                    | 12000                   | -                    |
| LLM model               | llama3.1:8b          | llama3.2:1b             | llama3.2:1b (env)    |
| Log level               | DEBUG                | INFO                    | INFO                 |

## Mode Details

### DEV (development/debugging)
- Full visibility with relaxed limits.
- Best for debugging retrieval or validation logic.
- Slowest mode, but highest observability.

Example:
```
RAGIFY_MODE=dev uvicorn main:app --reload
```

### DEMO (default, fast)
- Tuned for speed and reasonable recall.
- Smaller chunks, larger retrieval, context budget cap.
- Best for end-to-end testing and demos.

Example:
```
# Demo is the default
uvicorn main:app --host 0.0.0.0 --port 8000

# Or explicit:
RAGIFY_MODE=demo uvicorn main:app
```

### PROD (balanced + reranking)
- Balanced performance and quality.
- Reranking on by default.
- Best for production deployment.

Example:
```
RAGIFY_MODE=prod LLM_PROVIDER=openai LLM_MODEL=gpt-4 uvicorn main:app
```

## How to Enable Modes

Windows PowerShell:
```
setx RAGIFY_MODE demo
```

Temporary session:
```
$env:RAGIFY_MODE="demo"; uvicorn main:app
```

## Best Practices

- Use `demo` for end-to-end testing; it matches real usage.
- Use `dev` only when debugging or inspecting retrieval details.
- Use `prod` for staging/production to reflect final behavior.
- If you see refusals from low context, increase `top_k_*` or `context_budget_chars` in `demo`.
- Keep `chunk_size` and `top_k` aligned: smaller chunks usually need higher `top_k`.

## Overrides

You can override any preset with env vars:
```
RAGIFY_MODE=demo LLM_MODEL=llama3.2:1b RAGIFY_OLLAMA_TIMEOUT=600 uvicorn main:app
```

Common overrides:
- `LLM_PROVIDER`, `LLM_MODEL`
- `RERANKER_PROVIDER`
- `OLLAMA_BASE_URL`, `OPENAI_API_KEY`

## Inspect Active Config

```
GET /api/system/config
```

Returns the active values for the running mode.
