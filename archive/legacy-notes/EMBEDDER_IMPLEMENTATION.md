# Embedder Interface Implementation

## Overview
Created a clean abstraction layer for embeddings with support for both production (HTTP-based) and testing (deterministic mock) providers.

## Files Created/Modified

### New Files
- **app/services/embeddings.py** (380 lines)
  - `Embedder` protocol interface
  - `MockEmbedder` class (deterministic, no network)
  - `RealEmbedder` class (Ollama/OpenAI support)
  - `create_embedder()` factory function

- **test_embeddings.py** (241 lines)
  - 9 comprehensive unit tests
  - Tests: deterministic, discriminative, normalized, dimension stable, custom dimension, batch processing, empty input, similarity behavior, no HTTP client required
  - All tests passing ✓

### Modified Files
- **app/runtime.py**
  - Updated `build_runtime_from_env()` to use `create_embedder(provider_type, http_client)`
  - Updated `build_test_runtime()` to use `create_embedder("mock")`
  - Removed dependency on `create_embedding_provider()`

## Architecture

### Embedder Protocol
```python
class Embedder(Protocol):
    @abstractmethod
    async def embed_texts(self, texts: List[str], tenant_id: str) -> List[List[float]]:
        """Embed a list of texts into vector representations."""
        ...
```

### MockEmbedder
**Purpose:** Deterministic embeddings for testing without network dependencies

**Algorithm:**
1. SHA-256 hash of input text (32 bytes)
2. Extend by re-hashing to get `dimension * 4` bytes
3. Convert each 4-byte chunk to unsigned int (0 to 2^32-1)
4. Map to [-1, 1]: `(int_val / (2^32 - 1)) * 2 - 1`
5. Normalize to unit length: `vector / ||vector||`

**Properties:**
- ✓ Deterministic: same text → same vector
- ✓ Discriminative: different text → different vector
- ✓ Normalized: ||vector|| = 1.0 (realistic for similarity)
- ✓ No HTTP client required
- ✓ Fixed dimension (384 by default, configurable)

**Constructor:**
```python
MockEmbedder(dimension: int = 384)
```

### RealEmbedder
**Purpose:** Production embeddings via HTTP APIs (Ollama/OpenAI)

**Features:**
- Supports Ollama (nomic-embed-text, 768 dim)
- Supports OpenAI (text-embedding-3-small, 1536 dim)
- Parallel requests for Ollama (efficiency)
- Batch API for OpenAI
- Error handling and logging

**Constructor:**
```python
RealEmbedder(
    http_client: Any,
    provider_type: str = "ollama",
    model: Optional[str] = None,
    base_url: Optional[str] = None
)
```

### Factory Function
```python
create_embedder(
    provider_type: str = "ollama",
    http_client: Optional[Any] = None
) -> Embedder
```

**Returns:**
- `provider_type="mock"` → `MockEmbedder()` (no HTTP client)
- `provider_type="ollama"` → `RealEmbedder(http_client, "ollama")`
- `provider_type="openai"` → `RealEmbedder(http_client, "openai")`

## Integration with AppRuntime

### Production Runtime
```python
from app.services.embeddings import create_embedder

provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
embedding_provider = create_embedder(
    provider_type=provider_type,
    http_client=http_client
)
```

### Test Runtime
```python
from app.services.embeddings import create_embedder

embedding_provider = create_embedder(provider_type="mock")
```

## Test Results

### Runtime Tests (5/5 passing)
```
test_runtime.py::test_build_test_runtime_with_mock_provider PASSED
test_runtime.py::test_test_runtime_has_sync_task_runner PASSED
test_runtime.py::test_test_runtime_accepts_custom_providers PASSED
test_runtime.py::test_runtime_dataclass_structure PASSED
test_runtime.py::test_dummy_db_session_generator PASSED
```

### Embedder Tests (9/9 passing)
```
test_embeddings.py::test_mock_embedder_deterministic PASSED
test_embeddings.py::test_mock_embedder_discriminative PASSED
test_embeddings.py::test_mock_embedder_normalized PASSED
test_embeddings.py::test_mock_embedder_dimension_stable PASSED
test_embeddings.py::test_mock_embedder_custom_dimension PASSED
test_embeddings.py::test_mock_embedder_batch_processing PASSED
test_embeddings.py::test_mock_embedder_empty_input PASSED
test_embeddings.py::test_mock_embedder_similarity_behavior PASSED
test_embeddings.py::test_mock_embedder_no_http_client_required PASSED
```

### Runtime Configuration Verification
```
✅ Runtime Attributes:
   db_enabled: False
   http_client: None
   llm_provider: MockLLMProvider
   embedding_provider: MockEmbedder  ← Correctly using new embedder!
   task_runner: sync_task_runner
```

## Next Steps

### Integration with RAG Pipeline
To fully integrate the embedder into the retrieval pipeline, update `app/services/rag_service.py`:

**Current:**
```python
async def embed_texts(texts: List[str]):
    client = clients.get_http_client()
    # Direct HTTP calls to Ollama
    ...
```

**Target:**
```python
async def embed_texts(
    texts: List[str],
    tenant_id: str,
    embedder: Embedder
) -> List[List[float]]:
    return await embedder.embed_texts(texts, tenant_id)
```

**Callers should pass:**
```python
from app.runtime import runtime

embeddings = await embed_texts(texts, tenant_id, runtime.embedding_provider)
```

### Benefits
1. **CI/CD Testing:** No external dependencies (Ollama, OpenAI) required
2. **Deterministic Tests:** Same input → same output (predictable assertions)
3. **Fast Tests:** No network I/O (pure computation)
4. **Realistic Vectors:** Normalized unit-length vectors (valid for similarity)
5. **Clean Abstraction:** Easy to swap providers (mock ↔ real)
6. **Type Safety:** Protocol ensures interface compliance

## Environment Variables

### Production
```bash
LLM_PROVIDER=ollama  # or "openai"
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

### Testing
```bash
LLM_PROVIDER=mock
```

## Dimensions by Provider

| Provider | Model | Dimension |
|----------|-------|-----------|
| MockEmbedder | N/A | 384 (configurable) |
| Ollama | nomic-embed-text | 768 |
| OpenAI | text-embedding-3-small | 1536 |

## Key Implementation Details

### MockEmbedder Hash Algorithm
The SHA-256 based algorithm ensures:
- **Determinism:** Hash function guarantees same output for same input
- **Distribution:** Cryptographic hash provides good pseudo-random distribution
- **Normalization:** Unit length (||v|| = 1.0) ensures realistic similarity behavior
- **Dimension Flexibility:** Can extend hash to any dimension by re-hashing

### RealEmbedder Parallelization
- **Ollama:** Parallel async requests for batch efficiency
- **OpenAI:** Uses batch embedding API (single request with multiple inputs)
- **Error Handling:** Comprehensive logging and error messages

## Validation

All requirements met:
- ✅ Embedder protocol interface defined
- ✅ MockEmbedder uses SHA-256 (deterministic)
- ✅ MockEmbedder requires no HTTP client
- ✅ RealEmbedder supports Ollama and OpenAI
- ✅ Integrated into AppRuntime
- ✅ Factory function `create_embedder()` implemented
- ✅ Unit tests verify properties (9/9 passing)
- ✅ Runtime tests pass with new embedder (5/5 passing)

## Conclusion

The Embedder interface provides a clean abstraction for embeddings that supports both production (HTTP-based) and testing (deterministic mock) use cases. MockEmbedder enables fast, deterministic unit tests without external dependencies, while RealEmbedder provides production-ready integration with Ollama and OpenAI.
