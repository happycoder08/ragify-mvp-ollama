# CI/CD Testing Guide

This guide explains how to run RAGify tests in CI/CD environments without external dependencies.

## Mock LLM Provider

RAGify includes a **MockLLMProvider** for testing without Ollama or OpenAI. It returns deterministic, keyword-based responses and supports both grounded and ungrounded answer modes for validation testing.

### Configuration

Enable the mock provider via environment variable:

```bash
export LLM_PROVIDER=mock
```

### Modes

#### Grounded Mode (Default)

Returns answers that match the document content:

```bash
# Run integration tests with realistic mock answers
LLM_PROVIDER=mock pytest test_integration.py -v
```

**Example responses:**
- "vacation days" → "15 days per year"
- "sick leave" → "not specified in the documents"
- "onboarding time" → "8:00 AM on the 3rd floor"

#### Ungrounded Mode

Returns hallucinated answers for validation testing:

```bash
# Enable ungrounded mode
export MOCK_UNGROUNDED=true

# Run validation tests
LLM_PROVIDER=mock MOCK_UNGROUNDED=true pytest test_integration.py::test_ungrounded_answer_validation -v
```

**Example responses:**
- "vacation days" → "30 days per year" (hallucinated, should be rejected by validation)
- "benefits" → "comprehensive health insurance" (hallucinated)

The validation pipeline should detect and reject these ungrounded answers.

## CI/CD Configuration Examples

### GitHub Actions

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      
      - name: Run grounded integration tests
        env:
          LLM_PROVIDER: mock
        run: pytest test_integration.py -v
      
      - name: Run validation tests (ungrounded mode)
        env:
          LLM_PROVIDER: mock
          MOCK_UNGROUNDED: true
        run: pytest test_integration.py::test_ungrounded_answer_validation -v
```

### GitLab CI

```yaml
test:
  image: python:3.11
  
  before_script:
    - pip install -r requirements.txt
    - pip install pytest pytest-asyncio
  
  script:
    # Grounded mode tests
    - export LLM_PROVIDER=mock
    - pytest test_integration.py -v
    
    # Ungrounded validation tests
    - export MOCK_UNGROUNDED=true
    - pytest test_integration.py::test_ungrounded_answer_validation -v
```

### Jenkins

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-asyncio'
            }
        }
        
        stage('Integration Tests') {
            environment {
                LLM_PROVIDER = 'mock'
            }
            steps {
                sh 'pytest test_integration.py -v'
            }
        }
        
        stage('Validation Tests') {
            environment {
                LLM_PROVIDER = 'mock'
                MOCK_UNGROUNDED = 'true'
            }
            steps {
                sh 'pytest test_integration.py::test_ungrounded_answer_validation -v'
            }
        }
    }
}
```

## Available Tests

### Full Integration Test Suite

```bash
# Run all 7 integration tests
LLM_PROVIDER=mock pytest test_integration.py -v
```

**Tests included:**
1. `test_full_workflow` - Login → upload → query → verify sources
2. `test_unrelated_query_refusal` - Refused queries return exact refusal message
3. `test_tenant_isolation` - Cross-tenant access is blocked
4. `test_empty_collection_query` - Empty collection returns refusal
5. `test_multiple_documents` - Multi-file upload and querying
6. `test_authentication_failures` - Auth edge cases
7. `test_ungrounded_answer_validation` - Validation rejects hallucinated answers

### Individual Test Scenarios

```bash
# Test workflow (upload + query)
LLM_PROVIDER=mock pytest test_integration.py::test_full_workflow -v

# Test validation rejection
LLM_PROVIDER=mock MOCK_UNGROUNDED=true pytest test_integration.py::test_ungrounded_answer_validation -v

# Test tenant isolation
LLM_PROVIDER=mock pytest test_integration.py::test_tenant_isolation -v
```

## Test Assertions

### Grounded Mode Expectations

Tests expect specific keyword-based responses:

```python
# Vacation query
response = query("How many vacation days?")
assert "15 days" in response  # Matches document content

# Sick leave query
response = query("What's the sick leave policy?")
assert "not specified" in response  # Document doesn't mention it

# Onboarding query
response = query("When should new employees arrive?")
assert "8:00 AM" in response and "3rd floor" in response
```

### Ungrounded Mode Expectations

Validation should reject hallucinated answers:

```python
# With MOCK_UNGROUNDED=true
response_metadata = query("How many vacation days?")

# Mock returns "30 days" (hallucinated)
assert not response_metadata["answer_supported"]  # Validation rejects
assert "cannot answer" in response_text  # Replaced with refusal
assert "30 days" not in response_text  # Hallucination removed
```

## Local Development

### Quick Test

```bash
# Test with mock provider (no Ollama required)
LLM_PROVIDER=mock python -m pytest test_integration.py::test_full_workflow -v -s
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export LLM_PROVIDER=mock

pytest test_integration.py -v -s --log-cli-level=DEBUG
```

### Test Coverage

```bash
# Run with coverage report
LLM_PROVIDER=mock pytest test_integration.py --cov=app --cov-report=html
```

## Troubleshooting

### Mock Provider Not Loading

**Error:** `ValueError: Unsupported LLM_PROVIDER`

**Solution:** Ensure environment variable is set before running tests:

```bash
# Check current setting
echo $LLM_PROVIDER

# Set explicitly
export LLM_PROVIDER=mock
pytest test_integration.py -v
```

### Test Responses Don't Match

**Error:** `AssertionError: Expected "15 days" in response`

**Solution:** Check keyword matching in MockLLMProvider:

```python
# In app/services/llm_providers.py
grounded_responses = {
    "vacation": "15 days per year",  # Matches "vacation" keyword
    "sick": "not specified",         # Matches "sick" keyword
    # ...
}
```

Update test queries to use matching keywords.

### Validation Tests Failing

**Error:** Validation not detecting ungrounded answers

**Solution:** Verify MOCK_UNGROUNDED is set:

```bash
# Must be "true" (lowercase)
export MOCK_UNGROUNDED=true

# Check it's set
env | grep MOCK_UNGROUNDED
```

## Best Practices

### 1. Always Use Mock in CI

```yaml
# ✅ Good - explicitly set mock provider
env:
  LLM_PROVIDER: mock

# ❌ Bad - relies on default (may use Ollama)
# No LLM_PROVIDER set
```

### 2. Test Both Modes

```yaml
# Test grounded answers work correctly
- run: LLM_PROVIDER=mock pytest test_integration.py

# Test validation catches hallucinations
- run: LLM_PROVIDER=mock MOCK_UNGROUNDED=true pytest test_integration.py::test_ungrounded_answer_validation
```

### 3. Use Specific Test Selection

```bash
# ✅ Good - run specific tests
pytest test_integration.py::test_full_workflow -v

# ❌ Slow - runs all tests including slow e2e tests
pytest -v
```

### 4. Check Exit Codes

```bash
# CI should fail on test failures
LLM_PROVIDER=mock pytest test_integration.py -v
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "Tests failed"
  exit 1
fi
```

## Environment Variables Reference

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `LLM_PROVIDER` | `ollama`, `openai`, `mock` | `ollama` | Select LLM backend |
| `MOCK_UNGROUNDED` | `true`, `false` | `false` | Enable hallucinated answers |
| `OLLAMA_BASE_URL` | URL | `http://localhost:11434` | Ollama server address |
| `OPENAI_API_KEY` | API key | - | OpenAI authentication |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` | `INFO` | Logging verbosity |

## Mock Provider Implementation

The MockLLMProvider is implemented in `app/services/llm_providers.py`:

```python
class MockLLMProvider:
    """
    Mock LLM provider for CI/CD testing without external dependencies.
    
    Features:
    - Keyword-based deterministic responses
    - Grounded mode: realistic answers matching document content
    - Ungrounded mode: hallucinated answers for validation testing
    - Character-by-character streaming simulation
    - on_first_token callback support
    """
    
    async def generate_stream(self, prompt, tenant_id, ...):
        # Returns grounded or ungrounded response based on keywords
        # Streams character-by-character to simulate real LLM
        ...
```

For implementation details, see [app/services/llm_providers.py](app/services/llm_providers.py#L204-L295).

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md) - Full testing documentation
- [Setup Guide](SETUP_WITHOUT_DOCKER.md) - Local development setup
- [Configuration Guide](CONFIG_GUIDE.md) - Environment variables reference
