# Grounding Gate Feature

## Overview

The **grounding gate** is a deterministic safety mechanism that validates retrieved evidence before calling the LLM. It prevents hallucination by refusing to answer questions when the retrieved chunks lack sufficient lexical grounding.

## How It Works

The grounding gate performs three checks on retrieved chunks:

### 1. Evidence Lines Check
- Extracts the most relevant lines from chunks based on lexical overlap with the query
- Uses token-level matching after stopword filtering
- Lines shorter than 10 characters are filtered out
- Returns top `MAX_EVIDENCE_LINES` (default: 6) lines per chunk

### 2. Support Score Check
- Computes the maximum token overlap count across all extracted evidence lines
- Must meet minimum threshold: `support_score >= MIN_SUPPORT` (default: 2)
- Uses deterministic sorting: lines sorted by (overlap_count, line_length) descending

### 3. Numeric/Time Anchor Check
- Detects if question is numeric/time-sensitive (contains digits or time keywords)
- If yes, verifies that evidence contains numeric/time patterns
- Patterns include: digits, time formats (HH:MM), AM/PM markers

## Configuration

Constants in `app/services/rag_service.py`:

```python
MIN_SUPPORT = 2              # Minimum token overlap count required
MAX_EVIDENCE_LINES = 6       # Max evidence lines to extract per chunk
```

## Refusal Response

When any check fails, returns:

```json
{
  "answer": null,
  "evidence": [],
  "sources": [],
  "refused": true,
  "refusal_reason": "NOT_FOUND"
}
```

With debug info (when `debug >= 1`):

```json
{
  "retrieved_count": 10,
  "selected_count": 3,
  "chunks": [],
  "refused": true,
  "refusal_reason": "NOT_FOUND",
  "support_score": 1.0,
  "evidence_lines_count": 2
}
```

## Examples

### Refused: Low Support Score

**Query:** "What are the quantum computing policies?"  
**Retrieved chunks:** General onboarding documents (no quantum mentions)  
**Result:** Refused (support_score < 2)

### Refused: Numeric Question Without Numeric Evidence

**Query:** "How many vacation days do I get?"  
**Retrieved chunks:** "The vacation policy is generous and flexible."  
**Result:** Refused (numeric question but no numbers in evidence)

### Passed: Strong Lexical Grounding

**Query:** "What documents do I need to bring?"  
**Retrieved chunks:** "Bring your ID, signed offer letter, and work authorization."  
**Result:** Proceeds to LLM (support_score >= 2, good overlap)

### Passed: Numeric Question With Numeric Evidence

**Query:** "How many vacation days?"  
**Retrieved chunks:** "Employees receive 15 days of vacation per year."  
**Result:** Proceeds to LLM (support_score >= 2, has numeric anchor "15")

## Implementation Details

### Functions

**`extract_evidence_lines(chunk_text, question, max_lines=6)`**
- Tokenizes question and chunk text
- Filters stopwords
- Scores each line by token overlap
- Returns top lines sorted by (overlap, length) descending

**`_compute_grounding_gate(question, selected_chunks, chunk_ids)`**
- Extracts evidence lines from all chunks
- Computes support score (max overlap)
- Checks numeric/time requirements if applicable
- Returns: (should_proceed, refusal_reason, evidence_lines, support_score)

### Integration Point

Located in `query_collection()` function after evidence construction:

```python
# GROUNDING GATE: Check if evidence is sufficient before calling LLM
should_proceed, refusal_reason, gate_evidence_lines, gate_support_score = _compute_grounding_gate(
    question, filtered_results, ids
)

if not should_proceed:
    # Return refusal response
    async def refusal_gen():
        yield ""
    
    return refusal_gen(), [], [], "", refusal_debug_info
```

## Testing

### Unit Tests
Run `test_grounding_gate_simple.py`:
- Tests empty chunks, strong evidence, low support
- Tests numeric questions with/without numeric evidence
- Tests time questions with/without time patterns
- Validates deterministic behavior

### Integration Tests
Run `test_grounding_gate_integration.py`:
- Tests refusal responses via `/api/query` endpoint
- Validates debug info format
- Checks end-to-end behavior with real server

## Design Principles

1. **Deterministic**: No randomness, no model calls, stable sorting
2. **General**: Works for any question type, no question-specific keywords
3. **Transparent**: Logs gate decision and metrics for debugging
4. **Safe**: Prefers refusing over hallucinating when evidence is weak

## Tuning

Adjust constants based on your use case:

- **Increase `MIN_SUPPORT`** (e.g., to 3) for stricter grounding
- **Decrease `MIN_SUPPORT`** (e.g., to 1) for more permissive retrieval
- **Increase `MAX_EVIDENCE_LINES`** for longer context windows
- **Modify time/numeric patterns** in `_compute_grounding_gate()` for custom detection

## Limitations

- **Lexical matching only**: Doesn't understand semantic equivalence (e.g., "15 days" vs "three weeks")
- **Stopword filtering**: May miss important short words
- **No stemming**: "schedule" and "scheduled" treated as different tokens
- **Language-specific**: Assumes English text and stopwords

For semantic matching, consider adding sentence embeddings or using the reranker scores instead of pure lexical overlap.
