# Server-Sent Events (SSE) Streaming Format

The `/api/query` endpoint returns a streaming response using Server-Sent Events (SSE).

## Content Type
```
text/event-stream
```

## Event Flow

### 1. Debug Event (Always First)
Sent immediately with metadata about the query and retrieval process.

```
event: debug
data: {
  "evidence_count": 3,
  "sources_count": 3,
  "retrieved_count": 10,
  "selected_count": 3,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "tenant_id": "default",
  "collection_name": "documents_default",
  "collection_count": 42,
  "doc_ids_filter": null,
  "top10_scores": [0.89, 0.85, 0.82, 0.78, 0.75, 0.71, 0.68, 0.65, 0.62, 0.59],
  "grounding_gate": {
    "passed": true,
    "min_score": 0.82,
    "max_score": 0.89,
    "avg_score": 0.85
  }
}

```

**When debug=0 (default)**: Only includes `evidence_count`, `sources_count`, and minimal metadata.

**When debug>=1**: Includes all fields shown above for troubleshooting.

---

### 2a. Token Events (Success Case)
For successful queries with grounding, the LLM response streams token by token.

```
event: token
data: {"t": "The"}

event: token
data: {"t": " company"}

event: token
data: {"t": " policy"}

event: token
data: {"t": " states"}

event: token
data: {"t": " that"}

event: token
data: {"t": " employees"}

event: token
data: {"t": " must"}

event: token
data: {"t": "..."}

```

Each token event contains a single field `t` with the text fragment.

---

### 2b. Refusal (No Evidence Case)
If no relevant evidence is found or the grounding gate rejects the query, the stream skips tokens and goes directly to the final event.

**No token events are sent.**

---

### 3. Final Event (Always Last)

#### Success Response (with answer)
```
event: final
data: {
  "answer": "The company policy states that employees must submit expense reports within 30 days of incurring the expense. Late submissions require manager approval.",
  "refused": false,
  "refusal_reason": null,
  "evidence": [
    {
      "snippet": "All expense reports must be submitted within 30 days of the date the expense was incurred...",
      "chunk_id": "chunk_1",
      "heading": "Expense Reporting Policy",
      "doc_id": null
    },
    {
      "snippet": "Late submissions (beyond 30 days) require written approval from your direct manager...",
      "chunk_id": "chunk_5",
      "heading": "Late Submission Process",
      "doc_id": null
    },
    {
      "snippet": "Expense reports should include all receipts and a brief description of each expense...",
      "chunk_id": "chunk_3",
      "heading": "Required Documentation",
      "doc_id": null
    }
  ],
  "sources": [
    {
      "doc_id": null,
      "filename": "employee_handbook.pdf",
      "chunk_id": "chunk_1"
    },
    {
      "doc_id": null,
      "filename": "employee_handbook.pdf",
      "chunk_id": "chunk_5"
    },
    {
      "doc_id": null,
      "filename": "expense_policy.pdf",
      "chunk_id": "chunk_3"
    }
  ],
  "debug_info": null
}

```

#### Refusal Response (no evidence found)
```
event: final
data: {
  "answer": "The document does not specify this.",
  "refused": true,
  "refusal_reason": "NOT_FOUND",
  "evidence": [],
  "sources": [],
  "debug_info": {
    "evidence_count": 0,
    "sources_count": 0,
    "retrieved_count": 10,
    "selected_count": 0,
    "grounding_gate": {
      "passed": false,
      "failed_check": "min_score_threshold"
    }
  }
}

```

**Refusal Reasons:**
- `NOT_FOUND`: No relevant chunks found
- `LOW_CONFIDENCE`: Retrieved chunks did not meet grounding threshold
- `CONTEXT_TOO_LARGE`: Context exceeds model limits

---

## Complete Example Stream

```
event: debug
data: {"evidence_count": 2, "sources_count": 2, "retrieved_count": 8, "selected_count": 2}

event: token
data: {"t": "According"}

event: token
data: {"t": " to"}

event: token
data: {"t": " the"}

event: token
data: {"t": " vacation"}

event: token
data: {"t": " policy"}

event: token
data: {"t": ","}

event: token
data: {"t": " employees"}

event: token
data: {"t": " receive"}

event: token
data: {"t": " 15"}

event: token
data: {"t": " days"}

event: token
data: {"t": " of"}

event: token
data: {"t": " paid"}

event: token
data: {"t": " time"}

event: token
data: {"t": " off"}

event: token
data: {"t": " annually"}

event: token
data: {"t": "."}

event: final
data: {"answer": "According to the vacation policy, employees receive 15 days of paid time off annually.", "refused": false, "refusal_reason": null, "evidence": [{"snippet": "Full-time employees are entitled to 15 days of paid vacation per year...", "chunk_id": "chunk_2", "heading": "Vacation Entitlement", "doc_id": null}, {"snippet": "Vacation days must be requested at least 2 weeks in advance...", "chunk_id": "chunk_7", "heading": "Requesting Time Off", "doc_id": null}], "sources": [{"doc_id": null, "filename": "hr_policies.pdf", "chunk_id": "chunk_2"}, {"doc_id": null, "filename": "hr_policies.pdf", "chunk_id": "chunk_7"}], "debug_info": null}

```

---

## Client Implementation Guide

### Parsing SSE Stream (JavaScript/TypeScript)

```typescript
import { QueryFinalResponse, DebugInfo } from './types';

async function queryRAG(question: string, token: string) {
  const response = await fetch('http://localhost:8000/api/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ question, debug: 0 })
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  let buffer = '';
  let debugInfo: DebugInfo | null = null;
  let answer = '';
  let finalResponse: QueryFinalResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        const event = line.substring(7);
        continue;
      }

      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.substring(6));

        // Handle different event types based on previous event line
        if ('evidence_count' in data) {
          // Debug event
          debugInfo = data as DebugInfo;
          console.log('Debug info:', debugInfo);
        } else if ('t' in data) {
          // Token event
          answer += data.t;
          console.log('Token:', data.t);
          // Update UI with streaming answer
        } else if ('answer' in data) {
          // Final event
          finalResponse = data as QueryFinalResponse;
          console.log('Final response:', finalResponse);
          // Update UI with final answer and sources
        }
      }
    }
  }

  return { debugInfo, answer, finalResponse };
}
```

### React Hook Example

```typescript
import { useState, useCallback } from 'react';
import { QueryFinalResponse, DebugInfo } from './types';

export function useRAGQuery() {
  const [streaming, setStreaming] = useState(false);
  const [answer, setAnswer] = useState('');
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [finalResponse, setFinalResponse] = useState<QueryFinalResponse | null>(null);

  const query = useCallback(async (question: string, token: string) => {
    setStreaming(true);
    setAnswer('');
    setDebugInfo(null);
    setFinalResponse(null);

    const response = await fetch('/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ question })
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.substring(7);
        } else if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6));

          if (currentEvent === 'debug') {
            setDebugInfo(data);
          } else if (currentEvent === 'token') {
            setAnswer(prev => prev + data.t);
          } else if (currentEvent === 'final') {
            setFinalResponse(data);
            setStreaming(false);
          }
        }
      }
    }
  }, []);

  return { query, streaming, answer, debugInfo, finalResponse };
}
```

---

## Error Handling

If the request fails (invalid token, rate limit, etc.), you'll receive a standard HTTP error instead of an SSE stream:

```json
{
  "detail": "Invalid credentials"
}
```

**Status Codes:**
- `401`: Unauthorized (invalid/missing token)
- `400`: Bad request (empty question, invalid params)
- `429`: Rate limit exceeded
- `500`: Server error

---

## Headers Required

```
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

Get the JWT token from `POST /api/login` first.

---

## Query Parameters vs Request Body

All query parameters are sent in the **request body** as JSON, not URL params:

```typescript
// ✅ Correct
fetch('/api/query', {
  method: 'POST',
  body: JSON.stringify({
    question: "What is the vacation policy?",
    mode: "fast",
    top_k: 5,
    debug: 1
  })
})

// ❌ Wrong
fetch('/api/query?question=...&mode=fast')
```
