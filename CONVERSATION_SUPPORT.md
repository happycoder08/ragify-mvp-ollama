# Conversation Support Implementation

## Overview
Implemented multi-turn conversation support for the RAGify system, enabling stateful chat sessions with conversation history included in LLM prompts.

## Architecture

### Database Models (app/models.py)

#### Conversation Model
```python
- id: Primary key
- tenant_id: Tenant identifier (for isolation)
- title: Conversation title
- created_at: Timestamp (auto-generated)
- updated_at: Timestamp (auto-updated)
- messages: Relationship to Message model (cascade delete)
```

#### Message Model
```python
- id: Primary key
- conversation_id: Foreign key to Conversation
- role: "user" or "assistant"
- content: Message text
- sources: JSON array of source documents (for assistant messages)
- created_at: Timestamp (auto-generated)
```

### REST API Endpoints (main.py)

#### POST /api/conversations
- **Purpose**: Create a new conversation
- **Request**: `{"title": "optional title"}`
- **Response**: Conversation object with id
- **Auth**: Required (JWT token)

#### GET /api/conversations
- **Purpose**: List user's conversations
- **Query Params**: None (auto-filtered by tenant_id)
- **Response**: Array of conversations with message_count
- **Limit**: 50 conversations, ordered by updated_at desc
- **Auth**: Required

#### GET /api/conversations/{id}
- **Purpose**: Get conversation with all messages
- **Response**: Conversation object with messages array
- **Auth**: Required (verifies ownership)

#### POST /api/conversations/{id}/messages
- **Request**: `{"role": "user|assistant", "content": "text", "sources": ["file1.pdf"]}`
- **Response**: Created message object
- **Side Effect**: Updates conversation.updated_at
- **Auth**: Required (verifies ownership)

#### DELETE /api/conversations/{id}
- **Purpose**: Delete conversation and all messages
- **Response**: Success message
- **Side Effect**: Cascade deletes all messages
- **Auth**: Required (verifies ownership)

#### POST /api/query (Updated)
- **New Field**: `conversation_id` (optional)
- **Behavior**:
  1. If conversation_id provided, retrieve last N messages
  2. Save user question to conversation
  3. Include conversation history in LLM prompt
  4. Stream response to client
  5. Save assistant response to conversation
- **Auth**: Required

### RAG Service Updates (app/services/rag_service.py)

#### query_collection()
- **New Parameter**: `conversation_history: List[Dict] = None`
- **Purpose**: Pass conversation context to LLM

#### _call_chat_model()
- **New Parameter**: `conversation_history: List[Dict] = None`
- **Behavior**: Build prompt with conversation history
- **Format**:
  ```
  User: [previous question 1]
  
  Assistant: [previous answer 1]
  
  User: [previous question 2]
  
  Assistant: [previous answer 2]
  
  Context:
  [Retrieved documents]
  
  Question: [current question]
  
  Answer:
  ```

#### answer_question()
- **New Parameter**: `conversation_history: List[Dict] = None`
- **Purpose**: Wrapper that passes history to query_collection

### Configuration (app/config.py)

#### New Setting: max_conversation_turns
- **Purpose**: Limit conversation history to prevent context overflow
- **Dev Mode**: 10 messages (5 user + 5 assistant)
- **Demo Mode**: 6 messages (3 user + 3 assistant)
- **Prod Mode**: 8 messages (4 user + 4 assistant)
- **Exported As**: `MAX_CONVERSATION_TURNS`

## Data Flow

### Creating a Conversation
1. Client → POST /api/conversations
2. Server creates Conversation record with tenant_id and title
3. Response includes conversation.id
4. Client stores conversation.id for future queries

### Querying with Conversation Context
1. Client → POST /api/query with conversation_id
2. Server retrieves last MAX_CONVERSATION_TURNS messages
3. Server saves user question as Message (role="user")
4. Server passes conversation_history to query_collection()
5. RAG service builds prompt with history + retrieved context
6. LLM generates response (streamed to client)
7. Server collects full response
8. Server saves assistant response as Message (role="assistant", sources=[...])
9. Server updates conversation.updated_at

### Message Storage Format
```json
{
  "role": "user",
  "content": "What are the remote work policies?",
  "sources": null,
  "created_at": "2025-01-15T10:30:00Z"
}

{
  "role": "assistant",
  "content": "Based on the company handbook, remote work is allowed...",
  "sources": ["handbook.pdf", "policies.txt"],
  "created_at": "2025-01-15T10:30:05Z"
}
```

## Key Features

### Tenant Isolation
- All conversations filtered by tenant_id
- Users can only access their own conversations
- Ownership verified on every operation

### Conversation History Context
- Last N messages included in LLM prompt
- Configurable per mode (dev/demo/prod)
- Ordered chronologically (oldest to newest)

### Automatic Message Persistence
- User questions saved before LLM call
- Assistant responses saved after streaming
- Sources attached to assistant messages
- Timestamps auto-generated

### Conversation Management
- List conversations with message counts
- Get full conversation with all messages
- Delete conversations (cascade to messages)
- Update conversation timestamps on new messages

### Streaming Support
- Full response collected during streaming
- Saved to database after streaming completes
- No impact on streaming performance

## Testing

### Test Script: test_conversation.py
Demonstrates complete conversation flow:
1. Login and get JWT token
2. List existing conversations
3. Create new conversation
4. First query (no context)
5. Second query (with first message context)
6. Third query (multi-turn context)
7. Get conversation with all messages
8. Verify message count
9. Delete conversation (cleanup)

### Running Tests
```bash
# Start server in demo mode
$env:RAGIFY_MODE="demo"; uvicorn main:app --reload

# In another terminal, run test
python test_conversation.py
```

### Expected Behavior
- Conversation created successfully
- Each query includes previous messages in context
- LLM responses reference earlier conversation
- Message count increments correctly
- Conversation deletion removes all messages

## Configuration Examples

### Dev Mode (Long History)
```python
"max_conversation_turns": 10  # Last 10 messages
```

### Demo Mode (Short History)
```python
"max_conversation_turns": 6  # Last 6 messages
```

### Prod Mode (Balanced)
```python
"max_conversation_turns": 8  # Last 8 messages
```

## Database Schema Changes

### New Tables
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    sources TEXT,  -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX idx_conversations_tenant ON conversations(tenant_id);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
```

## Migration Path

### Initial Migration
1. Update database schema (tables auto-created on startup)
2. Deploy updated code
3. Existing users start with zero conversations
4. Conversations created on-demand via API

### No Data Migration Required
- Existing documents/queries unaffected
- Conversations are opt-in feature
- Queries without conversation_id work as before

## Performance Considerations

### Database Queries
- **List conversations**: Single query with LIMIT 50
- **Get conversation**: Single query with JOIN (or relationship load)
- **Add message**: Two INSERTs (message + conversation update)
- **Delete conversation**: Single DELETE (cascade handled by DB)

### LLM Context Size
- Conversation history adds to prompt length
- Configurable limit prevents excessive context
- Trade-off: longer context = better coherence, slower inference

### Optimization Opportunities
1. **Caching**: Cache conversation history for active sessions
2. **Summarization**: Summarize old messages instead of full text
3. **Pagination**: Paginate conversation list if user has many
4. **Archiving**: Archive old conversations to separate table

## Security

### Authorization
- JWT token required for all endpoints
- Tenant ID extracted from token
- All queries filtered by tenant_id

### Validation
- Conversation ownership verified on access
- Message role validated (user|assistant only)
- Sources must be valid JSON array

### Data Isolation
- Each tenant's conversations are isolated
- No cross-tenant conversation access
- Cascade delete ensures no orphaned messages

## Future Enhancements

### Potential Features
1. **Conversation Sharing**: Share conversations between users
2. **Conversation Search**: Full-text search across messages
3. **Message Editing**: Edit/delete individual messages
4. **Branching**: Create conversation branches from specific messages
5. **Export**: Export conversation to markdown/PDF
6. **Analytics**: Track conversation metrics (length, topics, etc.)

### Prompt Improvements
1. **System Messages**: Add system-level instructions
2. **Context Windows**: Sliding window with summarization
3. **Persona**: Per-conversation persona/tone settings
4. **Templates**: Conversation starter templates

### Performance Improvements
1. **Async Saving**: Save messages asynchronously
2. **Batch Operations**: Bulk message operations
3. **Redis Caching**: Cache active conversations
4. **Read Replicas**: Separate read/write databases

## Documentation

### API Examples

#### Create Conversation
```bash
curl -X POST http://localhost:8000/api/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Q&A about HR Policies"}'
```

#### Query with Conversation
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the vacation policy?",
    "conversation_id": 1,
    "mode": "fast"
  }'
```

#### Get Conversation
```bash
curl http://localhost:8000/api/conversations/1 \
  -H "Authorization: Bearer $TOKEN"
```

#### Delete Conversation
```bash
curl -X DELETE http://localhost:8000/api/conversations/1 \
  -H "Authorization: Bearer $TOKEN"
```

## Summary

✅ **Implemented**:
- Conversation and Message SQLAlchemy models
- 5 REST endpoints for conversation CRUD
- Query endpoint updated to accept conversation_id
- Conversation history included in LLM prompts
- Automatic message persistence (user + assistant)
- Configurable conversation history limits
- Tenant isolation and ownership verification
- Comprehensive test script

✅ **Benefits**:
- Multi-turn conversations with memory
- Context carries over between queries
- Conversation history persisted in database
- Support for multiple concurrent conversations
- Scalable and secure architecture

✅ **Next Steps** (Optional):
- Update frontend to support conversations
- Add conversation search/filtering
- Implement conversation summarization for long chats
- Add conversation export functionality
