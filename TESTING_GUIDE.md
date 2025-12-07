# RAGify AI – Multi-Tenant MVP Testing Guide

## Quick Start Testing (Without Postgres)

If you just want to test the authentication and RAG functionality without setting up Postgres:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (in separate terminal)
ollama serve

# 3. Pull required models
ollama pull nomic-embed-text
ollama pull llama3

# 4. Start the app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Note**: Without Postgres, the upload endpoint will fail, but you can still test login and query against any previously indexed documents in ChromaDB.

---

## Full Testing (With Postgres)

### Step 1: Start Postgres

Using Docker:
```bash
docker run --name ragify-postgres \
  -e POSTGRES_USER=ragify \
  -e POSTGRES_PASSWORD=ragify \
  -e POSTGRES_DB=ragify_db \
  -p 5432:5432 \
  -d postgres:15
```

Or use existing Postgres and set environment variable:
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
```

### Step 2: Start Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On startup, the app will:
- Test Postgres connection
- Create the `documents` table automatically
- Initialize in-memory user store

### Step 3: Test Authentication

Open browser to `http://localhost:8000/static/login.html`

#### Test Users

| Username | Password | Tenant | Description |
|----------|----------|--------|-------------|
| `demo` | `demo123` | `default` | Demo user (blue branding) |
| `acme_admin` | `acme123` | `acme` | ACME Corp (red branding) |
| `finance_user` | `finance123` | `finance` | Finance tenant (green branding) |

**Try logging in as each user and verify**:
- Login successful with JWT token stored in localStorage
- Redirect to main app (`/`)
- Tenant branding applied (title, colors, disclaimer)
- User name displayed in header

### Step 4: Test Upload & Indexing

1. **Login as `demo` user**
2. **Upload test documents**:
   - Create a simple `.txt` file with some content
   - Or use the demo script: `python scripts/setup_demo.py`
   - Upload via the UI
3. **Verify**:
   - Status shows "Uploading and indexing..."
   - Success message shows indexed chunks count
   - Document appears in "Your Documents" list with status "indexed"
4. **Check Postgres**:
   ```bash
   docker exec -it ragify-postgres psql -U ragify -d ragify_db
   
   SELECT * FROM documents;
   ```
   Should show your uploaded document with `tenant_id='default'` and `status='indexed'`

5. **Check ChromaDB**:
   ```bash
   ls vectorstore/
   ```
   Should see tenant-specific collection data

### Step 5: Test Query with Streaming

1. **Ask a question** related to your uploaded document
2. **Verify**:
   - "Thinking..." status appears
   - Answer streams in token by token
   - Sources listed below answer
   - Answer is relevant to uploaded documents

### Step 6: Test Tenant Isolation

#### Test A: Upload to Different Tenants

1. **Login as `demo`** and upload `policy1.txt`
2. **Logout** and **login as `acme_admin`**
3. **Upload `policy2.txt`**
4. **Verify in Postgres**:
   ```sql
   SELECT tenant_id, filename, status FROM documents;
   ```
   Should show:
   - `default | policy1.txt | indexed`
   - `acme | policy2.txt | indexed`

#### Test B: Query Isolation

1. **Login as `demo`**
2. **Ask about content from `policy1.txt`** → Should get answer
3. **Ask about content from `policy2.txt`** → Should say "not found"
4. **Logout and login as `acme_admin`**
5. **Ask about content from `policy2.txt`** → Should get answer
6. **Ask about content from `policy1.txt`** → Should say "not found"

**Result**: Users can only access their own tenant's documents ✓

#### Test C: Cross-Tenant Access Attempt

Try to manually call API with wrong tenant:
```bash
# Login as demo user and get token
TOKEN="<your_demo_token>"

# Try to access ACME documents (should fail with 403)
curl -X GET "http://localhost:8000/api/documents/acme" \
  -H "Authorization: Bearer $TOKEN"

# Expected: {"detail":"Access denied. User belongs to tenant 'default'"}
```

### Step 7: Test JWT Expiration & Logout

1. **Logout** from the UI
2. **Verify**:
   - Redirected to login page
   - localStorage cleared (token removed)
3. **Try to access main app** without logging in
   - Should auto-redirect to login

### Step 8: Test Error Handling

#### Bad Login
- Try logging in with wrong password → Error message shown

#### Upload Without Files
- Click upload without selecting files → Alert shown

#### Empty Question
- Click "Ask" with empty question → Alert shown

#### Invalid Token
- Manually edit localStorage token to invalid value
- Refresh page → Redirected to login

---

## API Testing with curl

### 1. Login and Get Token
```bash
curl -X POST http://localhost:8000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}'

# Save the access_token from response
```

### 2. Get Tenant Config
```bash
curl http://localhost:8000/api/config/default
```

### 3. List Tenants
```bash
curl http://localhost:8000/api/tenants
```

### 4. Upload Document (requires auth)
```bash
TOKEN="<your_token>"

curl -X POST "http://localhost:8000/api/upload?tenant_id=default" \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@test.txt"
```

### 5. List Documents (requires auth)
```bash
curl -X GET http://localhost:8000/api/documents/default \
  -H "Authorization: Bearer $TOKEN"
```

### 6. Query (requires auth, streaming response)
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "question": "What is the policy?",
    "top_k": 4
  }'
```

---

## Troubleshooting

### Issue: "Database connection failed"
- **Check**: Is Postgres running?
  ```bash
  docker ps | grep ragify-postgres
  ```
- **Fix**: Start Postgres or set correct `DATABASE_URL`

### Issue: "Table documents does not exist"
- **Check**: App should auto-create table on startup
- **Fix**: Restart app or manually run:
  ```sql
  CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(100) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'indexing',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  );
  ```

### Issue: "Ollama connection timeout"
- **Check**: Is Ollama running? `ollama list`
- **Fix**: Start Ollama: `ollama serve`

### Issue: Upload works but documents not indexed
- **Check app logs** for errors during embedding
- **Verify models pulled**: `ollama list` should show `nomic-embed-text` and `llama3`

### Issue: "Invalid token" after login
- **Check browser console** for errors
- **Clear localStorage**: Run in console: `localStorage.clear()`
- **Try logging in again**

---

## Performance Testing

### Test Concurrent Users
```bash
# Terminal 1: Login as demo
TOKEN_DEMO="<demo_token>"

# Terminal 2: Login as acme
TOKEN_ACME="<acme_token>"

# Concurrent queries
curl -X POST http://localhost:8000/api/query -H "Authorization: Bearer $TOKEN_DEMO" -H "Content-Type: application/json" -d '{"tenant_id":"default","question":"test"}' &

curl -X POST http://localhost:8000/api/query -H "Authorization: Bearer $TOKEN_ACME" -H "Content-Type: application/json" -d '{"tenant_id":"acme","question":"test"}' &
```

### Test Large Document Upload
```bash
# Create 10MB test file
python -c "print('test content\n' * 100000)" > large_test.txt

# Upload and measure time
time curl -X POST "http://localhost:8000/api/upload?tenant_id=default" \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@large_test.txt"
```

---

## Security Checklist

- [ ] JWT tokens expire after 24 hours
- [ ] Passwords hashed with bcrypt (never stored plain)
- [ ] Users cannot access other tenants' data (403 error)
- [ ] Invalid tokens redirect to login (401 error)
- [ ] All protected endpoints require authentication
- [ ] Tenant ID verified against JWT token on every request

---

## Database Inspection

### View All Documents
```sql
SELECT 
  id, 
  tenant_id, 
  filename, 
  status, 
  created_at 
FROM documents 
ORDER BY created_at DESC;
```

### Count Documents Per Tenant
```sql
SELECT 
  tenant_id, 
  COUNT(*) as doc_count,
  COUNT(CASE WHEN status = 'indexed' THEN 1 END) as indexed_count,
  COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
FROM documents 
GROUP BY tenant_id;
```

### Find Failed Uploads
```sql
SELECT * FROM documents WHERE status = 'failed';
```

---

## Clean Up After Testing

### Reset Everything
```bash
# Stop app (Ctrl+C)

# Remove vector store
rm -rf vectorstore/

# Remove uploaded files
rm -rf app/uploads/*

# Reset Postgres (if needed)
docker exec -it ragify-postgres psql -U ragify -d ragify_db -c "TRUNCATE documents;"
```

### Stop Postgres
```bash
docker stop ragify-postgres
docker rm ragify-postgres
```

---

## Next Steps After Testing

If all tests pass:
1. Update default credentials in `app/auth.py`
2. Set secure `JWT_SECRET_KEY` environment variable
3. Configure production `DATABASE_URL`
4. Add rate limiting
5. Add logging/monitoring
6. Deploy to production environment
