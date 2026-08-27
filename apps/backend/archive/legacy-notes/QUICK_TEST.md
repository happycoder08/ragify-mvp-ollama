# Quick End-to-End Testing Guide

## Prerequisites Check

### 1. Verify Ollama is Running
```powershell
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama (depends on your installation)
# Then pull required models:
ollama pull nomic-embed-text
ollama pull llama3
```

### 2. Verify Server is Running
```powershell
# Should see: INFO: Application startup complete
# If not running:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## End-to-End Test (Without PostgreSQL)

### Step 1: Login
1. Open browser: http://localhost:8000
2. You'll be redirected to login page
3. Login with:
   - Username: `demo`
   - Password: `demo123`
4. ✅ **Success**: You should see the main interface with blue branding

### Step 2: Create a Test Document
Create a simple text file to upload:

```powershell
# Create test document
@"
Company Policy: Remote Work

1. Employees can work remotely up to 3 days per week.
2. Core hours are 10 AM to 3 PM in your local timezone.
3. All remote workers must have a reliable internet connection.
4. Video must be enabled during team meetings.
5. Remote work requests must be submitted 24 hours in advance.

Benefits of Remote Work:
- Improved work-life balance
- Reduced commute time
- Increased productivity
- Lower stress levels

Contact HR at hr@company.com for questions.
"@ | Out-File -FilePath "test_policy.txt" -Encoding UTF8
```

### Step 3: Upload Document
1. In the web interface, click the **upload area** or **"Upload Selected Files"** button
2. Select `test_policy.txt`
3. Click **"Upload Selected Files"**
4. ✅ **Success**: You should see a success message
5. ⚠️ **Note**: Without PostgreSQL, you won't see the document in the list, but it's indexed!

### Step 4: Query the Document

**Test Query 1: Simple Fact Retrieval**
- Question: `How many days per week can employees work remotely?`
- ✅ **Expected**: "3 days per week" or "up to 3 days"
- ✅ **Sources**: Should show `test_policy.txt`

**Test Query 2: Information Synthesis**
- Question: `What are the benefits of remote work?`
- ✅ **Expected**: Should list: work-life balance, reduced commute, productivity, lower stress
- ✅ **Sources**: Should show `test_policy.txt`

**Test Query 3: Contact Information**
- Question: `Who should I contact about remote work questions?`
- ✅ **Expected**: "HR at hr@company.com"

**Test Query 4: Out-of-Scope (Should Fail)**
- Question: `What is the company's vacation policy?`
- ✅ **Expected**: "I could not find anything relevant" or "I don't know"

### Step 5: Validate Response Quality

**Good Response Indicators:**
- ✅ Answers are **directly from the document** (not hallucinated)
- ✅ Includes **source attribution** (shows filename)
- ✅ Says "I don't know" for information **not in the document**
- ✅ Response is **streaming** (appears word by word)

**Red Flags:**
- ❌ Answers include information **not in the uploaded document**
- ❌ Claims to know things that aren't in the document
- ❌ Sources list is empty

### Step 6: Test Tenant Isolation

1. **Logout** (click logout button)
2. **Login as different tenant**:
   - Username: `acme_admin`
   - Password: `acme123`
3. ✅ **Success**: Interface should change to **red branding**
4. Ask the same question: `How many days per week can employees work remotely?`
5. ✅ **Expected**: "I could not find anything relevant" (tenant isolation working!)

## End-to-End Test (With PostgreSQL)

### Option 1: Install PostgreSQL on Windows

#### Download and Install
1. Download PostgreSQL 15: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
2. Run the installer (select default options)
3. Remember the password you set for `postgres` user
4. Default port: `5432`

#### Create Database and User
```powershell
# Open PowerShell as Administrator
# Connect to PostgreSQL (enter password when prompted)
psql -U postgres

# In psql prompt, run:
CREATE DATABASE ragify_db;
CREATE USER ragify WITH PASSWORD 'ragify';
GRANT ALL PRIVILEGES ON DATABASE ragify_db TO ragify;
\q
```

#### Verify Connection
```powershell
# Test connection
psql -U ragify -d ragify_db -h localhost

# If successful, you'll see: ragify_db=>
# Type \q to exit
```

### Option 2: Use Docker (Easier!)

```powershell
# Start PostgreSQL in Docker
docker run --name ragify-postgres `
  -e POSTGRES_USER=ragify `
  -e POSTGRES_PASSWORD=ragify `
  -e POSTGRES_DB=ragify_db `
  -p 5432:5432 `
  -d postgres:15

# Verify it's running
docker ps | Select-String ragify-postgres

# Test connection
docker exec -it ragify-postgres psql -U ragify -d ragify_db
```

### Initialize Database

#### Restart the Server
```powershell
# Stop the server (Ctrl+C in the terminal where uvicorn is running)

# Start it again
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Look for this in the output:
# INFO:main:Initializing database...
# INFO:app.database:Initializing database...
# INFO:main:Database initialized successfully
```

#### Verify Tables Created
```powershell
# Connect to database
psql -U ragify -d ragify_db -h localhost

# List tables
\dt

# Should see: documents table

# Check table structure
\d documents

# Exit
\q
```

### Repeat Upload Test with PostgreSQL

1. **Clear browser localStorage**: F12 → Console → `localStorage.clear()`
2. **Refresh and login** with `demo` / `demo123`
3. **Upload `test_policy.txt` again**
4. ✅ **Success**: You should now see the document in the **"Your Documents"** list
5. Status should change: `indexing` → `indexed`
6. **Query the document** as before
7. **Check database**:
```powershell
psql -U ragify -d ragify_db -h localhost

SELECT * FROM documents;
# Should see your uploaded document
```

## Advanced Testing

### Test Multiple Documents

Create more test files:

```powershell
# Benefits policy
@"
Employee Benefits

Health Insurance:
- Company covers 100% of employee premiums
- Family coverage available at 50% employer contribution
- Dental and vision included

Retirement:
- 401k with 5% company match
- Vests after 2 years of employment
"@ | Out-File -FilePath "test_benefits.txt" -Encoding UTF8

# PTO policy
@"
Paid Time Off Policy

Vacation Days:
- 0-2 years: 15 days per year
- 3-5 years: 20 days per year
- 6+ years: 25 days per year

Sick Leave:
- 10 days per year (unlimited for serious illness)

Holidays:
- 12 paid holidays including your birthday
"@ | Out-File -FilePath "test_pto.txt" -Encoding UTF8
```

Upload both files, then test:

**Query 1**: `What health insurance benefits do we have?`
- ✅ **Expected**: Info about health, dental, vision coverage
- ✅ **Sources**: Should show `test_benefits.txt`

**Query 2**: `How many vacation days do I get?`
- ✅ **Expected**: Should mention the tiered system (15/20/25 days)
- ✅ **Sources**: Should show `test_pto.txt`

**Query 3**: `What are all the employee benefits including PTO?`
- ✅ **Expected**: Should combine info from **both documents**
- ✅ **Sources**: Should show **both** `test_benefits.txt` and `test_pto.txt`

### Test PDF Upload

1. Create or download a sample PDF
2. Upload it through the interface
3. Query its contents
4. ✅ **Success**: Should be able to retrieve information from PDF

### Performance Benchmarks

**Expected Response Times** (depends on hardware):
- Login: < 1 second
- Upload (small file): 2-5 seconds
- Query response (start): 1-3 seconds
- Query response (complete): 5-10 seconds

## Troubleshooting

### Issue: "I could not find anything relevant"
**Causes:**
- Document not fully indexed yet (wait a few seconds)
- Query doesn't match document content closely
- Wrong tenant (check tenant isolation)

**Solutions:**
- Wait 5-10 seconds after upload
- Rephrase question to match document wording
- Check if logged in as correct user

### Issue: Answers seem wrong or hallucinated
**Causes:**
- LLM is generating content not in the document
- Retrieved context isn't relevant

**Solutions:**
- Check the **Sources** - do they contain the answer?
- Try more specific questions
- Check chunk_size in `app/services/ingestion.py` (default: 800 chars)

### Issue: Slow responses
**Causes:**
- Ollama loading models into memory (first query)
- Large document requiring many chunks
- Slow hardware

**Solutions:**
- First query always slower (model loading)
- Use smaller `top_k` value (default: 4)
- Consider smaller embedding model

### Issue: Database not connecting
```powershell
# Check if PostgreSQL is running
# Windows service:
Get-Service -Name postgresql*

# Docker:
docker ps | Select-String postgres

# Test connection:
psql -U ragify -d ragify_db -h localhost
```

## Validation Checklist

✅ **Authentication**
- [ ] Login successful with demo/demo123
- [ ] Logout successful
- [ ] Login with different tenant shows different branding
- [ ] Invalid credentials rejected

✅ **Document Upload**
- [ ] TXT file uploads successfully
- [ ] PDF file uploads successfully
- [ ] Success message shown
- [ ] Document appears in list (if PostgreSQL running)

✅ **Document Query**
- [ ] Can retrieve specific facts from document
- [ ] Source attribution shown
- [ ] Says "don't know" for out-of-scope questions
- [ ] Answers are accurate to document content

✅ **Tenant Isolation**
- [ ] Different tenants see different data
- [ ] Tenant branding works (colors, logos)
- [ ] Cannot access other tenant's documents

✅ **Error Handling**
- [ ] Works without PostgreSQL (graceful degradation)
- [ ] Handles Ollama connection errors
- [ ] Shows meaningful error messages

## Next Steps

1. **Production Setup**: See README.md for production deployment checklist
2. **Add More Tenants**: Edit `app/tenant_config.py` and `app/auth.py`
3. **Customize Branding**: Update tenant configs with your logos/colors
4. **Performance Tuning**: Adjust chunk_size, top_k, and model selection

---

**Need Help?** Check:
- Main README.md for architecture overview
- TESTING_GUIDE.md for comprehensive testing
- SETUP_WITHOUT_DOCKER.md for alternative setups
