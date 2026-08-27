# RAGify AI – Testing Without Docker/Postgres

## Option 1: Skip Postgres (Recommended for Quick Testing)

The application gracefully handles missing Postgres. You can test most features without it:

### What Works Without Postgres:
- ✅ Login/authentication
- ✅ JWT token management
- ✅ Tenant branding
- ✅ Query existing documents (if ChromaDB has data)
- ✅ All public endpoints

### What Requires Postgres:
- ❌ Document upload (will fail)
- ❌ Document list view

### Quick Test Without Postgres:

```powershell
# 1. Activate venv (if not already active)
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app (it will warn about DB but continue)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Database connection test failed - continuing without Postgres. Metadata storage will be unavailable.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test Login & Authentication:

1. Open: `http://localhost:8000/static/login.html`
2. Login as: `demo` / `demo123`
3. You'll be redirected to main app
4. Tenant branding will load (blue theme)
5. Document list will show error (expected without Postgres)
6. Query will work if you have existing ChromaDB data

---

## Option 2: Install Postgres on Windows (No Docker)

### Download & Install:
1. Download from: https://www.postgresql.org/download/windows/
2. Run installer (PostgreSQL 15+)
3. During install:
   - Set password (remember it!)
   - Default port: 5432
   - Install pgAdmin 4 (optional GUI)

### Create Database:

```powershell
# Open Command Prompt or PowerShell as Admin
# Login to psql (replace 'postgres' with your username if different)
psql -U postgres

# In psql prompt:
CREATE USER ragify WITH PASSWORD 'ragify';
CREATE DATABASE ragify_db OWNER ragify;
\q
```

### Set Environment Variable:

```powershell
# PowerShell
$env:DATABASE_URL = "postgresql://ragify:ragify@localhost:5432/ragify_db"

# Or permanently:
[System.Environment]::SetEnvironmentVariable('DATABASE_URL', 'postgresql://ragify:ragify@localhost:5432/ragify_db', 'User')
```

### Start App:

```powershell
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Option 3: Use SQLite Instead (Code Modification)

If you want to avoid Postgres entirely, we can modify the code to use SQLite:

### Update `app/database.py`:

```python
# Change this line:
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./ragify.db"  # SQLite instead of Postgres
)

# Change engine creation:
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
```

### Update `requirements.txt`:

Remove or comment out:
```
# psycopg2-binary  # Not needed for SQLite
```

SQLite is built into Python, so no additional database installation needed!

---

## Option 4: Mock Mode (UI Testing Only)

To test just the UI without any database or Ollama:

```powershell
$env:RAGIFY_MOCK = "1"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

This mode:
- Skips all Ollama calls
- Returns canned responses
- Doesn't require Postgres
- Good for testing UI/UX only

---

## Recommended Path for You:

Since you don't have Docker, I recommend **Option 1** (Skip Postgres) to quickly test authentication:

```powershell
# Install dependencies
pip install -r requirements.txt

# Start app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then test:
1. **Login**: http://localhost:8000/static/login.html
   - Try all 3 users (demo, acme_admin, finance_user)
   - Verify different branding for each tenant
2. **Logout**: Click "Sign Out" button
3. **Auth Check**: Try accessing http://localhost:8000/ without logging in
   - Should redirect to login page

The upload/query features will require either:
- Installing Postgres locally (Option 2), OR
- Switching to SQLite (Option 3)

**Which option would you like to pursue?** I can help you set up any of these.
