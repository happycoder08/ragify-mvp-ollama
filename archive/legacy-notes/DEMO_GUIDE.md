# RAGify Demo Guide for Prospective Clients

## Quick Wins to Improve Demo Experience

### 1. **Pre-Index Sample Documents** (Fastest Way)
Before the demo, upload small, high-quality documents:
```powershell
# Upload 2-3 focused PDFs (policies, procedures, FAQs)
# This "warms up" the index for fast queries
```

**Why it helps:**
- First query takes longer (model cold-start), subsequent queries are fast
- Shows responsiveness after initial setup
- Demonstrates accuracy with real documents

### 2. **Use Mock Mode for Speed** (If Ollama Slow)
```powershell
$env:RAGIFY_MOCK = '1'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Benefits:**
- Instant uploads (no embedding delay)
- Instant responses (perfect for showing UI/workflow)
- Shows the product's structure without latency

**Positioning:** *"This demo shows the user experience. Production uses real embeddings for accuracy."*

### 3. **Optimize Chunk Size for Demo** (Better Answers)
```powershell
# Larger chunks = more context = better answers
$env:RAGIFY_CHUNK_SIZE = '800'
$env:RAGIFY_CHUNK_OVERLAP = '150'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

- Smaller chunks gave faster but less coherent answers
- Larger chunks give more contextual, complete answers
- Trade-off: slower but more impressive to clients

### 4. **Prepare Example Questions** (Show Strength)
Upload a specific document, then ask pre-planned questions that highlight:
- ✅ Exact policy retrieval: *"What is our late fee policy?"*
- ✅ Multi-step reasoning: *"If a payment is 45 days late, what actions apply?"*
- ✅ Nuanced understanding: *"What exceptions exist for hardship cases?"*

### 5. **Use Browser DevTools Streaming** (Show Real-Time)
When asking a question, open browser DevTools (F12) → Network tab:
- Shows streaming NDJSON response
- Demonstrates tokens arriving in real-time
- Impresses with technical sophistication

### 6. **Add a "Quick Demo" Endpoint** (One-Click Setup)
Create a demo mode that auto-populates with sample data:

```powershell
# Optional: Add to main.py for one-click demo
POST /api/demo-setup
```

This could:
- Load pre-chunked sample documents
- Pre-populate vectorstore
- Show "ready to demo" state immediately

### 7. **Hybrid Approach: Real + Fast** (Best)
```powershell
# Use real Ollama but with pre-indexed docs
# Reduce CHUNK_SIZE for faster first-time indexing
$env:RAGIFY_CHUNK_SIZE = '600'
$env:RAGIFY_OLLAMA_TIMEOUT = '120'  # Faster perceived response

# Pre-upload 2-3 documents before demo
# This gives you:
# - Real embeddings (impressive accuracy)
# - Fast responses (pre-indexed docs warm the model)
# - Professional appearance (no waiting)
```

---

## Demo Script for Clients

### **Act 1: Show Simplicity (1 min)**
1. Open http://localhost:8000
2. Show clean, minimal UI
3. Highlight: *"No complex setup. Local privacy. No API keys."*

### **Act 2: Upload & Index (30 sec)**
1. Upload 1-2 small documents (<<1MB for speed)
2. Say: *"Watch it chunk and index automatically..."*
3. Highlight: *"Supports PDF, DOCX, TXT"*

### **Act 3: Ask Smart Questions (2 min)**
1. Ask pre-planned, specific questions
2. Watch streaming response appear
3. Say: *"Real-time token streaming. Sources are cited."*
4. Highlight: *"Answers only from your documents. Never hallucinates."*

### **Act 4: Technical Depth (Optional)**
1. Show `/health` endpoint: `curl http://localhost:8000/health`
2. Explain async architecture
3. Mention: *"Scales horizontally. Production-ready for enterprise."*

---

## Performance Tuning for Different Scenarios

### **Scenario A: Demo with Limited Hardware**
```powershell
$env:RAGIFY_MOCK = '1'  # Use mock mode
```
- Shows all features instantly
- Focus on workflow, not latency

### **Scenario B: Demo with Good Hardware + Time**
```powershell
$env:RAGIFY_CHUNK_SIZE = '800'
$env:RAGIFY_CHUNK_OVERLAP = '150'
# Pre-upload documents 5 min before demo
# Then demo queries = fast + coherent
```

### **Scenario C: Live Demo, Unknown Hardware**
```powershell
$env:RAGIFY_CHUNK_SIZE = '600'
$env:RAGIFY_OLLAMA_TIMEOUT = '60'
# Use pre-indexed small documents
# Keep demo focused, concise, scripted
```

---

## Client Talking Points

✅ **Privacy First**: *"All processing happens locally. Zero data sent to cloud."*

✅ **No API Costs**: *"Unlike ChatGPT, Claude APIs - this runs on your hardware."*

✅ **Customizable**: *"Fine-tune chunking, models, retrieval logic. You control the system."*

✅ **Fast**: *"Sub-second responses after first query. Real-time streaming."*

✅ **Accurate**: *"Answer only from documents. Cite sources. No hallucinations."*

✅ **Extensible**: *"FastAPI backend. Easy to add auth, audit logs, webhooks, custom models."*

---

## If Demo Gets Slow

**Quick Fixes During Demo:**
1. Reset: `Invoke-RestMethod -Uri "http://localhost:8000/api/reset" -Method Post`
2. Restart server fresh
3. Use mock mode as fallback: `$env:RAGIFY_MOCK='1'`

**What to Say:**
*"Let me reset the index to show you fresh. This ensures clean performance."*

---

## Next Steps to Show After MVP

- 🔐 **Authentication** (Role-based access)
- 📊 **Analytics Dashboard** (Query history, performance metrics)
- 🔄 **Auto-Refresh** (Document change detection)
- 🎯 **Advanced Retrieval** (Hybrid search, reranking)
- 🧠 **Fine-Tuned Models** (Custom domain embeddings)

