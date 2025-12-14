#!/usr/bin/env python3
"""Enhanced e2e test with diagnostics"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("="*70)
print("RAGIFY DEMO - END-TO-END TEST (DIAGNOSTIC)")
print("="*70)

# Step 1: Login
print("\n[1/4] Logging in...")
login_response = requests.post(
    f"{BASE_URL}/api/login",
    json={"username": "demo", "password": "demo123"}
)
if login_response.status_code != 200:
    print(f"✗ Login failed: {login_response.text}")
    exit(1)
    
token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("✓ Login successful")

# Step 2: Upload document
print("\n[2/4] Uploading Employee_Onboarding_Guide.txt...")
try:
    with open("demo_docs/Employee_Onboarding_Guide.txt", "rb") as f:
        files = {"files": ("Employee_Onboarding_Guide.txt", f, "text/plain")}
        upload_response = requests.post(
            f"{BASE_URL}/api/upload",
            files=files,
            headers=headers,
            timeout=30
        )
except Exception as e:
    print(f"✗ Upload failed with error: {e}")
    exit(1)

if upload_response.status_code == 200:
    print("✓ Upload successful")
    print(f"  Response: {upload_response.json()['message']}")
else:
    print(f"✗ Upload failed: {upload_response.status_code}")
    print(f"  Response: {upload_response.text}")
    exit(1)

# Step 2b: Wait for indexing with retry
print("\n[2b/4] Waiting for document indexing (max 30s)...")
indexed = False
for i in range(10):
    time.sleep(3)
    print(f"  Wait {(i+1)*3}s...", end=" ", flush=True)
    
    # Try querying to see if data exists
    query_resp = requests.post(
        f"{BASE_URL}/api/query",
        json={"question": "test"},
        headers=headers,
        timeout=10
    )
    
    if query_resp.status_code == 200:
        indexed = True
        print("✓ Index ready")
        break
    print()

if not indexed:
    print("✗ Timed out waiting for indexing")

# Step 3: Check documents endpoint (if available)
print("\n[3/4] Checking indexed documents...")
try:
    docs_resp = requests.get(
        f"{BASE_URL}/api/documents",
        headers=headers,
        timeout=10
    )
    if docs_resp.status_code == 200:
        docs = docs_resp.json()
        print(f"✓ {len(docs)} documents found")
        for doc in docs:
            print(f"  - {doc.get('filename')} ({doc.get('status')})")
    else:
        print(f"  No /api/documents endpoint")
except:
    print(f"  Could not check documents")

# Step 4: Test query
print("\n[4/4] Testing query: 'What time should I arrive on my first day?'")
print("-" * 70)

question = "What time should I arrive on my first day?"
query_response = requests.post(
    f"{BASE_URL}/api/query",
    json={"question": question},
    headers=headers,
    timeout=30,
    stream=True
)

if query_response.status_code != 200:
    print(f"✗ Query failed: {query_response.status_code}")
    print(f"  Response: {query_response.text}")
    exit(1)

answer_text = ""
sources = []

# Stream the response
for line in query_response.iter_lines():
    if not line:
        continue
    try:
        data = json.loads(line)
        if "token" in data:
            token_text = data["token"]
            answer_text += token_text
            print(token_text, end="", flush=True)
        elif "sources" in data:
            sources = data["sources"]
    except json.JSONDecodeError:
        print(f"\n[Parse error: {line}]")

print("\n" + "-" * 70)

# Check answer quality
if "8:00 AM" in answer_text or "8:00" in answer_text:
    print(f"\n✓ CORRECT ANSWER - Contains '8:00 AM'")
elif "I don't have enough information" in answer_text or "could not find" in answer_text:
    print(f"\n✗ INCORRECT ANSWER - Generic 'not found' response")
    print(f"  This suggests documents were not indexed or chunks were filtered out")
else:
    print(f"\n? UNCLEAR ANSWER - Check manually")

if sources:
    print(f"\nSources ({len(sources)}):")
    for src in sources:
        print(f"  - {src}")
else:
    print(f"\nNo sources provided")

print("\n" + "="*70)
