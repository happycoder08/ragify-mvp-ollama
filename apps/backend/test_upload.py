#!/usr/bin/env python3
"""Test script to upload document and verify indexing"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

# Step 1: Login
print("[1/4] Logging in...")
login_response = requests.post(
    f"{BASE_URL}/api/login",
    json={"username": "demo", "password": "demo123"}
)
if login_response.status_code != 200:
    print(f"✗ Login failed: {login_response.text}")
    exit(1)

token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"✓ Login successful")

# Step 2: Upload file
print("\n[2/4] Uploading Employee_Onboarding_Guide.txt...")
with open("demo_docs/Employee_Onboarding_Guide.txt", "rb") as f:
    files = {"files": f}
    upload_response = requests.post(
        f"{BASE_URL}/api/upload",
        files=files,
        headers=headers
    )

if upload_response.status_code != 200:
    print(f"✗ Upload failed: {upload_response.text}")
    exit(1)

upload_data = upload_response.json()
print(f"✓ Upload response: {upload_data}")

# Step 3: Wait for background indexing
print("\n[3/4] Waiting for background indexing (5 seconds)...")
time.sleep(5)

# Step 4: Check if document is indexed
print("\n[4/4] Checking indexed documents...")
docs_response = requests.get(
    f"{BASE_URL}/api/documents",
    headers=headers
)

if docs_response.status_code != 200:
    print(f"✗ Documents check failed: {docs_response.text}")
    exit(1)

docs_data = docs_response.json()
print(f"✓ Documents response: {json.dumps(docs_data, indent=2)[:500]}...")

num_docs = len(docs_data.get("documents", []))
print(f"\n📊 RESULT: {num_docs} documents indexed")

if num_docs > 0:
    print("✓ Document indexing successful!")
    # Try a test query
    print("\n[BONUS] Testing query retrieval...")
    query_payload = {"question": "What time should I arrive on my first day?"}
    query_response = requests.post(
        f"{BASE_URL}/api/query",
        json=query_payload,
        headers=headers,
        stream=True
    )
    
    if query_response.status_code == 200:
        print("Query response:")
        for line in query_response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "token" in chunk:
                        print(chunk["token"], end="", flush=True)
                except:
                    pass
        print("\n✓ Query completed")
    else:
        print(f"✗ Query failed: {query_response.text}")
else:
    print("✗ No documents indexed - background task may not be running")
