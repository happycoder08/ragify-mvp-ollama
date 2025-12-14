#!/usr/bin/env python3
"""Upload document and test query end-to-end"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("="*70)
print("RAGIFY DEMO - END-TO-END TEST")
print("="*70)

# Step 1: Login
print("\n[1/3] Logging in...")
login_response = requests.post(
    f"{BASE_URL}/api/login",
    json={"username": "demo", "password": "demo123"}
)
token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("✓ Login successful")

# Step 2: Upload document
print("\n[2/3] Uploading Employee_Onboarding_Guide.txt...")
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
    exit(1)

# Wait for document to finish indexing
print("  Waiting for indexing to complete...")
doc_id = None
for _ in range(30):  # ~30s max
    try:
        docs_resp = requests.get(f"{BASE_URL}/api/documents", headers=headers, timeout=5)
        if docs_resp.status_code != 200:
            time.sleep(1)
            continue
        docs = docs_resp.json().get("documents", [])
        for d in docs:
            if d.get("filename") == "Employee_Onboarding_Guide.txt":
                doc_id = d.get("id")
                status = d.get("status")
                if status == "indexed":
                    break
        if doc_id and status == "indexed":
            break
    except Exception:
        pass
    time.sleep(1)

if not doc_id:
    print("✗ Could not find uploaded doc in /api/documents after waiting")
    exit(1)
if status != "indexed":
    print(f"✗ Doc not indexed in time (status={status})")
    exit(1)
print(f"  ✓ Indexed (doc_id={doc_id})")

# Step 3: Test query
print("\n[3/3] Testing query: 'What time should I arrive on my first day?'")
print("-" * 70)

query_payload = {
    "question": "What time should I arrive on my first day?",
    "doc_ids": [doc_id],
    "top_k": 8,
    "mode": "fast"
}
query_response = requests.post(
    f"{BASE_URL}/api/query",
    json=query_payload,
    headers=headers,
    stream=True,
    timeout=60
)

if query_response.status_code == 200:
    print("\nANSWER:")
    answer = ""
    for line in query_response.iter_lines():
        if line:
            try:
                chunk = json.loads(line)
                if "token" in chunk:
                    token_text = chunk["token"]
                    print(token_text, end="", flush=True)
                    answer += token_text
            except:
                pass
    print("\n")
    
    # Check if answer contains key information
    if "8:00" in answer and ("main reception" in answer or "3rd floor" in answer):
        print("✓ CORRECT ANSWER! Contains '8:00' and location information")
    elif "8:00" in answer:
        print("✓ PARTIAL SUCCESS - Found '8:00' but missing location")
    else:
        print("✗ INCORRECT ANSWER - Does not contain '8:00 AM'")
        print(f"\nExpected: 8:00 AM at main reception on 3rd floor")
        print(f"Got: {answer[:200]}...")
else:
    print(f"✗ Query failed: {query_response.status_code}")
    print(query_response.text)

print("\n" + "="*70)
