#!/usr/bin/env python
"""Simple test to verify upload and retrieval"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DIR = Path(__file__).parent

print("\n" + "="*60)
print("RAGify Upload & Retrieval Test")
print("="*60 + "\n")

# 1. Login
print("[1/4] Logging in...")
login_resp = requests.post(
    f"{BASE_URL}/api/login",
    json={"username": "demo", "password": "demo123"},
    timeout=5
)
if login_resp.status_code != 200:
    print(f"ERROR: {login_resp.text}")
    exit(1)

token = login_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("✓ Login successful\n")

# 2. Create test file
print("[2/4] Creating test document...")
test_file = TEST_DIR / "simple_test.txt"
test_content = "This is a test document about Python programming. Python is a great language."
test_file.write_text(test_content)
print(f"✓ Created: {test_file}\n")

# 3. Upload file
print("[3/4] Uploading document...")
with open(test_file, "rb") as f:
    files = {"files": (test_file.name, f)}
    upload_resp = requests.post(
        f"{BASE_URL}/api/upload",
        files=files,
        headers=headers,
        timeout=30
    )

if upload_resp.status_code != 200:
    print(f"ERROR: {upload_resp.text}")
    exit(1)

upload_data = upload_resp.json()
print(f"✓ Upload successful: {upload_data['message']}")
print(f"  Files processed: {upload_data['files_processed']}\n")

# 4. Check documents list
print("[4/4] Verifying documents appear...")
import time
time.sleep(2)  # Wait for background processing

docs_resp = requests.get(f"{BASE_URL}/api/documents", headers=headers, timeout=5)
if docs_resp.status_code != 200:
    print(f"ERROR: {docs_resp.text}")
    exit(1)

docs = docs_resp.json()
doc_list = docs.get("documents", [])
print(f"✓ Documents retrieved: {len(doc_list)}")

for doc in doc_list:
    print(f"  - {doc['filename']} (status: {doc['status']})")

if any(d["filename"] == test_file.name for d in doc_list):
    print("\n✓ SUCCESS: Uploaded document appears in list!")
else:
    print("\n✗ FAILED: Document not found in list")
    exit(1)

print("\n" + "="*60)
print("All tests passed!")
print("="*60 + "\n")
