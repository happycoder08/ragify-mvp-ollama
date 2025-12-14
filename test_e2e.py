#!/usr/bin/env python
"""End-to-end test for RAGify MVP"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DIR = Path(__file__).parent

print("\n" + "="*60)
print("RAGify End-to-End Test")
print("="*60 + "\n")

# 1. Check health
print("[1/5] Checking API health...")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"ERROR: {e}\n")
    exit(1)

# 1.5 Login to get token
print("[2/5] Logging in...")
try:
    login_data = {"username": "demo", "password": "demo123"}
    resp = requests.post(f"{BASE_URL}/api/login", json=login_data, timeout=5)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        auth_response = resp.json()
        token = auth_response.get("access_token")
        print(f"Token: {token[:20]}...\n")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print(f"ERROR: {resp.text}\n")
        exit(1)
except Exception as e:
    print(f"ERROR: {e}\n")
    exit(1)

# 2. Create a test document
print("[3/5] Creating test document...")
test_file = TEST_DIR / "test_document.txt"
test_content = """
Les Schwab Tire Centers

Les Schwab is a tire retail company based in Bend, Oregon. 
Founded in 1952, it operates tire shops across the western United States.
They specialize in tire sales, installation, and vehicle maintenance services.
Les Schwab is known for their customer service and tire rotation guarantees.
"""
test_file.write_text(test_content)
print(f"Created: {test_file}\n")

# 3. Upload document
print("[4/5] Uploading document...")
try:
    with open(test_file, 'rb') as f:
        files = {'files': f}
        resp = requests.post(f"{BASE_URL}/api/upload", files=files, headers=headers, timeout=30)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        print(f"File ID: {result.get('file_id', 'N/A')}")
        print(f"Message: {result.get('message', '')}\n")
    else:
        print(f"ERROR: {resp.text}\n")
        exit(1)
except Exception as e:
    print(f"ERROR: {e}\n")
    exit(1)

# Wait for indexing
time.sleep(2)

# 4. Query document
print("[5/5] Querying document...")
query_data = {
    "query": "What is Les Schwab and what do they do?",
    "top_k": 3
}
try:
    resp = requests.post(f"{BASE_URL}/api/query", json=query_data, headers=headers, timeout=30)
    print(f"Status: {resp.status_code}")
    
    if resp.status_code == 200:
        # Handle streaming response
        full_response = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "token" in data:
                        full_response += data["token"]
                        print(data["token"], end="", flush=True)
                    elif "metrics" in data:
                        metrics = data["metrics"]
                        print(f"\n\nMetrics:")
                        print(f"  Chunks retrieved: {metrics.get('num_chunks', 'N/A')}")
                        print(f"  Tokens generated: {metrics.get('tokens_generated', 'N/A')}")
                except json.JSONDecodeError:
                    pass
        print("\n")
    else:
        print(f"ERROR: {resp.text}\n")
        exit(1)
except Exception as e:
    print(f"ERROR: {e}\n")
    exit(1)

print("="*60)
print("TEST COMPLETE - All systems operational!")
print("="*60 + "\n")

# Cleanup
test_file.unlink()
print("Cleaned up test file.\n")
