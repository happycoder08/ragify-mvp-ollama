#!/usr/bin/env python3
"""Quick test: upload and query with the fix"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("="*70)
print("QUICK TEST - UPLOAD AND QUERY")
print("="*70)

# Login
print("\n[1] Logging in...")
r = requests.post(f"{BASE_URL}/api/login", json={"username": "demo", "password": "demo123"})
token = r.json()["access_token"]
h = {"Authorization": f"Bearer {token}"}
print("✓ Login successful")

# Upload
print("\n[2] Uploading document...")
with open("demo_docs/Employee_Onboarding_Guide.txt", "rb") as f:
    r = requests.post(f"{BASE_URL}/api/upload", files={"files": ("Employee_Onboarding_Guide.txt", f, "text/plain")}, headers=h)
print(f"✓ Upload: {r.json()['message']}")

# Wait
print("\n[3] Waiting for indexing (10 seconds)...")
time.sleep(10)

# Query
print("\n[4] Testing query...")
r = requests.post(f"{BASE_URL}/api/query", json={"question": "What time should I arrive on my first day?"}, headers=h, stream=True)

answer = ""
for line in r.iter_lines():
    if not line:
        continue
    data = json.loads(line)
    if "token" in data:
        answer += data["token"]
        print(data["token"], end="", flush=True)

print("\n" + "="*70)
if "8:00 AM" in answer:
    print("✓✓✓ SUCCESS - Contains '8:00 AM' ✓✓✓")
else:
    print("✗✗✗ FAILED - Does not contain '8:00 AM' ✗✗✗")
    print(f"Got: {answer[:200]}")
