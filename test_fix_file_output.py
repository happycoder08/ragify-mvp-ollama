#!/usr/bin/env python3
"""Quick test: upload and query with the fix"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

with open("test_output.txt", "w") as log:
    def log_print(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()
    
    log_print("="*70)
    log_print("QUICK TEST - UPLOAD AND QUERY")
    log_print("="*70)

    # Login
    log_print("\n[1] Logging in...")
    r = requests.post(f"{BASE_URL}/api/login", json={"username": "demo", "password": "demo123"})
    token = r.json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}
    log_print("✓ Login successful")

    # Upload
    log_print("\n[2] Uploading document...")
    with open("demo_docs/Employee_Onboarding_Guide.txt", "rb") as f:
        r = requests.post(f"{BASE_URL}/api/upload", files={"files": ("Employee_Onboarding_Guide.txt", f, "text/plain")}, headers=h)
    log_print(f"✓ Upload: {r.json()['message']}")

    # Wait
    log_print("\n[3] Waiting for indexing (10 seconds)...")
    time.sleep(10)

    # Query
    log_print("\n[4] Testing query...")
    r = requests.post(f"{BASE_URL}/api/query", json={"question": "What time should I arrive on my first day?"}, headers=h, stream=True)

    answer = ""
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "token" in data:
            answer += data["token"]
            log_print(data["token"], end="")

    log_print("\n" + "="*70)
    if "8:00 AM" in answer:
        log_print("✓✓✓ SUCCESS - Contains '8:00 AM' ✓✓✓")
    else:
        log_print("✗✗✗ FAILED - Does not contain '8:00 AM' ✗✗✗")
        log_print(f"Got: {answer[:200]}")
