#!/usr/bin/env python3
"""Quick test to verify server configuration"""

import requests

BASE_URL = "http://localhost:8000"

# Test 1: Server is up
try:
    response = requests.get(f"{BASE_URL}/docs", timeout=5)
    if response.status_code == 200:
        print("Server is running and responsive")
    else:
        print(f"Server returned status {response.status_code}")
except Exception as e:
    print(f"Server not running: {e}")
    exit(1)

# Test 2: Login works
try:
    login_response = requests.post(
        f"{BASE_URL}/api/login",
        json={"username": "demo", "password": "demo123"},
        timeout=5
    )
    if login_response.status_code == 200:
        token = login_response.json()["access_token"]
        print(f"Login successful - got token: {token[:30]}...")
    else:
        print(f"Login failed: {login_response.status_code}")
except Exception as e:
    print(f"Login error: {e}")
    exit(1)

print("\nServer is ready for testing!")
