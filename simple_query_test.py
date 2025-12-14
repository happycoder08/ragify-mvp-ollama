#!/usr/bin/env python3
"""Simple HTTP test of query endpoint"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Get token
login_response = requests.post(
    f"{BASE_URL}/api/login",
    json={"username": "demo", "password": "demo123"}
)
token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Test query
query_payload = {"question": "What time should I arrive on my first day?"}
print("Question: What time should I arrive on my first day?")
print("\nWaiting for response...\n")

query_response = requests.post(
    f"{BASE_URL}/api/query",
    json=query_payload,
    headers=headers,
    stream=True,
    timeout=60
)

if query_response.status_code == 200:
    print("ANSWER:")
    for line in query_response.iter_lines():
        if line:
            try:
                chunk = json.loads(line)
                if "token" in chunk:
                    print(chunk["token"], end="", flush=True)
            except:
                pass
    print("\n")
else:
    print(f"Error: {query_response.status_code} - {query_response.text}")
