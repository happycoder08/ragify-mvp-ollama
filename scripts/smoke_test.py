"""
Simple smoke-test script to exercise the upload -> index -> query flow.

Usage:
  1) Start the server: `& .\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
  2) Run this script from the repo root (venv activated):
       & .\.venv\Scripts\python.exe scripts\smoke_test.py

Assumes the server is running at http://localhost:8000
"""
import json
import sys
from typing import Any

import requests

BASE = "http://localhost:8000"
DEFAULT_TIMEOUT = 120


def upload_sample() -> Any:
    # Send an in-memory file under the 'files' field (matches frontend)
    files = {"files": ("sample_doc.txt", "Late fee policy: A late fee of $25 applies after 30 days.", "text/plain")}
    try:
        resp = requests.post(f"{BASE}/api/upload", files=files, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.ReadTimeout:
        print(f"Upload request timed out after {DEFAULT_TIMEOUT}s. Check server logs and Ollama availability.")
        raise
    print("Upload HTTP status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
    return resp


def query_sample() -> Any:
    payload = {"question": "What is the late fee policy?", "top_k": 4}
    try:
        resp = requests.post(f"{BASE}/api/query", json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.ReadTimeout:
        print(f"Query request timed out after {DEFAULT_TIMEOUT}s. The chat model or embedding call may be slow or unreachable.")
        raise
    print("Query HTTP status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
    return resp


def main() -> None:
    print("0) Checking server health (GET /)...")
    try:
        h = requests.get(f"{BASE}/", timeout=10)
        print("Root endpoint status:", h.status_code)
    except Exception as e:
        print("Failed to reach server root. Is the FastAPI server running?", str(e))
        sys.exit(1)

    print("1) Uploading sample document...")
    up = upload_sample()
    if up.status_code != 200:
        print("Upload failed — check server logs and Ollama/Chroma availability.")
        sys.exit(1)

    print("\n2) Sending a query to /api/query...")
    q = query_sample()
    if q.status_code != 200:
        print("Query failed — see server logs or ensure models are loaded in Ollama.")
        sys.exit(1)

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
