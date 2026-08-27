#!/usr/bin/env python3
"""
Verification script for synchronous upload behavior in CI/inline mode.

Demonstrates:
1. Upload returns with status="indexed" immediately
2. No polling needed before querying
3. Evidence available in first query after upload
"""

import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app
from app.runtime import build_test_runtime

def test_sync_upload():
    """Test synchronous upload and immediate query."""
    
    # Build test runtime (inline mode)
    runtime = build_test_runtime()
    
    # Import and override main.runtime
    import main
    main.runtime = runtime
    
    # Create test client
    client = TestClient(app)
    
    # Login
    login_resp = client.post("/api/login", json={"username": "test", "password": "test123"})
    if login_resp.status_code != 200:
        print(f"Login failed: {login_resp.status_code} - {login_resp.text}")
        sys.exit(1)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Sample document
    doc_content = """
    Employee Benefits Policy
    
    Vacation Policy:
    - All full-time employees receive 15 days of paid vacation per year
    - Vacation days accrue monthly at 1.25 days per month
    - Unused vacation can be carried over up to 5 days
    
    Health Insurance:
    - Company provides comprehensive health insurance
    - Coverage begins on first day of employment
    - Family coverage available
    """
    
    print("1. Uploading document...")
    files = {"files": ("benefits.txt", doc_content.encode(), "text/plain")}
    upload_resp = client.post("/api/upload", files=files, headers=headers)
    
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    
    print(f"   Status: {upload_data['status']}")
    print(f"   Message: {upload_data['message']}")
    
    # Verify synchronous completion
    assert "indexed successfully" in upload_data["message"], "Expected synchronous message"
    assert len(upload_data["documents"]) == 1
    
    doc = upload_data["documents"][0]
    print(f"   Document ID: {doc['id']}")
    print(f"   Document Status: {doc['status']}")
    print(f"   Filename: {doc['filename']}")
    
    assert doc["status"] == "indexed", f"Expected indexed, got {doc['status']}"
    print("   ✓ Document indexed synchronously")
    
    # Query IMMEDIATELY (no sleep)
    print("\n2. Querying immediately after upload (no polling)...")
    query_resp = client.post(
        "/api/query",
        json={"question": "How many vacation days do employees get?", "mode": "full"},
        headers=headers
    )
    
    assert query_resp.status_code == 200
    
    # Parse SSE response
    import json
    events = []
    for line in query_resp.text.split("\n"):
        if line.startswith("event: "):
            event_type = line[7:].strip()
            events.append({"type": event_type})
        elif line.startswith("data: ") and events:
            events[-1]["data"] = json.loads(line[6:])
    
    # Find debug and final events
    debug_event = next((e for e in events if e["type"] == "debug"), None)
    final_event = next((e for e in events if e["type"] == "final"), None)
    
    assert debug_event is not None, "Expected debug event"
    assert final_event is not None, "Expected final event"
    
    debug_data = debug_event["data"]
    final_data = final_event["data"]
    
    print(f"   Evidence count: {debug_data['evidence_count']}")
    print(f"   Retrieved count: {debug_data.get('retrieved_count', 'N/A')}")
    print(f"   Selected count: {debug_data.get('selected_count', 'N/A')}")
    
    assert debug_data["evidence_count"] > 0, "Expected evidence immediately after upload"
    assert final_data["refused"] is False, "Should not refuse when evidence available"
    
    print(f"   Answer: {final_data['answer'][:100]}...")
    print("   ✓ Query returned evidence immediately")
    
    print("\n✅ Synchronous upload verification PASSED")
    print("   - Upload completes indexing before returning")
    print("   - No polling needed between upload and query")
    print("   - Evidence available in first query")

if __name__ == "__main__":
    test_sync_upload()
