"""
Test debug mode functionality for /api/query endpoint.
"""
import requests
import json

API_BASE = "http://127.0.0.1:8000"

def authenticate():
    """Get JWT token."""
    auth_resp = requests.post(
        f"{API_BASE}/api/login",
        json={"username": "demo", "password": "demo123"}
    )
    auth_resp.raise_for_status()
    return auth_resp.json()["access_token"]

def test_debug_mode():
    """Test query with debug=1 to get detailed retrieval diagnostics."""
    token = authenticate()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test with debug=0 (default, legacy mode)
    print("\n[1] Testing with debug=0 (legacy mode)...")
    resp0 = requests.post(
        f"{API_BASE}/api/query",
        json={
            "question": "What time should I arrive on my first day?",
            "debug": 0
        },
        headers=headers,
        stream=True
    )
    
    debug_obj_0 = None
    for line in resp0.iter_lines():
        if not line:
            continue
        try:
            event = json.loads(line)
            if "debug" in event:
                debug_obj_0 = event["debug"]
                break
        except json.JSONDecodeError:
            continue
    
    print(f"  Debug object keys (debug=0): {list(debug_obj_0.keys()) if debug_obj_0 else 'None'}")
    if debug_obj_0:
        print(f"  - evidence_count: {debug_obj_0.get('evidence_count')}")
        print(f"  - sources_count: {debug_obj_0.get('sources_count')}")
        print(f"  - has retrieved_count: {'retrieved_count' in debug_obj_0}")
        print(f"  - has selected_count: {'selected_count' in debug_obj_0}")
    
    # Test with debug=1 (detailed diagnostics)
    print("\n[2] Testing with debug=1 (detailed diagnostics)...")
    resp1 = requests.post(
        f"{API_BASE}/api/query",
        json={
            "question": "What time should I arrive on my first day?",
            "debug": 1
        },
        headers=headers,
        stream=True
    )
    
    debug_obj_1 = None
    for line in resp1.iter_lines():
        if not line:
            continue
        try:
            event = json.loads(line)
            if "debug" in event:
                debug_obj_1 = event["debug"]
                break
        except json.JSONDecodeError:
            continue
    
    print(f"  Debug object keys (debug=1): {list(debug_obj_1.keys()) if debug_obj_1 else 'None'}")
    if debug_obj_1:
        print(f"  - evidence_count: {debug_obj_1.get('evidence_count')}")
        print(f"  - sources_count: {debug_obj_1.get('sources_count')}")
        print(f"  - retrieved_count: {debug_obj_1.get('retrieved_count')}")
        print(f"  - selected_count: {debug_obj_1.get('selected_count')}")
        
        chunks = debug_obj_1.get("selected_chunks", [])
        print(f"  - Number of chunks: {len(chunks)}")
        
        if chunks:
            print("\n  First chunk details:")
            chunk = chunks[0]
            print(f"    - id: {chunk.get('id')}")
            print(f"    - header: {chunk.get('header', '')[:80]}")
            print(f"    - snippet: {chunk.get('snippet', '')[:100]}")
            print(f"    - distance: {chunk.get('distance')}")
    
    print("\n[3] Summary:")
    print(f"  debug=0 works: {'retrieved_count' not in (debug_obj_0 or {})}")
    print(f"  debug=1 works: {'retrieved_count' in (debug_obj_1 or {})}")
    print(f"  Chunk details included: {bool(debug_obj_1 and debug_obj_1.get('selected_chunks'))}")

if __name__ == "__main__":
    test_debug_mode()
