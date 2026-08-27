"""
End-to-end test for time/numeric anchor bonus in grounding gate.
Tests the full query flow with authentication.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Default demo user credentials (from demo mode)
USERNAME = "demo@example.com"
PASSWORD = "demo123"

def get_auth_token():
    """Get JWT token for default demo user."""
    response = requests.post(
        f"{BASE_URL}/api/login",
        json={"username": USERNAME, "password": PASSWORD}
    )
    response.raise_for_status()
    data = response.json()
    return data["access_token"]

def query_with_debug(token, question, mode="full", debug=2):
    """Query the API with debug output."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "mode": mode,
        "debug": debug
    }
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    
    # Parse SSE stream
    for line in response.text.strip().split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]  # Remove 'data: ' prefix
            if data_str.strip() == '[DONE]':
                continue
            try:
                data = json.loads(data_str)
                if 'debug' in data:
                    return data
            except json.JSONDecodeError:
                pass
    
    return None

def main():
    print("=" * 80)
    print("End-to-End Test: Time/Numeric Anchor Bonus in Grounding Gate")
    print("=" * 80)
    
    # Step 1: Authenticate
    print("\n[1] Authenticating...")
    token = get_auth_token()
    print(f"✓ Got auth token: {token[:20]}...")
    
    # Step 2: Query with time question
    question = "What time should I arrive on my first day?"
    print(f"\n[2] Querying: {question}")
    print(f"    Mode: full, Debug: 2")
    
    debug_data = query_with_debug(token, question, mode="full", debug=2)
    
    if not debug_data:
        print("❌ No debug data received")
        return
    
    # Step 3: Check grounding gate results
    print("\n[3] Grounding Gate Results:")
    grounding = debug_data.get('grounding_gate', {})
    
    print(f"    should_proceed: {grounding.get('should_proceed')}")
    print(f"    max_overlap: {grounding.get('max_overlap')}")
    print(f"    sum_top3: {grounding.get('sum_top3')}")
    print(f"    failed_check: {grounding.get('failed_check') or 'NONE'}")
    print(f"    evidence_lines_count: {grounding.get('evidence_lines_count')}")
    print(f"    thresholds: {grounding.get('thresholds')}")
    
    # Step 4: Verify time bonus was applied
    max_overlap = grounding.get('max_overlap', 0)
    sum_top3 = grounding.get('sum_top3', 0)
    should_proceed = grounding.get('should_proceed', False)
    
    print("\n[4] Verification:")
    if should_proceed:
        print("✓ Query PASSED grounding gate (as expected with time bonus)")
        print(f"  max_overlap={max_overlap} (>= 2 threshold)")
        print(f"  sum_top3={sum_top3} (>= 4 threshold)")
    else:
        print("❌ Query FAILED grounding gate (unexpected!)")
        print(f"  max_overlap={max_overlap} (threshold: 2)")
        print(f"  sum_top3={sum_top3} (threshold: 4)")
        print(f"  failed_check: {grounding.get('failed_check')}")
    
    # Step 5: Show selected chunks
    chunks = debug_data.get('chunks', [])
    print(f"\n[5] Selected Chunks ({len(chunks)}):")
    for idx, chunk in enumerate(chunks):
        print(f"    [{idx}] {chunk.get('header', 'no header')[:80]}")
        print(f"        distance={chunk.get('distance')}, source={chunk.get('source')}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
