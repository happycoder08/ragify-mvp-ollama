"""
Test evidence selection for email signature question.
Verify that evidence includes "Email signature" and "Arial" from the selected chunks.
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

def test_email_signature_evidence():
    """Test that email signature question returns evidence with expected keywords."""
    token = authenticate()
    headers = {"Authorization": f"Bearer {token}"}
    
    question = "How do I set up my email signature?"
    
    print("\n" + "="*70)
    print(f"Testing Email Signature Evidence")
    print("="*70)
    print(f"\nQuestion: {question}")
    
    resp = requests.post(
        f"{API_BASE}/api/query",
        json={
            "question": question,
            "debug": 1
        },
        headers=headers,
        stream=True
    )
    
    answer = ""
    evidence = []
    debug_info = None
    
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            event = json.loads(line)
            if "debug" in event:
                debug_info = event["debug"]
            elif "token" in event:
                answer += event["token"]
            elif "answer" in event:
                answer = event["answer"]
                evidence = event.get("evidence", [])
        except json.JSONDecodeError:
            pass
    
    print(f"\n[1] Answer:")
    print(f"  {answer}")
    
    print(f"\n[2] Evidence ({len(evidence)} chunks):")
    for idx, ev in enumerate(evidence, 1):
        print(f"  [{idx}] {ev[:200]}{'...' if len(ev) > 200 else ''}")
    
    print(f"\n[3] Selected Chunks ({debug_info.get('selected_count', 0) if debug_info else 0}):")
    if debug_info and 'chunks' in debug_info:
        for idx, chunk in enumerate(debug_info['chunks'], 1):
            print(f"  [{idx}] {chunk.get('header', '')}")
            print(f"      Snippet: {chunk.get('snippet', '')[:150]}")
    
    # Check for required keywords in evidence
    evidence_text = " ".join(evidence).lower()
    
    has_signature = "signature" in evidence_text or "email signature" in evidence_text
    has_arial = "arial" in evidence_text
    has_format = "format" in evidence_text
    
    print(f"\n[4] Evidence Content Check:")
    print(f"  Contains 'signature': {has_signature}")
    print(f"  Contains 'arial': {has_arial}")
    print(f"  Contains 'format': {has_format}")
    
    # Check if any of the selected chunks contain the keywords
    if debug_info and 'chunks' in debug_info:
        print(f"\n[5] Chunk Content Check:")
        for idx, chunk in enumerate(debug_info['chunks'], 1):
            snippet_lower = chunk.get('snippet', '').lower()
            header_lower = chunk.get('header', '').lower()
            combined = f"{header_lower} {snippet_lower}"
            
            chunk_has_sig = 'signature' in combined or 'email signature' in combined
            chunk_has_arial = 'arial' in combined
            
            if chunk_has_sig or chunk_has_arial:
                print(f"  Chunk[{idx}] - signature:{chunk_has_sig}, arial:{chunk_has_arial}")
                print(f"    Header: {chunk.get('header', '')}")
    
    print(f"\n[6] Test Result:")
    if has_signature and has_arial:
        print(f"  ✓ PASS - Evidence contains both 'signature' and 'arial'")
        return True
    else:
        print(f"  ✗ FAIL - Missing keywords:")
        if not has_signature:
            print(f"    - 'signature' not found in evidence")
        if not has_arial:
            print(f"    - 'arial' not found in evidence")
        return False

if __name__ == "__main__":
    success = test_email_signature_evidence()
    exit(0 if success else 1)
