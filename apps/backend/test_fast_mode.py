"""
Test fast mode constraints: max 2 sentences, highest relevance evidence only.
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

def test_fast_vs_full_mode():
    """Compare fast mode (2 sentence limit) vs full mode."""
    token = authenticate()
    headers = {"Authorization": f"Bearer {token}"}
    
    test_question = "What time should I arrive on my first day?"
    
    print("\n" + "="*70)
    print("Testing Fast vs Full Mode")
    print("="*70)
    
    # Test fast mode
    print(f"\n[1] FAST MODE - Question: {test_question}")
    resp_fast = requests.post(
        f"{API_BASE}/api/query",
        json={
            "question": test_question,
            "mode": "fast",
            "debug": 1
        },
        headers=headers,
        stream=True
    )
    
    answer_fast = ""
    evidence_fast = []
    debug_fast = None
    
    for line in resp_fast.iter_lines():
        if not line:
            continue
        try:
            event = json.loads(line)
            if "debug" in event:
                debug_fast = event["debug"]
            elif "token" in event:
                answer_fast += event["token"]
            elif "answer" in event:
                answer_fast = event["answer"]
                evidence_fast = event.get("evidence", [])
        except json.JSONDecodeError:
            pass
    
    sentence_count_fast = answer_fast.count('.') + answer_fast.count('?') + answer_fast.count('!')
    
    print(f"  Answer: {answer_fast}")
    print(f"  Sentence count: ~{sentence_count_fast}")
    print(f"  Evidence snippets: {len(evidence_fast)}")
    if debug_fast:
        print(f"  Retrieved: {debug_fast.get('retrieved_count', 0)} -> Selected: {debug_fast.get('selected_count', 0)}")
    
    # Test full mode
    print(f"\n[2] FULL MODE - Question: {test_question}")
    resp_full = requests.post(
        f"{API_BASE}/api/query",
        json={
            "question": test_question,
            "mode": "full",
            "debug": 1
        },
        headers=headers,
        stream=True
    )
    
    answer_full = ""
    evidence_full = []
    debug_full = None
    
    for line in resp_full.iter_lines():
        if not line:
            continue
        try:
            event = json.loads(line)
            if "debug" in event:
                debug_full = event["debug"]
            elif "token" in event:
                answer_full += event["token"]
            elif "answer" in event:
                answer_full = event["answer"]
                evidence_full = event.get("evidence", [])
        except json.JSONDecodeError:
            pass
    
    sentence_count_full = answer_full.count('.') + answer_full.count('?') + answer_full.count('!')
    
    print(f"  Answer: {answer_full}")
    print(f"  Sentence count: ~{sentence_count_full}")
    print(f"  Evidence snippets: {len(evidence_full)}")
    if debug_full:
        print(f"  Retrieved: {debug_full.get('retrieved_count', 0)} -> Selected: {debug_full.get('selected_count', 0)}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)
    print(f"Fast mode: {sentence_count_fast} sentences, {len(evidence_fast)} evidence snippets")
    print(f"Full mode: {sentence_count_full} sentences, {len(evidence_full)} evidence snippets")
    print(f"Fast mode constrained: {sentence_count_fast <= 2 and len(evidence_fast) <= 2}")

if __name__ == "__main__":
    test_fast_vs_full_mode()
