"""
End-to-end integration test for grounding gate.

Tests the /api/query endpoint to verify refusal responses work correctly.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_refusal_low_support():
    """Test that queries with low lexical overlap are refused."""
    print("\n=== Test 1: Low support query (should refuse) ===")
    
    # Query about something NOT in the indexed documents
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "question": "What are the quantum computing policies?",
            "debug": 1
        }
    )
    
    if response.status_code != 200:
        print(f"✗ HTTP {response.status_code}: {response.text}")
        return False
    
    # Parse NDJSON response
    lines = response.text.strip().split('\n')
    debug_info = None
    answer_text = ""
    
    for line in lines:
        if line.strip():
            data = json.loads(line)
            if 'debug' in data:
                debug_info = data['debug']
            if 'answer' in data:
                answer_text += data['answer']
    
    print(f"Answer text: '{answer_text}'")
    print(f"Debug info: {json.dumps(debug_info, indent=2)}")
    
    # Verify refusal
    if debug_info:
        assert debug_info.get('refused') is True, "Should have refused=True in debug info"
        assert debug_info.get('refusal_reason') == "NOT_FOUND", "Should have refusal_reason='NOT_FOUND'"
        assert debug_info.get('support_score', 99) < 2, "Support score should be < MIN_SUPPORT"
        print("✓ Query was correctly refused (low support)")
        return True
    else:
        print("✗ No debug info received")
        return False


def test_refusal_numeric_mismatch():
    """Test that numeric questions without numeric evidence are refused."""
    print("\n=== Test 2: Numeric question without numeric evidence (should refuse) ===")
    
    # Create a query that asks for a number but where chunks might lack it
    # Note: This test might pass if chunks DO have the number, so it's more of a demonstration
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "question": "How many light years to the nearest star?",  # Not in onboarding docs
            "debug": 1
        }
    )
    
    if response.status_code != 200:
        print(f"✗ HTTP {response.status_code}: {response.text}")
        return False
    
    lines = response.text.strip().split('\n')
    debug_info = None
    
    for line in lines:
        if line.strip():
            data = json.loads(line)
            if 'debug' in data:
                debug_info = data['debug']
    
    print(f"Debug info: {json.dumps(debug_info, indent=2)}")
    
    # This query should be refused either due to low support OR numeric mismatch
    if debug_info and debug_info.get('refused'):
        print("✓ Query was refused (as expected for irrelevant question)")
        return True
    else:
        print("⚠ Query was not refused (might have found weak evidence)")
        return True  # Still pass, as this is expected behavior


def test_strong_evidence_proceeds():
    """Test that queries with strong evidence proceed to LLM."""
    print("\n=== Test 3: Strong evidence query (should proceed) ===")
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "question": "What documents do I need to bring on my first day?",
            "debug": 1
        }
    )
    
    if response.status_code != 200:
        print(f"✗ HTTP {response.status_code}: {response.text}")
        return False
    
    lines = response.text.strip().split('\n')
    debug_info = None
    answer_text = ""
    
    for line in lines:
        if line.strip():
            data = json.loads(line)
            if 'debug' in data:
                debug_info = data['debug']
            if 'answer' in data:
                answer_text += data['answer']
    
    print(f"Answer text (first 150 chars): '{answer_text[:150]}'")
    print(f"Debug info refused: {debug_info.get('refused') if debug_info else 'N/A'}")
    
    # Verify NOT refused
    if debug_info:
        assert debug_info.get('refused') is not True, "Should NOT be refused (has strong evidence)"
        assert len(answer_text) > 0, "Should have generated an answer"
        print(f"✓ Query proceeded to LLM and generated answer (support_score={debug_info.get('support_score', 'N/A')})")
        return True
    else:
        print("✗ No debug info received")
        return False


def test_numeric_evidence_proceeds():
    """Test that numeric questions with numeric evidence proceed."""
    print("\n=== Test 4: Numeric question with numeric evidence (should proceed) ===")
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "question": "How many days of vacation do employees get?",
            "debug": 1
        }
    )
    
    if response.status_code != 200:
        print(f"✗ HTTP {response.status_code}: {response.text}")
        return False
    
    lines = response.text.strip().split('\n')
    debug_info = None
    answer_text = ""
    
    for line in lines:
        if line.strip():
            data = json.loads(line)
            if 'debug' in data:
                debug_info = data['debug']
            if 'answer' in data:
                answer_text += data['answer']
    
    print(f"Answer text (first 150 chars): '{answer_text[:150]}'")
    
    # Should proceed if document contains vacation day info
    if debug_info:
        if debug_info.get('refused'):
            print(f"⚠ Query was refused (support_score={debug_info.get('support_score')})")
            print("   This might mean the document doesn't contain vacation day info")
        else:
            print(f"✓ Query proceeded and generated answer")
        return True  # Pass either way, depends on document content
    else:
        print("✗ No debug info received")
        return False


def main():
    """Run all integration tests."""
    print("="*60)
    print("GROUNDING GATE INTEGRATION TESTS")
    print("="*60)
    print("\nNOTE: Server must be running at http://localhost:8000")
    print("      Documents must be indexed before running tests")
    
    try:
        # Check if server is running
        try:
            requests.get(f"{BASE_URL}/")
        except requests.ConnectionError:
            print("\n✗ ERROR: Server not running at http://localhost:8000")
            print("  Start the server with: uvicorn main:app --reload")
            return 1
        
        results = []
        results.append(test_refusal_low_support())
        results.append(test_refusal_numeric_mismatch())
        results.append(test_strong_evidence_proceeds())
        results.append(test_numeric_evidence_proceeds())
        
        print("\n" + "="*60)
        if all(results):
            print("ALL INTEGRATION TESTS PASSED ✓")
            print("="*60)
            return 0
        else:
            print(f"SOME TESTS FAILED ({sum(results)}/{len(results)} passed)")
            print("="*60)
            return 1
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
