"""
Simple test runner for grounding gate (no pytest required).
"""

import sys
from app.services.rag_service import (
    extract_evidence_lines,
    _compute_grounding_gate,
    MIN_SUPPORT,
    MAX_EVIDENCE_LINES
)


def test_extract_evidence_lines():
    """Test the line-level evidence extraction."""
    print("\n=== Testing extract_evidence_lines ===")
    
    # Test 1: Empty chunk
    result = extract_evidence_lines("", "what is the policy?")
    assert result == [], "Empty chunk should return empty list"
    print("✓ Empty chunk test passed")
    
    # Test 2: Lines with query overlap
    chunk = """
The vacation policy allows 15 days per year.
Sick leave is handled separately.
For vacation requests, submit at least 2 weeks notice.
    """
    result = extract_evidence_lines(chunk, "vacation policy requests")
    assert len(result) > 0, "Should extract lines with query overlap"
    assert "vacation" in result[0].lower(), "Top line should contain query terms"
    print("✓ Lexical overlap test passed")
    
    # Test 3: Max lines limit
    chunk = "\n".join([f"Line {i} with some content here" for i in range(20)])
    result = extract_evidence_lines(chunk, "content", max_lines=3)
    assert len(result) <= 3, f"Should respect max_lines limit, got {len(result)}"
    print("✓ Max lines limit test passed")
    
    # Test 4: Deterministic sorting
    chunk = """
First line with important information here.
Second line also has important details.
Third line contains important facts too.
    """
    result1 = extract_evidence_lines(chunk, "important information")
    result2 = extract_evidence_lines(chunk, "important information")
    assert result1 == result2, "Should be deterministic"
    print("✓ Deterministic sorting test passed")


def test_grounding_gate():
    """Test the grounding gate decision logic."""
    print("\n=== Testing _compute_grounding_gate ===")
    
    # Test 1: Empty chunks refused
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "what is the policy?", [], []
    )
    assert should_proceed is False, "Empty chunks should be refused"
    assert reason == "NOT_FOUND"
    assert score == 0.0
    print("✓ Empty chunks refused test passed")
    
    # Test 2: Strong evidence passes
    selected_chunks = [
        (
            "The vacation policy allows employees to take 15 days per year. "
            "Vacation requests must be submitted at least 2 weeks in advance.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    assert should_proceed is True, "Strong evidence should pass"
    assert reason == ""
    assert len(lines) > 0
    assert score >= MIN_SUPPORT
    print(f"✓ Strong evidence test passed (score={score}, lines={len(lines)})")
    
    # Test 3: Numeric question without numeric evidence refused
    selected_chunks = [
        (
            "The vacation policy is generous and allows employees to take time off. "
            "Submit your request to your manager for approval.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "how many vacation days do I get?", selected_chunks, chunk_ids
    )
    assert should_proceed is False, "Numeric question without numeric evidence should be refused"
    assert reason == "NOT_FOUND"
    print("✓ Numeric question without evidence refused test passed")
    
    # Test 4: Numeric question with numeric evidence passes
    selected_chunks = [
        (
            "The vacation policy allows employees to take 15 days per year. "
            "Sick leave provides an additional 10 days annually.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "how many vacation days", selected_chunks, chunk_ids
    )
    assert should_proceed is True, "Numeric question with numeric evidence should pass"
    assert reason == ""
    print("✓ Numeric question with evidence test passed")
    
    # Test 5: Time question without time evidence refused
    selected_chunks = [
        (
            "The onboarding process covers many important topics. "
            "You will meet with your manager and HR team.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "when does onboarding start?", selected_chunks, chunk_ids
    )
    assert should_proceed is False, "Time question without time evidence should be refused"
    assert reason == "NOT_FOUND"
    print("✓ Time question without evidence refused test passed")
    
    # Test 6: Time question with time evidence passes
    selected_chunks = [
        (
            "Onboarding starts at 9:00 AM on Monday morning. "
            "The first day onboarding session covers company policies and procedures.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "onboarding first day start time", selected_chunks, chunk_ids
    )
    assert should_proceed is True, f"Time question with time evidence should pass (score={score})"
    assert reason == ""
    print("✓ Time question with evidence test passed")
    
    # Test 7: Multiple chunks aggregation
    selected_chunks = [
        ("The vacation policy is comprehensive.", {"chunk": 0}, 0.3),
        ("Employees receive 15 days of vacation per year.", {"chunk": 1}, 0.4),
        ("Submit requests at least 2 weeks in advance.", {"chunk": 2}, 0.5)
    ]
    chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    assert should_proceed is True, "Multiple chunks should aggregate evidence"
    assert len(lines) > 0
    print(f"✓ Multiple chunks aggregation test passed (lines={len(lines)})")
    
    # Test 8: Deterministic behavior
    selected_chunks = [
        (
            "The vacation policy allows 15 days per year. "
            "Submit requests 2 weeks in advance.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    question = "vacation policy"
    
    result1 = _compute_grounding_gate(question, selected_chunks, chunk_ids)
    result2 = _compute_grounding_gate(question, selected_chunks, chunk_ids)
    assert result1 == result2, "Should be deterministic"
    print("✓ Deterministic behavior test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing edge cases ===")
    
    # Test 1: Low support score refused
    selected_chunks = [
        ("This chunk talks about something unrelated to the query topic.", {"chunk": 0}, 0.5)
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "vacation policy details", selected_chunks, chunk_ids
    )
    assert should_proceed is False, "Low support score should be refused"
    assert score < MIN_SUPPORT
    print(f"✓ Low support refused test passed (score={score})")
    
    # Test 2: Case insensitive
    selected_chunks = [
        ("THE VACATION POLICY ALLOWS 15 DAYS.", {"chunk": 0}, 0.3)
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, score = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    assert should_proceed is True, "Should be case insensitive"
    assert score >= MIN_SUPPORT
    print("✓ Case insensitive test passed")


def main():
    """Run all tests."""
    print("="*60)
    print("GROUNDING GATE UNIT TESTS")
    print(f"Configuration: MIN_SUPPORT={MIN_SUPPORT}, MAX_EVIDENCE_LINES={MAX_EVIDENCE_LINES}")
    print("="*60)
    
    try:
        test_extract_evidence_lines()
        test_grounding_gate()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
