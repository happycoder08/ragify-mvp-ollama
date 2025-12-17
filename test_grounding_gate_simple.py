"""
Simple test runner for grounding gate (no pytest required).
"""

import sys
from app.services.rag_service import (
    extract_evidence_lines,
    _compute_grounding_gate,
    MIN_SUPPORT,
    MIN_TOTAL_SUPPORT,
    MAX_EVIDENCE_LINES_TOTAL,
    MAX_EVIDENCE_LINES_PER_CHUNK
)


def test_extract_evidence_lines():
    """Test the line-level evidence extraction."""
    print("\n=== Testing extract_evidence_lines ===")
    
    # Test 1: Empty chunk
    result = extract_evidence_lines("", "what is the policy?")
    assert result == [], "Empty chunk should return empty list"
    print("✓ Empty chunk test passed")
    
    # Test 2: Lines with query overlap (returns tuples now)
    chunk = """
The vacation policy allows 15 days per year.
Sick leave is handled separately.
For vacation requests, submit at least 2 weeks notice.
    """
    result = extract_evidence_lines(chunk, "vacation policy requests")
    assert len(result) > 0, "Should extract lines with query overlap"
    assert isinstance(result[0], tuple), "Should return (line, overlap) tuples"
    assert "vacation" in result[0][0].lower(), "Top line should contain query terms"
    print("✓ Lexical overlap test passed")
    
    # Test 3: Max lines per chunk limit
    chunk = "\n".join([f"Line {i} with some content here" for i in range(20)])
    result = extract_evidence_lines(chunk, "content", max_lines=3)
    assert len(result) <= 3, f"Should respect max_lines limit, got {len(result)}"
    print("✓ Max lines limit test passed")
    
    # Test 4: Token-based filtering (lines with <2 tokens should be excluded unless they have anchors)
    chunk = """
A
The vacation policy allows 15 days.
OK
Submit your request with 2 weeks notice.
    """
    result = extract_evidence_lines(chunk, "vacation policy")
    # "A" and "OK" should be filtered out (< 2 tokens, no anchors)
    lines = [line for line, _ in result]
    assert not any(l in ["A", "OK"] for l in lines), "Short lines without anchors should be filtered"
    print("✓ Token-based filtering test passed")
    
    # Test 5: Deterministic sorting with tie-breakers
    chunk = """
First line with important information here.
- Important bullet with information.
Second line also has important details.
    """
    result1 = extract_evidence_lines(chunk, "important information")
    result2 = extract_evidence_lines(chunk, "important information")
    assert result1 == result2, "Should be deterministic"
    # Bullet should rank higher than non-bullet with same overlap
    assert result1[0][0].startswith("-"), "Bullet lines should rank higher in tie-breaker"
    print("✓ Deterministic sorting with tie-breakers test passed")


def test_grounding_gate():
    """Test the grounding gate decision logic."""
    print("\n=== Testing _compute_grounding_gate ===")
    
    # Test 1: Empty chunks refused
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "what is the policy?", [], []
    )
    assert should_proceed is False, "Empty chunks should be refused"
    assert reason == "NOT_FOUND"
    assert max_overlap == 0.0
    assert sum_top3 == 0.0
    assert failed_check == "NO_EVIDENCE"
    print("✓ Empty chunks refused test passed")
    
    # Test 2: Strong evidence passes (high overlap + high sum_top3)
    # Need multiple lines with "vacation" and "policy" to get sum_top3 >= 4
    selected_chunks = [
        (
            "The vacation policy allows employees to take time off.\n"
            "Review the vacation policy before submitting requests.\n"  
            "The vacation policy handbook is available online.\n"
            "Contact HR about the vacation policy if needed.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    assert should_proceed is True, f"Strong evidence should pass (max={max_overlap}, sum={sum_top3}, MIN_TOTAL={MIN_TOTAL_SUPPORT})"
    assert reason == ""
    assert failed_check == ""
    assert len(lines) > 0
    assert max_overlap >= MIN_SUPPORT
    assert sum_top3 >= MIN_TOTAL_SUPPORT
    print(f"✓ Strong evidence test passed (max_overlap={max_overlap}, sum_top3={sum_top3}, lines={len(lines)})")
    
    # Test 3: Low max_overlap refused
    selected_chunks = [
        (
            "This document discusses completely unrelated topics here.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy details", selected_chunks, chunk_ids
    )
    # May fail with NO_EVIDENCE or LOW_SUPPORT depending on whether any tokens match
    assert should_proceed is False, "Low max_overlap should be refused"
    assert failed_check in ["NO_EVIDENCE", "LOW_SUPPORT"]
    print(f"✓ Low max_overlap refused test passed (max_overlap={max_overlap}, failed_check={failed_check})")
    
    # Test 4: Low sum_top3 refused (even if max_overlap is acceptable)
    selected_chunks = [
        (
            "The vacation time is mentioned. "
            "Other unrelated content here. "
            "More unrelated content.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    # This should be refused if sum_top3 < MIN_TOTAL_SUPPORT
    if sum_top3 < MIN_TOTAL_SUPPORT:
        assert should_proceed is False, "Low sum_top3 should be refused"
        assert failed_check == "LOW_SUPPORT"
        print(f"✓ Low sum_top3 refused test passed (max_overlap={max_overlap}, sum_top3={sum_top3})")
    else:
        print(f"⚠ Skipped low sum_top3 test (got sum_top3={sum_top3} >= MIN_TOTAL_SUPPORT)")
    
    # Test 5: Numeric question without numeric evidence refused
    selected_chunks = [
        (
            "The vacation policy is generous and allows employees to take time off. "
            "Submit your request to your manager for approval.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "how many vacation days do I get?", selected_chunks, chunk_ids
    )
    # Should refuse either due to LOW_SUPPORT or MISSING_ANCHOR
    assert should_proceed is False, "Numeric question without numeric evidence should be refused"
    assert failed_check in ["LOW_SUPPORT", "MISSING_ANCHOR"]
    print(f"✓ Numeric question without evidence refused test passed (failed_check={failed_check})")
    
    # Test 6: Numeric question with numeric evidence passes
    selected_chunks = [
        (
            "The vacation policy allows employees to take 15 days per year.\n"
            "All vacation requests follow the vacation policy guidelines.\n"
            "The vacation policy provides 10 days for sick leave.\n"
            "Review the vacation policy for complete details.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "how many vacation days", selected_chunks, chunk_ids
    )
    assert should_proceed is True, f"Numeric question with numeric evidence should pass (max={max_overlap}, sum={sum_top3})"
    assert reason == ""
    assert failed_check == ""
    print(f"✓ Numeric question with evidence test passed (max_overlap={max_overlap}, sum_top3={sum_top3})")
    
    # Test 7: Time question without time evidence refused
    selected_chunks = [
        (
            "The onboarding process covers many important topics. "
            "You will meet with your manager and HR team.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "when does onboarding start?", selected_chunks, chunk_ids
    )
    assert should_proceed is False, "Time question without time evidence should be refused"
    assert failed_check in ["LOW_SUPPORT", "MISSING_ANCHOR"]
    print(f"✓ Time question without evidence refused test passed (failed_check={failed_check})")
    
    # Test 8: Time question with time evidence passes
    selected_chunks = [
        (
            "Onboarding starts at 9:00 AM on Monday morning.\n"
            "The first day onboarding session covers company policies.\n"
            "Review the onboarding schedule before your first day.\n"
            "The onboarding process includes orientation sessions.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "onboarding first day start time", selected_chunks, chunk_ids
    )
    assert should_proceed is True, f"Time question with time evidence should pass (max={max_overlap}, sum={sum_top3})"
    assert reason == ""
    assert failed_check == ""
    print(f"✓ Time question with evidence test passed (max_overlap={max_overlap}, sum_top3={sum_top3})")
    
    # Test 9: Global aggregation across multiple chunks
    selected_chunks = [
        ("The vacation policy is comprehensive.", {"chunk": 0}, 0.3),
        ("Employees receive 15 days of vacation per year.", {"chunk": 1}, 0.4),
        ("Submit vacation requests at least 2 weeks in advance.", {"chunk": 2}, 0.5),
        ("The vacation policy applies to all full-time staff.", {"chunk": 3}, 0.6)
    ]
    chunk_ids = ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    assert should_proceed is True, "Multiple chunks should aggregate evidence"
    # Should select top MAX_EVIDENCE_LINES_TOTAL globally
    assert len(lines) <= MAX_EVIDENCE_LINES_TOTAL
    print(f"✓ Global aggregation test passed (lines={len(lines)}, max_overlap={max_overlap}, sum_top3={sum_top3})")
    
    # Test 10: Deterministic behavior
    selected_chunks = [
        (
            "The vacation policy allows 15 days per year. "
            "Submit vacation requests 2 weeks in advance. "
            "The vacation policy is subject to manager approval.",
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
    
    # Test 1: Exactly at MIN_SUPPORT and MIN_TOTAL_SUPPORT thresholds
    selected_chunks = [
        (
            "The vacation policy information is available in the handbook here. "
            "The vacation policy applies to all employees. "
            "Review the vacation policy for details.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    
    # Should pass if thresholds are met
    if max_overlap >= MIN_SUPPORT and sum_top3 >= MIN_TOTAL_SUPPORT:
        assert should_proceed is True
        print(f"✓ Threshold boundary test passed (max_overlap={max_overlap}, sum_top3={sum_top3})")
    else:
        assert should_proceed is False
        print(f"✓ Below threshold test passed (max_overlap={max_overlap}, sum_top3={sum_top3})")
    
    # Test 2: Case insensitive matching
    selected_chunks = [
        (
            "THE VACATION POLICY ALLOWS 15 DAYS.\n"
            "ALL VACATION REQUESTS NEED APPROVAL.\n"
            "THE VACATION POLICY IS COMPREHENSIVE.\n"
            "REVIEW THE VACATION POLICY FOR DETAILS.",
            {"chunk": 0},
            0.3
        )
    ]
    chunk_ids = ["chunk_0"]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy", selected_chunks, chunk_ids
    )
    assert should_proceed is True, f"Should be case insensitive (max={max_overlap}, sum={sum_top3})"
    assert max_overlap >= MIN_SUPPORT
    assert sum_top3 >= MIN_TOTAL_SUPPORT
    print(f"✓ Case insensitive test passed (max_overlap={max_overlap}, sum_top3={sum_top3})")
    
    # Test 3: Global aggregation limit
    # Create many chunks with evidence to test MAX_EVIDENCE_LINES_TOTAL limit
    many_chunks = []
    for i in range(10):
        many_chunks.append((
            f"Line {i}: The vacation policy allows employees time off. "
            f"Vacation requests must follow the vacation policy guidelines.",
            {"chunk": i},
            0.3
        ))
    chunk_ids = [f"chunk_{i}" for i in range(10)]
    
    should_proceed, reason, lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        "vacation policy", many_chunks, chunk_ids
    )
    
    # Should limit to MAX_EVIDENCE_LINES_TOTAL
    assert len(lines) <= MAX_EVIDENCE_LINES_TOTAL, f"Should limit to {MAX_EVIDENCE_LINES_TOTAL} lines, got {len(lines)}"
    print(f"✓ Global aggregation limit test passed (got {len(lines)} lines, max={MAX_EVIDENCE_LINES_TOTAL})")


def main():
    """Run all tests."""
    print("="*60)
    print("GROUNDING GATE UNIT TESTS (Updated)")
    print(f"Configuration:")
    print(f"  MIN_SUPPORT = {MIN_SUPPORT}")
    print(f"  MIN_TOTAL_SUPPORT = {MIN_TOTAL_SUPPORT}")
    print(f"  MAX_EVIDENCE_LINES_TOTAL = {MAX_EVIDENCE_LINES_TOTAL}")
    print(f"  MAX_EVIDENCE_LINES_PER_CHUNK = {MAX_EVIDENCE_LINES_PER_CHUNK}")
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
