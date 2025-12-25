"""
Test grounding gate numeric/time anchor bonus.

Verifies that time-sensitive questions get a +1 bonus when evidence lines contain
numeric/time anchors (digits, HH:MM, AM/PM), allowing them to pass the grounding gate.
"""

import pytest
from app.services.grounding import _compute_grounding_gate, extract_evidence_lines


def test_time_question_with_time_anchor_passes():
    """
    Time-sensitive question should pass when evidence contains time anchor.
    
    Question: "What time should I arrive on my first day?"
    Evidence: Multiple lines with "arrive", "time", "day", and time anchor "(8:00 AM)"
    
    Expected:
    - Header line: "arrive" + time anchor → overlap=2 (1 token + 1 bonus)
    - Bullet lines: "day", "first" matches + potential bonuses
    - sum_top3 >= MIN_TOTAL_SUPPORT (4) → PASS
    """
    question = "What time should I arrive on my first day?"
    
    # Simulated chunk with arrival time and supporting context
    chunk_text = """1. ARRIVE AT THE OFFICE (8:00 AM)
    - This is your first day orientation
    - Report to the main reception on the 3rd floor
    - Arrive on time to complete badge registration
    - Bring: Government-issued ID, signed offer letter, completed I-9 form"""
    
    selected_chunks = [(chunk_text, {}, 100.0)]
    chunk_ids = ["chunk_0"]
    
    should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, selected_chunks, chunk_ids
    )
    
    # Assertions
    assert should_proceed is True, f"Expected to proceed but got refused: {failed_check} (max_overlap={max_overlap}, sum_top3={sum_top3})"
    assert refusal_reason == ""
    assert failed_check == ""
    assert max_overlap >= 2, f"Expected max_overlap >= 2, got {max_overlap}"
    assert sum_top3 >= 4, f"Expected sum_top3 >= 4, got {sum_top3}"
    assert len(evidence_lines) > 0
    # Verify the arrival line is in evidence
    assert any("8:00 AM" in line for line in evidence_lines), "Expected evidence to contain time anchor"


def test_non_time_question_no_bonus():
    """
    Non-time question should NOT get the bonus, even if evidence contains time.
    
    Question: "What documents do I need to bring?"
    Evidence: "Bring: Government-issued ID (8:00 AM)" - has time but question isn't time-sensitive
    
    Expected:
    - No time bonus applied
    - Overlap based only on tokens: "bring"
    """
    question = "What documents do I need to bring?"
    
    chunk_text = """WHAT TO BRING (8:00 AM):
    - Government-issued ID
    - Signed offer letter
    - Completed I-9 form"""
    
    selected_chunks = [(chunk_text, {}, 100.0)]
    chunk_ids = ["chunk_0"]
    
    # Extract evidence lines to inspect overlap scores
    evidence_tuples = extract_evidence_lines(chunk_text, question, max_lines=3)
    
    # The header line "WHAT TO BRING (8:00 AM):" should match "bring"
    # But since question isn't time-sensitive, no bonus should be applied
    header_line = next((line for line, overlap in evidence_tuples if "WHAT TO BRING" in line), None)
    
    if header_line:
        header_overlap = next((overlap for line, overlap in evidence_tuples if line == header_line), 0)
        # Without bonus, overlap should be 1 (just "bring")
        # With bonus (incorrect), it would be 2
        # This tests that non-time questions don't get bonus
        assert header_overlap == 1, f"Expected overlap=1 for non-time question, got {header_overlap}"


def test_time_question_without_time_anchor_may_fail():
    """
    Time-sensitive question without time anchor in evidence should not get bonus.
    
    Question: "What time should I arrive on my first day?"
    Evidence: "Please arrive at the office" (no time specified)
    
    Expected:
    - Raw overlap: 1 (arrive)
    - No bonus (no time anchor)
    - With limited supporting lines, likely fails sum_top3 < MIN_TOTAL_SUPPORT
    """
    question = "What time should I arrive on my first day?"
    
    # Chunk mentions arrival but no specific time
    chunk_text = """FIRST DAY CHECKLIST:
    - Please arrive at the office
    - Bring required documents
    - Meet your manager"""
    
    selected_chunks = [(chunk_text, {}, 100.0)]
    chunk_ids = ["chunk_0"]
    
    should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, selected_chunks, chunk_ids
    )
    
    # Should fail LOW_SUPPORT (either max_overlap < 2 or sum_top3 < 4)
    assert should_proceed is False
    assert failed_check == "LOW_SUPPORT"
    # Without time bonus, expect weaker support
    assert sum_top3 < 4, f"Expected sum_top3 < 4 without time bonus, got {sum_top3}"


def test_time_bonus_enables_weak_matches():
    """
    Time bonus should only apply when there's already token overlap.
    
    Question: "When should I arrive?"
    Evidence: "Report at 8:00 AM" (no "arrive" token, no overlap)
    
    Expected:
    - No token overlap between "arrive" and "report"
    - Even with time anchor, no bonus (bonus requires raw_overlap > 0)
    - Lines with no overlap should not appear in evidence
    """
    question = "When should I arrive?"
    
    chunk_text = (
        "SCHEDULE:\n"
        "    Report at 8:00 AM\n"
        "    Lunch at 12:00 PM\n"
        "    End at 5:00 PM"
    )
    
    # Extract evidence to verify bonus application
    evidence_tuples = extract_evidence_lines(chunk_text, question, max_lines=3)
    
    # Since there's no token overlap between question and evidence, bonus won't apply
    # (Bonus only applies when raw_overlap > 0)
    # So we expect NO lines in evidence
    assert len(evidence_tuples) == 0, f"Expected no evidence lines (no token overlap), got {len(evidence_tuples)}: {evidence_tuples}"


def test_multiple_chunks_aggregate_support():
    """
    Multiple chunks with time anchors should aggregate to pass total support.
    
    Question: "What time should I arrive on my first day?"
    Chunks: Two chunks, each with time-anchored lines
    
    Expected:
    - Each chunk contributes time-bonus lines
    - Aggregate max_overlap and sum_top3 should pass thresholds
    """
    question = "What time should I arrive on my first day?"
    
    chunk1 = "1. ARRIVE AT THE OFFICE (8:00 AM)\n    - Report to main reception"
    
    chunk2 = "FIRST DAY SCHEDULE:\n    - Arrival time: 8:00 AM at third floor\n    - Orientation starts at 9:00 AM"
    
    selected_chunks = [
        (chunk1, {}, 100.0),
        (chunk2, {}, 110.0)
    ]
    chunk_ids = ["chunk_0", "chunk_1"]
    
    should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, selected_chunks, chunk_ids
    )
    
    # Should pass with multiple supporting lines
    assert should_proceed is True, f"Expected to proceed with multiple chunks, failed: {failed_check}"
    assert max_overlap >= 2
    assert sum_top3 >= 4
    assert len(evidence_lines) >= 2


def test_numeric_question_with_digit_anchor():
    """
    Numeric question (not time) should also get bonus with digit anchors.
    
    Question: "How many days of vacation do I get?"
    Evidence: "Vacation: 15 days per year"
    
    Expected:
    - Question is numeric-sensitive (contains "days")
    - Evidence has digit anchor "15"
    - Bonus applied: overlap increased
    """
    question = "How many days of vacation do I get?"
    
    chunk_text = (
        "BENEFITS:\n"
        "    - Vacation: 15 days per year\n"
        "    - Sick leave: 10 days per year\n"
        "    - Holidays: 12 company holidays"
    )
    
    selected_chunks = [(chunk_text, {}, 100.0)]
    chunk_ids = ["chunk_0"]
    
    should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, selected_chunks, chunk_ids
    )
    
    # Should pass with numeric bonus
    assert should_proceed is True, f"Expected numeric question to pass, failed: {failed_check}"
    assert max_overlap >= 2
    # Verify vacation line is in evidence
    assert any("15 days" in line or "vacation" in line.lower() for line in evidence_lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
