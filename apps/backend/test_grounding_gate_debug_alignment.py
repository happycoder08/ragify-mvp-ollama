"""
Unit test verifying grounding gate debug metrics match decision thresholds.
"""
import pytest
from app.services.grounding import MIN_SUPPORT, MIN_TOTAL_SUPPORT


def test_grounding_gate_debug_metrics_align_with_thresholds():
    """
    When grounding gate refuses due to low support, debug info must report
    exact values used in the decision and include threshold constants.
    """
    from app.services.rag_service import _compute_grounding_gate
    
    # Mock question and chunks with insufficient overlap
    question = "What time should I arrive?"
    # Create chunks with deliberately low overlap (only 1 token matches)
    weak_chunks = [
        ("Meeting time is noon.", {}, 100.0),
        ("Lunch at cafeteria.", {}, 200.0),
    ]
    chunk_ids = ["chunk_0", "chunk_1"]
    
    should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, weak_chunks, chunk_ids
    )
    
    # Verify refusal due to low support
    assert not should_proceed, "Should refuse due to insufficient overlap"
    assert refusal_reason == "NOT_FOUND"
    assert failed_check == "LOW_SUPPORT"
    
    # Verify metrics align with thresholds
    assert max_overlap < MIN_SUPPORT, f"max_overlap ({max_overlap}) should be below MIN_SUPPORT ({MIN_SUPPORT})"
    assert sum_top3 < MIN_TOTAL_SUPPORT, f"sum_top3 ({sum_top3}) should be below MIN_TOTAL_SUPPORT ({MIN_TOTAL_SUPPORT})"
    
    # Verify evidence lines are preserved even on refusal
    assert len(evidence_lines) > 0, "Evidence lines should be returned even on refusal for debugging"


def test_grounding_gate_debug_strong_support_passes():
    """
    When grounding gate allows query through, verify metrics exceed thresholds.
    """
    from app.services.rag_service import _compute_grounding_gate
    
    question = "What time should I arrive on my first day?"
    # Create chunks with strong overlap (many matching tokens)
    strong_chunks = [
        ("ARRIVE AT THE OFFICE (8:00 AM)\nYou should arrive at 8:00 AM on your first day at the main reception on the 3rd floor.", {}, 50.0),
        ("FIRST DAY SCHEDULE\nYour first day begins at 8:00 AM. Please arrive early and report to reception.", {}, 60.0),
    ]
    chunk_ids = ["chunk_0", "chunk_1"]
    
    should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, strong_chunks, chunk_ids
    )
    
    # Verify acceptance
    assert should_proceed, "Should proceed with strong evidence"
    assert refusal_reason == ""
    assert failed_check == ""
    
    # Verify metrics exceed thresholds
    assert max_overlap >= MIN_SUPPORT, f"max_overlap ({max_overlap}) should meet or exceed MIN_SUPPORT ({MIN_SUPPORT})"
    assert sum_top3 >= MIN_TOTAL_SUPPORT, f"sum_top3 ({sum_top3}) should meet or exceed MIN_TOTAL_SUPPORT ({MIN_TOTAL_SUPPORT})"
    assert len(evidence_lines) > 0, "Evidence lines should be extracted"
