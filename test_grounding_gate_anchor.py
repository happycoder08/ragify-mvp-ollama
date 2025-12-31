import pytest
from app.services.grounding import _compute_grounding_gate

def make_chunk(text):
    return (text, {}, 0.0)

def test_time_question_with_anchor():
    question = "What time do I arrive on my first day?"
    chunks = [make_chunk("ARRIVE AT THE OFFICE (8:00 AM)")]
    chunk_ids = ["c1"]
    should_proceed, refusal_reason, evidence_lines, *_ = _compute_grounding_gate(question, chunks, chunk_ids)
    assert should_proceed is True
    assert refusal_reason == ""
    assert any("8:00 AM" in l or "8:00 am" in l.lower() for l in evidence_lines)

def test_time_question_without_anchor():
    question = "What time do I arrive on my first day?"
    chunks = [make_chunk("ARRIVE AT THE OFFICE (see HR)")]
    chunk_ids = ["c1"]
    should_proceed, refusal_reason, evidence_lines, *_ = _compute_grounding_gate(question, chunks, chunk_ids)
    assert should_proceed is False
    assert refusal_reason in ("NOT_FOUND", "MISSING_ANCHOR")

def test_non_time_question_overlap():
    question = "Who is my manager?"
    chunks = [make_chunk("Your manager is Alice Smith."), make_chunk("Contact Alice Smith for onboarding.")]
    chunk_ids = ["c1", "c2"]
    should_proceed, refusal_reason, evidence_lines, max_overlap, *_ = _compute_grounding_gate(question, chunks, chunk_ids)
    # Should pass if overlap is sufficient
    assert (should_proceed and max_overlap >= 2) or (not should_proceed and max_overlap < 2)
