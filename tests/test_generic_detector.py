import pytest
from app.services.rag_service import _is_generic_or_low_overlap

def test_generic_phrases_detection():
    # Case 1: Contains generic phrases
    generic_answer = "This typically involves checking the settings."
    evidence = "The settings are in the file."
    is_generic, reason = _is_generic_or_low_overlap(generic_answer, evidence)
    assert is_generic is True
    assert "typically involves" in reason

    # Case 2: Another generic phrase
    generic_answer_2 = "Generally, users should log in first."
    is_generic, reason = _is_generic_or_low_overlap(generic_answer_2, evidence)
    assert is_generic is True
    assert "generally" in reason

def test_low_overlap_detection():
    # Case 1: Low overlap
    answer = "The sky is blue and the sun is bright."
    evidence = "The database configuration requires port 5432."
    # No common tokens -> overlap 0 -> should return True
    is_generic, reason = _is_generic_or_low_overlap(answer, evidence)
    assert is_generic is True
    assert "low_evidence_overlap" in reason

    # Case 2: High overlap
    evidence_2 = "The database configuration requires port 5432."
    answer_2 = "The database needs port 5432 for configuration."
    # "database", "configuration", "requires"/"needs", "port", "5432"
    # Should have high overlap -> return False
    is_generic, reason = _is_generic_or_low_overlap(answer_2, evidence_2)
    assert is_generic is False

def test_short_answer_exception():
    # "Yes." has no tokens > 3 chars, so it should be ignored (False)
    answer = "Yes."
    evidence = "Is the server running? The server is running."
    is_generic, reason = _is_generic_or_low_overlap(answer, evidence)
    assert is_generic is False

def test_generic_boilerplate_mixed_with_content():
    # "As an AI language model, I can tell you that the port is 8080."
    # Now added to the list
    answer = "As an AI language model, I can tell you that the port is 8080."
    evidence = "The port is 8080."
    is_generic, reason = _is_generic_or_low_overlap(answer, evidence)
    assert is_generic is True
    assert "as an ai language model" in reason
