import pytest
from unittest.mock import MagicMock
from app.services.rag_service import _detect_numeric_conflict
from app.schemas.query import EvidenceItem

def test_detect_numeric_conflict_found():
    # Setup evidence with conflicting values
    ev1 = EvidenceItem(
        snippet="Employees get 15 vacation days per year.",
        chunk_id="03_Benefits_Policy_2025.txt_0",
        heading="Vacation",
        doc_id=1
    )
    ev2 = EvidenceItem(
        snippet="Employees get 20 vacation days per year.",
        chunk_id="04_Benefits_Policy_2026.txt_0",
        heading="Vacation",
        doc_id=2
    )
    
    question = "How many vacation days do I get?"
    conflict = _detect_numeric_conflict(question, [ev1, ev2])
    
    assert conflict is not None
    assert conflict["pipeline_marker"] == "CLARIFICATION_REQUIRED"
    assert conflict["needs_clarification"] is True
    assert conflict["clarification"]["question"] == "Which policy year are you referring to?"
    assert conflict["clarification"]["type"] == "TIMEFRAME"
    assert "2025" in conflict["clarification"]["options"]
    assert "2026" in conflict["clarification"]["options"]

def test_detect_numeric_conflict_no_conflict():
    # Setup evidence with same values
    ev1 = EvidenceItem(
        snippet="Employees get 15 vacation days per year.",
        chunk_id="03_Benefits_Policy_2025.txt_0",
        heading="Vacation",
        doc_id=1
    )
    ev2 = EvidenceItem(
        snippet="As stated, 15 days are allowed.",
        chunk_id="04_Benefits_Policy_2026.txt_0",
        heading="Vacation",
        doc_id=2
    )
    
    question = "How many vacation days do I get?"
    conflict = _detect_numeric_conflict(question, [ev1, ev2])
    
    assert conflict is None

def test_detect_numeric_conflict_irrelevant_question():
    # Question not about vacation days
    ev1 = EvidenceItem(
        snippet="Employees get 15 vacation days per year.",
        chunk_id="03_Benefits_Policy_2025.txt_0",
        heading="Vacation",
        doc_id=1
    )
    
    question = "What is the wifi password?"
    conflict = _detect_numeric_conflict(question, [ev1])
    
    assert conflict is None

def test_detect_numeric_conflict_single_source():
    # Only one source file
    ev1 = EvidenceItem(
        snippet="Employees get 15 vacation days per year.",
        chunk_id="03_Benefits_Policy_2025.txt_0",
        heading="Vacation",
        doc_id=1
    )
    ev2 = EvidenceItem(
        snippet="Employees get 20 vacation days per year.",
        chunk_id="03_Benefits_Policy_2025.txt_1",
        heading="Vacation",
        doc_id=1
    )
    
    question = "How many vacation days do I get?"
    conflict = _detect_numeric_conflict(question, [ev1, ev2])
    
    # Should be None because it's the same file (internal contradiction or nuance, not cross-file conflict)
    assert conflict is None
