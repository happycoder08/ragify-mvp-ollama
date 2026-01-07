import pytest
from app.services.rag_service import _construct_schema_correct_answer_from_evidence, AnswerSchema
from collections import namedtuple

# Mock EvidenceItem
EvidenceItem = namedtuple("EvidenceItem", ["snippet", "doc_id", "heading", "chunk_id"])

def test_construct_summary_overview_onboarding():
    """
    Test constructing a summary from onboarding guide evidence.
    Should prioritize chunks from the same doc and extract key steps.
    """
    evidence = [
        EvidenceItem(
            snippet="ARRIVAL: Please arrive by 9:00 AM at the main reception.",
            doc_id="doc_1",
            heading="Arrival",
            chunk_id=1
        ),
        EvidenceItem(
            snippet="ORIENTATION: Orientation starts at 9:30 AM in Conference Room A.",
            doc_id="doc_1",
            heading="Orientation",
            chunk_id=2
        ),
        EvidenceItem(
            snippet="IT SETUP: Pick up your laptop from IT on the 2nd floor.",
            doc_id="doc_1",
            heading="IT Setup",
            chunk_id=3
        ),
        EvidenceItem(
            snippet="WORKSPACE: Your desk is assigned in the Engineering zone.",
            doc_id="doc_1",
            heading="Workspace",
            chunk_id=4
        ),
        EvidenceItem(
            snippet="LUNCH: Lunch is provided at 12:00 PM.",
            doc_id="doc_1",
            heading="Lunch",
            chunk_id=5
        ),
        EvidenceItem(
            snippet="Irrelevant doc content...",
            doc_id="doc_2",
            heading="Other",
            chunk_id=6
        )
    ]

    summary = _construct_schema_correct_answer_from_evidence(
        AnswerSchema.SUMMARY_OVERVIEW,
        evidence
    )

    assert summary is not None
    lines = summary.split('\n')
    assert len(lines) >= 4
    assert len(lines) <= 8
    
    # Check content
    text = summary.lower()
    assert "arrive" in text
    assert "orientation" in text
    assert "it setup" in text or "laptop" in text
    assert "workspace" in text or "desk" in text

def test_construct_summary_overview_mixed_docs():
    """
    Test that it prioritizes the top doc_id but falls back if needed (though requirements say prefer same doc).
    """
    evidence = [
        EvidenceItem(
            snippet="Step 1: Do this.",
            doc_id="doc_A",
            heading="Step 1",
            chunk_id=1
        ),
        EvidenceItem(
            snippet="Step 2: Do that.",
            doc_id="doc_B", # Different doc
            heading="Step 2",
            chunk_id=2
        ),
        EvidenceItem(
            snippet="Step 3: Do another thing.",
            doc_id="doc_A", # Same as top doc
            heading="Step 3",
            chunk_id=3
        )
    ]

    summary = _construct_schema_correct_answer_from_evidence(
        AnswerSchema.SUMMARY_OVERVIEW,
        evidence
    )
    
    # Should prioritize doc_A, so Step 1 and Step 3 should definitely be there.
    # Step 2 might be included if we need more points, but doc_A comes first.
    
    assert "Step 1" in summary
    assert "Step 3" in summary

def test_construct_summary_overview_extraction_logic():
    """
    Test extraction from snippets with multiple lines/bullets.
    """
    evidence = [
        EvidenceItem(
            snippet="""
            Overview of benefits:
            - Health Insurance
            - Dental Plan
            - Vision Coverage
            """,
            doc_id="doc_1",
            heading="Benefits",
            chunk_id=1
        )
    ]

    summary = _construct_schema_correct_answer_from_evidence(
        AnswerSchema.SUMMARY_OVERVIEW,
        evidence
    )
    
    assert "- Health Insurance" in summary
    assert "- Dental Plan" in summary
    assert "- Vision Coverage" in summary
