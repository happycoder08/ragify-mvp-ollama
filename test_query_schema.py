"""
Unit tests for /api/query canonical response schema enforcement.

Validates that all query responses match the QueryFinalResponse schema.
"""

import pytest
import json
from fastapi.testclient import TestClient
from app.schemas.query import QueryFinalResponse, QueryRequest, EvidenceItem, SourceItem


def test_query_final_response_schema_refusal():
    """Test that refusal responses match canonical schema."""
    # Valid refusal response
    refusal = QueryFinalResponse(
        answer="The document does not specify this.",
        refused=True,
        refusal_reason="NOT_FOUND",
        evidence=[],
        sources=[]
    )
    
    assert refusal.answer == "The document does not specify this."
    assert refusal.refused is True
    assert refusal.refusal_reason == "NOT_FOUND"
    assert len(refusal.evidence) == 0
    assert len(refusal.sources) == 0
    
    # Invalid: wrong refusal message
    with pytest.raises(ValueError, match="canonical refusal message"):
        QueryFinalResponse(
            answer="I don't know",
            refused=True,
            refusal_reason="NOT_FOUND",
            evidence=[],
            sources=[]
        )


def test_query_final_response_schema_success():
    """Test that successful responses match canonical schema."""
    # Valid success response
    success = QueryFinalResponse(
        answer="Employees receive 15 days of vacation per year.",
        refused=False,
        refusal_reason=None,
        evidence=[
            EvidenceItem(
                snippet="All employees receive 15 days of vacation per year",
                chunk_id="1_onboarding.txt_3",
                heading="Vacation Policy",
                doc_id=1
            )
        ],
        sources=[
            SourceItem(
                doc_id=1,
                filename="onboarding.txt",
                chunk_id="1_onboarding.txt_3"
            )
        ]
    )
    
    assert success.refused is False
    assert len(success.evidence) >= 1
    assert len(success.sources) >= 1
    assert success.evidence[0].snippet == "All employees receive 15 days of vacation per year"
    assert success.sources[0].filename == "onboarding.txt"
    
    # Invalid combinations are coerced into standardized refusal responses
    # Case 1: no evidence for non-refused query -> coerced to refusal
    coerced_no_evidence = QueryFinalResponse(
        answer="Some answer",
        refused=False,
        refusal_reason=None,
        evidence=[],
        sources=[SourceItem(doc_id=1, filename="test.txt")]
    )
    assert coerced_no_evidence.refused is True
    assert coerced_no_evidence.answer == "The document does not specify this."
    assert coerced_no_evidence.evidence == []
    assert coerced_no_evidence.sources == []

    # Case 2: no sources for non-refused query -> coerced to refusal
    coerced_no_sources = QueryFinalResponse(
        answer="Some answer",
        refused=False,
        refusal_reason=None,
        evidence=[EvidenceItem(snippet="text", chunk_id="1")],
        sources=[]
    )
    assert coerced_no_sources.refused is True
    assert coerced_no_sources.answer == "The document does not specify this."
    assert coerced_no_sources.evidence == []
    assert coerced_no_sources.sources == []


def test_query_request_schema_validation():
    """Test QueryRequest schema validation."""
    # Valid request
    req = QueryRequest(
        question="What is the vacation policy?",
        top_k=5,
        mode="full",
        debug=1
    )
    assert req.question == "What is the vacation policy?"
    assert req.top_k == 5
    assert req.mode == "full"
    assert req.debug == 1
    
    # Invalid: empty question
    with pytest.raises(ValueError):
        QueryRequest(question="", top_k=5)
    
    # Invalid: top_k out of range
    with pytest.raises(ValueError):
        QueryRequest(question="test", top_k=100)
    
    # Invalid: debug out of range
    with pytest.raises(ValueError):
        QueryRequest(question="test", debug=5)


def test_evidence_item_schema():
    """Test EvidenceItem schema."""
    evidence = EvidenceItem(
        snippet="Sample text from document",
        chunk_id="1_doc.txt_0",
        heading="Introduction",
        doc_id=1
    )
    
    assert evidence.snippet == "Sample text from document"
    assert evidence.chunk_id == "1_doc.txt_0"
    assert evidence.heading == "Introduction"
    assert evidence.doc_id == 1
    
    # Optional fields
    evidence_minimal = EvidenceItem(
        snippet="Text",
        chunk_id="chunk_1"
    )
    assert evidence_minimal.heading is None
    assert evidence_minimal.doc_id is None


def test_source_item_schema():
    """Test SourceItem schema."""
    source = SourceItem(
        doc_id=1,
        filename="onboarding.txt",
        chunk_id="1_onboarding.txt_3"
    )
    
    assert source.doc_id == 1
    assert source.filename == "onboarding.txt"
    assert source.chunk_id == "1_onboarding.txt_3"
    
    # Optional doc_id and chunk_id
    source_minimal = SourceItem(filename="doc.txt")
    assert source_minimal.doc_id is None
    assert source_minimal.chunk_id is None


def test_json_serialization():
    """Test that schemas serialize to JSON correctly."""
    response = QueryFinalResponse(
        answer="Test answer",
        refused=False,
        evidence=[
            EvidenceItem(snippet="Evidence 1", chunk_id="1")
        ],
        sources=[
            SourceItem(filename="test.txt", chunk_id="1")
        ]
    )
    
    # Serialize to JSON
    json_str = response.json()
    data = json.loads(json_str)
    
    assert data["answer"] == "Test answer"
    assert data["refused"] is False
    assert len(data["evidence"]) == 1
    assert len(data["sources"]) == 1
    assert data["evidence"][0]["snippet"] == "Evidence 1"
    assert data["sources"][0]["filename"] == "test.txt"
    
    # Deserialize from JSON
    response_parsed = QueryFinalResponse.parse_raw(json_str)
    assert response_parsed.answer == "Test answer"
    assert response_parsed.refused is False


def test_response_schema_with_debug_info():
    """Test QueryFinalResponse with optional debug info."""
    from app.schemas.query import DebugInfo
    
    debug = DebugInfo(
        evidence_count=2,
        sources_count=1,
        retrieved_count=10,
        selected_count=5,
        request_id="test-123"
    )
    
    response = QueryFinalResponse(
        answer="Answer with debug",
        refused=False,
        evidence=[
            EvidenceItem(snippet="Evidence", chunk_id="1")
        ],
        sources=[
            SourceItem(filename="test.txt")
        ],
        debug_info=debug
    )
    
    assert response.debug_info is not None
    assert response.debug_info.evidence_count == 2
    assert response.debug_info.request_id == "test-123"
    
    # Exclude None values when serializing
    data = response.dict(exclude_none=True)
    assert "debug_info" in data
    assert "refusal_reason" not in data  # Should be excluded since it's None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
