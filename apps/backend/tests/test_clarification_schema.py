import pytest
from pydantic import ValidationError
from app.schemas.query import QueryFinalResponse, ClarificationPayload, ClarificationType, build_clarification

def test_clarification_response_structure():
    # Test building a clarification response
    response = build_clarification(
        question="Which document?",
        options=["Doc A", "Doc B"],
        type="DOCUMENT"
    )
    
    assert response.needs_clarification is True
    assert response.refused is False
    assert response.answer == "Which document?"
    assert response.pipeline_marker == "CLARIFICATION_REQUIRED"
    assert response.clarification.question == "Which document?"
    assert response.clarification.options == ["Doc A", "Doc B"]
    # type is now a string in ClarificationPayload
    assert response.clarification.type == "DOCUMENT"

def test_clarification_invariants():
    # Test that manual construction respects invariants
    
    # Valid clarification
    valid = QueryFinalResponse(
        answer="Clarify please",
        refused=False,
        pipeline_marker="CLARIFICATION_REQUIRED",
        needs_clarification=True,
        clarification=ClarificationPayload(question="Clarify please", type="OTHER")
    )
    assert valid.needs_clarification is True

    # Invalid: needs_clarification=True but refused=True
    with pytest.raises(ValidationError):
        QueryFinalResponse(
            answer="Clarify please",
            refused=True, # Invalid
            pipeline_marker="CLARIFICATION_REQUIRED",
            needs_clarification=True,
            clarification=ClarificationPayload(question="Clarify please", type="OTHER")
        )

    # Invalid: answer != clarification.question
    with pytest.raises(ValidationError):
        QueryFinalResponse(
            answer="Something else", # Invalid
            refused=False,
            pipeline_marker="CLARIFICATION_REQUIRED",
            needs_clarification=True,
            clarification=ClarificationPayload(question="Clarify please", type="OTHER")
        )

    # Invalid: pipeline_marker mismatch
    with pytest.raises(ValidationError):
        QueryFinalResponse(
            answer="Clarify please",
            refused=False,
            pipeline_marker="LLM_VALIDATED", # Invalid
            needs_clarification=True,
            clarification=ClarificationPayload(question="Clarify please", type="OTHER")
        )

def test_backward_compatibility():
    # Existing responses should still work without new fields
    response = QueryFinalResponse(
        answer="The answer.",
        refused=False,
        pipeline_marker="LLM_VALIDATED",
        evidence=[],
        sources=[]
    )
    assert response.needs_clarification is None
    assert response.clarification is None
