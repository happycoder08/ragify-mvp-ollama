import pytest

from app.services.rag_service import (
    repair_answer_by_schema,
    _validate_fact_single_response,
    _validate_policy_excerpt_response,
    _validate_boolean_specified_response,
)
from app.schemas.query import AnswerSchema, EvidenceItem


def _ev(snippet: str, chunk_id: str) -> EvidenceItem:
    return EvidenceItem(snippet=snippet, chunk_id=chunk_id, heading=None, doc_id=None)


def test_repair_fact_single_uses_evidence_single_sentence():
    # LLM produced multi-sentence answer without proper citation
    bad_answer = "Employees get 15 days of vacation per year. They can use them any time."
    evidence = [_ev("Employees get 15 days of vacation per year.", "chunk1")]

    repaired = repair_answer_by_schema(bad_answer, AnswerSchema.FACT_SINGLE, evidence)

    assert repaired is not None
    assert _validate_fact_single_response(repaired)
    assert "(CHUNK_ID=chunk1)" in repaired


def test_repair_policy_excerpt_builds_bullets_from_evidence():
    bad_answer = "Policy details are as follows: ..."  # missing bullets + citations
    evidence = [
        _ev("Employees must badge in at the lobby.", "c1"),
        _ev("Visitors must be escorted at all times.", "c2"),
    ]

    repaired = repair_answer_by_schema(bad_answer, AnswerSchema.POLICY_EXCERPT, evidence)

    assert repaired is not None
    assert _validate_policy_excerpt_response(repaired)
    # Should reference at least one evidence chunk
    assert "CHUNK_ID=c1" in repaired or "CHUNK_ID=c2" in repaired


def test_repair_boolean_specified_wraps_yes_with_citation():
    bad_answer = "Yes, you can work remotely two days a week."
    evidence = [_ev("Employees may work remotely two days per week.", "chunkB")]    

    repaired = repair_answer_by_schema(bad_answer, AnswerSchema.BOOLEAN_SPECIFIED, evidence)

    assert repaired is not None
    assert repaired.startswith("Yes")
    assert _validate_boolean_specified_response(repaired)
    assert "(CHUNK_ID=chunkB)" in repaired
