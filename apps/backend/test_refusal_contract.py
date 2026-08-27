import pytest
from app.schemas.query import QueryFinalResponse, EvidenceItem, SourceItem
from fastapi.testclient import TestClient
from main import app


def test_refused_matches_refusal_phrase_in_answer():
    """refused should be True iff the canonical refusal text appears in answer."""
    refusal_text = "The document does not specify this."

    # Case 1: non-refusal answer, no refusal text present
    resp_ok = QueryFinalResponse(
        answer="Some grounded answer",
        refused=False,
        refusal_reason=None,
        evidence=[EvidenceItem(snippet="foo", chunk_id="c1", heading="h", doc_id=1)],
        sources=[SourceItem(doc_id=1, filename="f.txt", chunk_id="c1")],
        debug_info=None,
    )
    assert resp_ok.refused == (refusal_text in resp_ok.answer)

    # Case 2: refusal answer, canonical refusal text present
    resp_refusal = QueryFinalResponse(
        answer=refusal_text,
        refused=True,
        refusal_reason="NOT_FOUND",
        evidence=[],
        sources=[],
        debug_info=None,
    )
    assert resp_refusal.refused == (refusal_text in resp_refusal.answer)

def test_validator_coerces_to_refusal():
    # refused==false but answer is refusal message
    resp = QueryFinalResponse(
        answer="The document does not specify this.",
        refused=False,
        refusal_reason=None,
        evidence=[EvidenceItem(snippet="foo", chunk_id="c1", heading="h", doc_id=1)],
        sources=[SourceItem(doc_id=1, filename="f.txt", chunk_id="c1")],
        debug_info=None
    )
    assert resp.refused is True
    assert resp.answer == "The document does not specify this."
    assert resp.evidence == []
    assert resp.sources == []

def test_validator_coerces_empty_evidence():
    # refused==false but evidence is empty
    resp = QueryFinalResponse(
        answer="Some answer",
        refused=False,
        refusal_reason=None,
        evidence=[],
        sources=[SourceItem(doc_id=1, filename="f.txt", chunk_id="c1")],
        debug_info=None
    )
    assert resp.refused is True
    assert resp.answer == "The document does not specify this."
    assert resp.evidence == []
    assert resp.sources == []

def test_validator_coerces_empty_sources():
    # refused==false but sources is empty
    resp = QueryFinalResponse(
        answer="Some answer",
        refused=False,
        refusal_reason=None,
        evidence=[EvidenceItem(snippet="foo", chunk_id="c1", heading="h", doc_id=1)],
        sources=[],
        debug_info=None
    )
    assert resp.refused is True
    assert resp.answer == "The document does not specify this."
    assert resp.evidence == []
    assert resp.sources == []

def test_validator_accepts_valid():
    # Valid non-refusal
    resp = QueryFinalResponse(
        answer="Valid answer",
        refused=False,
        refusal_reason=None,
        evidence=[EvidenceItem(snippet="foo", chunk_id="c1", heading="h", doc_id=1)],
        sources=[SourceItem(doc_id=1, filename="f.txt", chunk_id="c1")],
        debug_info=None
    )
    assert resp.refused is False
    assert resp.answer == "Valid answer"
    assert len(resp.evidence) == 1
    assert len(resp.sources) == 1

def test_api_query_contract(monkeypatch):
    # Patch query_collection to simulate contract violation
    from main import app
    client = TestClient(app)
    # Authenticate as demo user to satisfy auth dependency
    login_resp = client.post("/api/login", json={"username": "demo", "password": "demo123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    # Simulate endpoint returning refused=false but answer is refusal message
    async def dummy_gen():
        yield "The document does not specify this."

    async def fake_query_collection(*args, **kwargs):
        return dummy_gen(), [], [], "", {"refused": False, "refusal_reason": None, "selected": []}
    monkeypatch.setattr("app.services.rag_service.query_collection", fake_query_collection)
    payload = {"question": "irrelevant", "tenant_id": "default", "stream": True}
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post("/api/query", json=payload, headers=headers, timeout=10)
    assert resp.status_code == 200
    # Should always emit refused=true and refusal message
    assert "The document does not specify this." in resp.text