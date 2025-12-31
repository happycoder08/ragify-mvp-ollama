import pytest
from app.schemas.query import QueryFinalResponse, EvidenceItem, SourceItem
from fastapi.testclient import TestClient
from main import app

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
    # Simulate endpoint returning refused=false but answer is refusal message
    class DummyGen:
        def __aiter__(self):
            yield "The document does not specify this."
    async def fake_query_collection(*args, **kwargs):
        return DummyGen(), [], [], "", {"refused": False, "refusal_reason": None, "selected": []}
    monkeypatch.setattr("app.services.rag_service.query_collection", fake_query_collection)
    payload = {"question": "irrelevant", "tenant_id": "default", "stream": True}
    resp = client.post("/api/query", json=payload, timeout=10)
    assert resp.status_code == 200
    # Should always emit refused=true and refusal message
    assert "The document does not specify this." in resp.text