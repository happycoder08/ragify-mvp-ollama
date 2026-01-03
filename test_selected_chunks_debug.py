import pytest

from app.services import rag_service


class FakeCollection:
    def count(self):
        return 3

    def query(self, query_embeddings, n_results, where=None, include=None):
        return {
            "documents": [[
                "Arrive at 8:00 AM at the main reception on the 3rd floor.",
                "Check in with reception upon arrival; orientation begins at 9:00 AM.",
                "Bring a government ID for your first-day badge creation.",
            ]],
            "metadatas": [[
                {"source_file": "onboarding.txt", "chunk": 0},
                {"source_file": "onboarding.txt", "chunk": 1},
                {"source_file": "onboarding.txt", "chunk": 2},
            ]],
            "distances": [[120.0, 180.0, 240.0]],
            "ids": [["doc1_0", "doc1_1", "doc1_2"]],
            "embeddings": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
        }


ASYNC_REFUSAL_REASON = "NOT_FOUND"


@pytest.mark.asyncio
async def test_selected_chunks_and_context_preserved_on_refusal(monkeypatch):
    """Ensure selected chunk previews and context survive a refusal."""

    async def fake_embed_texts(texts, tenant_id="default"):
        return [[0.0] * 3 for _ in texts]

    async def fake_get_collection(tenant_id: str):
        return FakeCollection()

    def fake_call_chat_model(question, context, tenant_id, mode, conversation_history, request_id, prompt_template):
        async def gen():
            yield "The document does not specify this."
        return gen()

    def fake_grounding_gate(question, selected_chunks, chunk_ids):
        # Grounding gate must see the selected chunks
        assert len(selected_chunks) == 3
        return False, ASYNC_REFUSAL_REASON, ["l1", "l2"], 0.0, 0.0, "LOW_SUPPORT"

    monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_service, "get_collection_async", fake_get_collection)
    monkeypatch.setattr(rag_service, "_compute_grounding_gate", fake_grounding_gate)
    monkeypatch.setattr(rag_service, "_call_chat_model", fake_call_chat_model)

    answer_gen, sources, evidence, context, debug_info = await rag_service.query_collection(
        tenant_id="test-tenant",
        question="What time should I arrive on my first day?",
        top_k=5,
        mode="full",
        conversation_history=None,
        doc_ids=None,
        debug=1,
        request_id="test-request",
    )

    # Drain the refusal generator to mirror streaming behavior
    refusal_text = ""
    async for chunk in answer_gen:
        refusal_text += chunk

    assert refusal_text == "The document does not specify this."
    assert debug_info["hits_count"] == 3
    assert debug_info["selected_count"] == 3
    # Check MMR debug fields
    assert "mmr_lambda" in debug_info
    assert "selected_by" in debug_info
    assert debug_info["selected_by"] == "mmr"
    assert debug_info["mmr_lambda"] == 0.6
    assert len(debug_info["selected_chunk_ids"]) == 3
    # Check coverage gate debug fields
    assert "coverage_ok" in debug_info
    assert "fallback_used" in debug_info
    assert debug_info["fallback_used"] is False  # No fallback in this test case
    assert debug_info["rewritten_query"] is None
    assert debug_info.get("selected_chunks"), "Selected chunks should be present in debug info"
    assert len(context) > 0
    assert evidence, "Evidence should be derived from the selected chunks"
