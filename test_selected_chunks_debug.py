import pytest

from app.services import rag_service


class FakeCollection:
    def count(self):
        return 3

    def query(self, query_embeddings, n_results, where=None):
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
        }


ASYNC_REFUSAL_REASON = "NOT_FOUND"


@pytest.mark.asyncio
async def test_selected_chunks_and_context_preserved_on_refusal(monkeypatch):
    """Ensure selected chunk previews and context survive a refusal."""

    async def fake_embed_texts(texts, tenant_id="default"):
        return [[0.0] * 3 for _ in texts]

    def fake_get_collection(tenant_id: str):
        return FakeCollection()

    def fake_grounding_gate(question, selected_chunks, chunk_ids):
        # Grounding gate must see the selected chunks
        assert len(selected_chunks) == 3
        return False, ASYNC_REFUSAL_REASON, ["l1", "l2"], 0.0, 0.0, "LOW_SUPPORT"

    monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_service, "_get_collection", fake_get_collection)
    monkeypatch.setattr(rag_service, "_compute_grounding_gate", fake_grounding_gate)

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
    assert debug_info["retrieved_count"] == 3
    assert debug_info["selected_count"] == 3
    assert debug_info["refused"] is True
    assert debug_info["refusal_reason"] == ASYNC_REFUSAL_REASON
    assert debug_info.get("grounding_gate", {}).get("failed_check") == "LOW_SUPPORT"
    assert debug_info.get("gate_evidence_lines_count") == 2
    assert debug_info.get("support_score") == 0.0
    assert debug_info.get("chunks"), "Selected chunks should be present in debug info"
    assert len(context) > 0
    assert evidence, "Evidence should be derived from the selected chunks"
