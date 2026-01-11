import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag_service import query_collection


async def _collect_answer(gen):
    answer = ""
    async for token in gen:
        answer += token
    return answer


def _mock_collection_with_benefits():
    mock_collection = MagicMock()
    mock_collection.metadata = {"embed_dim": 768}
    mock_collection.query.return_value = {
        "ids": [["id_2025", "id_2026"]],
        "documents": [[
            "BENEFITS POLICY (2025)\nVACATION\nFull-time employees receive 15 vacation days per calendar year.\nSICK TIME\nFull-time employees receive 10 sick days per calendar year.",
            "BENEFITS POLICY (2026)\nVACATION\nFull-time employees receive 20 vacation days per calendar year.\nSICK TIME\nFull-time employees receive 10 sick days per calendar year.",
        ]],
        "metadatas": [[
            {"source_file": "Benefits_Policy_2025.txt", "header": "BENEFITS POLICY (2025)"},
            {"source_file": "Benefits_Policy_2026.txt", "header": "BENEFITS POLICY (2026)"},
        ]],
        "distances": [[0.1, 0.2]],
        "embeddings": None,
    }
    return mock_collection


@pytest.mark.asyncio
async def test_sick_days_fact_single_alignment():
    def fake_call_chat_model(*args, **kwargs):
        async def gen():
            yield "Full-time employees receive 20 vacation days per calendar year."
        return gen()

    with patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_collection, \
         patch("app.services.rag_service._call_chat_model", side_effect=fake_call_chat_model):
        mock_embed.return_value = [[0.1] * 768]
        mock_get_collection.return_value = _mock_collection_with_benefits()
        question = "How many sick days do full-time employees receive per year?"
        answer_gen, sources, evidence, context, debug_info = await query_collection("default", question)
        answer = await _collect_answer(answer_gen)

        assert "10" in answer
        assert debug_info.get("pipeline_marker") in {"EXTRACTOR_FACT_SINGLE", "EXTRACTOR_EVIDENCE_FALLBACK"}
        assert debug_info.get("target_field") == "SICK"


@pytest.mark.asyncio
async def test_vacation_days_2025_fact_single():
    def fake_call_chat_model(*args, **kwargs):
        async def gen():
            if False:
                yield ""
        return gen()

    with patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_collection, \
         patch("app.services.rag_service._call_chat_model", side_effect=fake_call_chat_model):
        mock_embed.return_value = [[0.1] * 768]
        mock_get_collection.return_value = _mock_collection_with_benefits()
        question = "How many vacation days are there in 2025?"
        answer_gen, sources, evidence, context, debug_info = await query_collection("default", question)
        answer = await _collect_answer(answer_gen)

        assert "15" in answer
        assert "20" not in answer
        assert debug_info.get("pipeline_marker") in {"EXTRACTOR_FACT_SINGLE", "EXTRACTOR_EVIDENCE_FALLBACK"}
        assert debug_info.get("target_field") == "VACATION"


@pytest.mark.asyncio
async def test_vacation_days_2026_fact_single():
    def fake_call_chat_model(*args, **kwargs):
        async def gen():
            if False:
                yield ""
        return gen()

    with patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_collection, \
         patch("app.services.rag_service._call_chat_model", side_effect=fake_call_chat_model):
        mock_embed.return_value = [[0.1] * 768]
        mock_get_collection.return_value = _mock_collection_with_benefits()
        question = "How many vacation days are there in 2026?"
        answer_gen, sources, evidence, context, debug_info = await query_collection("default", question)
        answer = await _collect_answer(answer_gen)

        assert "20" in answer
        assert "15" not in answer
        assert debug_info.get("pipeline_marker") in {"EXTRACTOR_FACT_SINGLE", "EXTRACTOR_EVIDENCE_FALLBACK"}
        assert debug_info.get("target_field") == "VACATION"


@pytest.mark.asyncio
async def test_fact_single_hallucination_falls_back_to_slot_number():
    def fake_call_chat_model(*args, **kwargs):
        async def gen():
            yield "Reimbursement is available after 90 days."
        return gen()

    with patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_collection, \
         patch("app.services.rag_service._call_chat_model", side_effect=fake_call_chat_model):
        mock_embed.return_value = [[0.1] * 768]
        mock_get_collection.return_value = _mock_collection_with_benefits()
        question = "How many sick days do full-time employees get per year?"
        answer_gen, sources, evidence, context, debug_info = await query_collection("default", question)
        answer = await _collect_answer(answer_gen)

        assert "10" in answer
        assert debug_info.get("pipeline_marker") in {"EXTRACTOR_FACT_SINGLE", "EXTRACTOR_EVIDENCE_FALLBACK"}
        assert debug_info.get("fallback_from_evidence") is True
