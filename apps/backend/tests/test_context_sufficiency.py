import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.rag_service import query_collection, AnswerSchema

# Mock objects
class MockChunk:
    def __init__(self, doc, meta, dist=0.5, chunk_id=0):
        self.doc = doc
        self.metadata = meta
        self.distance = dist
        self.id = str(chunk_id)

@pytest.mark.asyncio
async def test_context_sufficiency_gate_refusal():
    """
    Test that a broad question with insufficient context (short/header-only chunks)
    triggers a forced refusal.
    """
    # Mock dependencies
    with patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_coll, \
         patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service._call_chat_model", new_callable=MagicMock) as mock_chat:

        # Setup mocks
        mock_embed.return_value = [[0.1] * 384]  # Dummy embedding
        
        # Mock chat model response (async generator)
        async def chat_gen(*args, **kwargs):
            yield "Here is the summary."
        mock_chat.side_effect = chat_gen

        # Mock collection query results
        # Return 2 very short chunks (headers)
        mock_collection = MagicMock()
        mock_collection.metadata = {"embed_dim": 384}
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["Header 1", "Header 2"]],
            "metadatas": [[{"source_file": "test.pdf"}, {"source_file": "test.pdf"}]],
            "distances": [[0.1, 0.2]],
            "embeddings": None
        }
        mock_get_coll.return_value = mock_collection

        # Call query_collection with a broad question
        # "overview" triggers broad mode
        gen, sources, evidence, context, debug = await query_collection(
            tenant_id="test",
            question="give me an overview",
            top_k=5,
            debug=1
        )

        # Consume generator
        response = ""
        async for token in gen:
            response += token

        # Assertions
        assert debug["refused"] is True
        assert debug["pipeline_marker"] == "FORCED_REFUSAL_INSUFFICIENT_CONTEXT"
        assert response == "The document does not specify this."
        
        # Verify stats in debug info if available
        if "context_stats" in debug:
            assert debug["context_stats"]["total_chars"] < 500
            assert debug["context_stats"]["contentful_chunks_count"] < 2

@pytest.mark.asyncio
async def test_context_sufficiency_gate_pass():
    """
    Test that a broad question with sufficient context proceeds.
    """
    with patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_coll, \
         patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service._call_chat_model", new_callable=MagicMock) as mock_chat:

        mock_embed.return_value = [[0.1] * 384]
        
        # Return 2 long contentful chunks
        long_text_1 = "This is a long sentence " * 20 + " ONE"
        long_text_2 = "This is a long sentence " * 20 + " TWO"
        mock_collection = MagicMock()
        mock_collection.metadata = {"embed_dim": 384}
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [[long_text_1, long_text_2]],
            "metadatas": [[
                {"source_file": "test.pdf", "header": "Header One"}, 
                {"source_file": "test.pdf", "header": "Header Two"}
            ]],
            "distances": [[0.1, 0.2]],
            "embeddings": None
        }
        mock_get_coll.return_value = mock_collection
        
        # Mock chat model response
        async def chat_gen(*args, **kwargs):
            yield "Here is the summary."
        mock_chat.side_effect = chat_gen

        gen, sources, evidence, context, debug = await query_collection(
            tenant_id="test",
            question="give me an overview",
            top_k=5,
            debug=1
        )

        response = ""
        async for token in gen:
            response += token

        # Assertions
        # Should NOT be refused by the gate
        # Note: It might be refused by other gates (coverage), but we expect it to pass sufficiency
        # If it hits the LLM, debug["refused"] might be False (or missing if not set by LLM wrapper)
        # But definitely pipeline_marker should NOT be FORCED_REFUSAL_INSUFFICIENT_CONTEXT
        
        if "pipeline_marker" in debug:
            assert debug["pipeline_marker"] != "FORCED_REFUSAL_INSUFFICIENT_CONTEXT"
        
        # If it passed, response should be from chat model
        assert response == "Here is the summary."
