import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.rag_service import query_collection, AnswerSchema

@pytest.mark.asyncio
async def test_query_collection_clarification():
    # Mock dependencies
    with patch("app.services.rag_service.embed_texts", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.rag_service.get_collection_async", new_callable=AsyncMock) as mock_get_collection, \
         patch("app.services.rag_service._call_chat_model", new_callable=AsyncMock) as mock_llm:
        
        # Setup mocks
        mock_embed.return_value = [[0.1] * 768]
        
        # Mock Collection object
        mock_collection = MagicMock()
        mock_collection.metadata = {"embed_dim": 768}
        mock_get_collection.return_value = mock_collection
        
        # Mock Chroma results
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["Full-time employees receive 15 vacation days per calendar year.", "Full-time employees receive 20 vacation days per calendar year."]],
            "metadatas": [[
                {"source_file": "Benefits_Policy_2025.txt", "header": "2025 Policy"}, 
                {"source_file": "Benefits_Policy_2026.txt", "header": "2026 Policy"}
            ]],
            "distances": [[0.1, 0.2]],
            "embeddings": None
        }
        
        # Call query_collection
        question = "How many vacation days do full-time employees receive per year?"
        
        # We need to provide tenant_id as first arg
        answer_gen, sources, evidence, context, debug_info = await query_collection("default", question)
             
        # Consume generator
        ans = ""
        async for token in answer_gen:
            ans += token
                 
        # Assertions
        assert debug_info["pipeline_marker"] == "CLARIFICATION_REQUIRED"
        assert debug_info["needs_clarification"] is True
        assert debug_info["clarification"]["options"] == ["2025", "2026"]
        assert len(sources) == 0
        assert len(evidence) == 0
        assert mock_llm.call_count == 0
        assert "Which policy year" in ans
