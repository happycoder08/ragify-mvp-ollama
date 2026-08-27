def test_refusal_invariant_behavior():
    """Ensure final answer/refused invariant: refusal phrase sets debug_info['refused']=True and is emitted once without chunk citations."""
    import asyncio
    from unittest.mock import patch
    from app.services.rag_service import query_collection

    async def mock_get_collection_async(tenant_id):
        class MockCollection:
            def query(self, **kwargs):
                return {
                    "documents": [["This document has unrelated content."]],
                    "metadatas": [[{
                        "source_file": "doc.pdf",
                        "filename": "doc.pdf",
                        "chunk": 0,
                        "doc_id": 1,
                        "chunk_id": "chunk_1",
                    }]],
                    "distances": [[0.1]],
                    "ids": [["chunk_1"]]
                }
        return MockCollection()

    async def mock_embed_texts(texts, tenant_id=None):
        return [[0.1] * 768]

    async def mock_generate_answer_stream(*args, **kwargs):
        # Simulate an LLM that produces the canonical refusal phrase with a chunk citation
        yield "Prelude text "
        yield "The document does not specify this. (chunk_id:chunk_1)"

    async def run_test():
           from app.services import clients
           # initialize shared HTTP client used by rag_service internals
           await clients.initialize_http_client()

           with patch('app.services.rag_service.get_collection_async', side_effect=mock_get_collection_async), \
               patch('app.services.rag_service.embed_texts', side_effect=mock_embed_texts), \
               patch('app.services.rag_service.generate_answer_stream', side_effect=mock_generate_answer_stream):

            gen, sources, evidence, context, debug_info = await query_collection(
                tenant_id="test",
                question="Unanswerable question?",
                top_k=4,
                mode="full",
                debug=0,
                request_id="refusal-invariant-test"
            )

            # Collect the answer
            answer = ""
            async for chunk in gen:
                answer += chunk

            # The wrapper should set refused=True when canonical phrase appears
            assert debug_info.get("refused") is True, f"Expected refused=True, got: {debug_info}"

            # The emitted answer should contain the canonical refusal phrase and should not include chunk citations
            assert "The document does not specify this." in answer, f"Canonical refusal phrase missing: {answer}"
            assert "chunk_id" not in answer, f"Chunk citation should be removed from emitted refusal: {answer}"

    asyncio.run(run_test())
