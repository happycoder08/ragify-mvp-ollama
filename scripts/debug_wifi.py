import asyncio
import os
import sys
from unittest.mock import patch

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

# Ensure CI/test mode
os.environ["EMBEDDING_PROVIDER"] = "tfidf_test"
os.environ["LLM_PROVIDER"] = "mock"

from app.services import rag_service

async def mock_get_collection_async(tenant_id):
    class MockCollection:
        def query(self, **kwargs):
            return {
                    "documents": [["SECTION: WiFi\nFor guests: use SSID RAGIFY-GUEST and password RAGIFY-1234.\nUnique anchor: UNIQUE_TOKEN_PDF_2_1A7B4F"]],
                    "metadatas": [[{"source_file": "employee_handbook_excerpt.pdf", "chunk": 0, "doc_id": 1, "filename": "employee_handbook_excerpt.pdf"}]],
                    "distances": [[0.1]],
                    "ids": [["chunk_1"]],
                    "embeddings": [[[0.1]*768]]
            }
    return MockCollection()

async def mock_embed_texts(texts, tenant_id=None):
    return [[0.1]*768]

async def run():
    if hasattr(rag_service, "reset_embedding_provider_for_tests"):
        rag_service.reset_embedding_provider_for_tests()
    if hasattr(rag_service, "reset_llm_provider_for_tests"):
        rag_service.reset_llm_provider_for_tests()

    with patch('app.services.rag_service.get_collection_async', side_effect=mock_get_collection_async), \
         patch('app.services.rag_service.embed_texts', side_effect=mock_embed_texts):
        gen, sources, evidence, context, debug_info = await rag_service.query_collection(
            tenant_id="test",
            question="What is the wifi password?",
            top_k=4,
            mode="full",
            debug=1,
            request_id="debug-wifi"
        )
        answer = ""
        async for chunk in gen:
            answer += chunk
        print("ANSWER:\n", answer)
        print("DEBUG_INFO:\n", debug_info)
        print("SOURCES:\n", sources)
        print("EVIDENCE:\n", evidence)

if __name__ == '__main__':
    asyncio.run(run())
