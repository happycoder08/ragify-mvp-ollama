import pytest
import os
import sys
import asyncio
import json
import shutil
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.getcwd())
sys.path.insert(0, REPO_ROOT)

# Set up test environment like conftest.py does
VECTOR_DIR_TEST = os.path.join(REPO_ROOT, "vectorstore_test")
os.environ["APP_MODE"] = "ci"
os.environ["CI"] = "true"
os.environ["VECTOR_DIR"] = VECTOR_DIR_TEST
os.environ["ALLOW_CHROMA_INDEXING_IN_MOCK"] = "true"
os.environ["EMBEDDING_PROVIDER"] = "tfidf_test"
os.environ["LLM_PROVIDER"] = "mock"

from app.services import clients, rag_service
from app.services.ingestion import load_file_to_text, chunk_text


@pytest.mark.asyncio
async def test_arrival_time_extractor_real_document():
    """Test that the arrival time extractor correctly extracts 8:00 AM from the real Employee_Onboarding_Guide.txt document."""

    # Clean slate for vectorstore
    if os.path.exists(VECTOR_DIR_TEST):
        shutil.rmtree(VECTOR_DIR_TEST)
    os.makedirs(VECTOR_DIR_TEST, exist_ok=True)

    # Initialize Chroma client
    clients.initialize_chroma_client()

    # Reset providers to pick up environment variables
    if hasattr(rag_service, "reset_embedding_provider_for_tests"):
        rag_service.reset_embedding_provider_for_tests()
    if hasattr(rag_service, "reset_llm_provider_for_tests"):
        rag_service.reset_llm_provider_for_tests()

    # Load and index test documents (like conftest.py does)
    docs_dir = Path(REPO_ROOT) / "demo_docs"  # demo_docs is in repo root
    allowed_ext = {".txt", ".md", ".pdf", ".docx"}
    paths = sorted([p for p in docs_dir.iterdir() if p.suffix.lower() in allowed_ext])

    # Collect all chunks across all documents
    all_chunks = []
    per_file_chunks = {}

    for path in paths:
        filename = path.name
        # Extract text using production ingestion code
        text = load_file_to_text(str(path))
        # Chunk using production chunker
        chunks = chunk_text(text)
        if chunks:  # Only add if chunks were produced
            per_file_chunks[filename] = chunks
            all_chunks.extend(chunks)

    # Fit TF-IDF embedder on entire corpus
    if hasattr(rag_service, "fit_tfidf_test_embedder"):
        rag_service.fit_tfidf_test_embedder(all_chunks)

    # Index documents
    for filename, chunks in per_file_chunks.items():
        await rag_service.add_documents("default", chunks, filename)

    # Test the specific question that should trigger arrival time extraction
    question = "What time do I arrive my first day?"

    # Call query_collection which should use the real document data
    answer_gen, sources, evidence_items, context, debug_info = await rag_service.query_collection(
        tenant_id="default",
        question=question,
        top_k=4,
        debug=1,
        request_id="arrival-time-extractor-test"
    )

    # Collect the full answer
    answer = ""
    async for chunk in answer_gen:
        answer += chunk

    # Verify the extractor was used
    assert debug_info["pipeline_marker"] == "EXTRACTOR_ARRIVAL_TIME", f"Expected EXTRACTOR_ARRIVAL_TIME marker, got: {debug_info.get('pipeline_marker')}"

    # Verify the answer contains the correct time from the document
    assert "ARRIVAL_TIME:" in answer, f"Expected ARRIVAL_TIME prefix not found in: {answer}"
    assert "8:00" in answer or "8 am" in answer.lower(), f"Expected 8:00 AM or 8 am not found in: {answer}"

    # Verify not refused
    assert debug_info["refused"] == False, f"Expected refused=False, got: {debug_info.get('refused')}"

    print(f"✓ Arrival time extractor test passed. Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(test_arrival_time_extractor_real_document())