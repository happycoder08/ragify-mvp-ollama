#!/usr/bin/env python3
"""Test query with detailed logging of retrieval pipeline"""

import asyncio
import logging
from app.services import clients, rag_service

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    # Initialize clients
    clients.initialize_chroma_client()
    await clients.initialize_http_client()
    
    # Check collection
    collection = rag_service._get_collection("default")
    print(f"Collection: documents_default")
    print(f"Total chunks: {collection.count()}\n")
    
    # Manually trace through the retrieval pipeline
    question = "What time should I arrive on my first day?"
    print(f"Question: {question}\n")
    
    # Step 1: Embed question
    print("Step 1: Embedding question...")
    q_embedding = (await rag_service.embed_texts([question]))[0]
    print(f"  ✓ Got embedding with {len(q_embedding)} dimensions\n")
    
    # Step 2: Query ChromaDB
    print("Step 2: Querying ChromaDB (top_k=8)...")
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=8,
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    print(f"  Retrieved {len(docs)} chunks")
    print(f"  Distances: {distances}\n")
    
    # Step 3: Check similarity threshold filtering
    SIMILARITY_THRESHOLD = 400  # From config.py DEMO mode
    print(f"Step 3: Filtering by threshold ({SIMILARITY_THRESHOLD})...")
    filtered = [(d, m, dist) for d, m, dist in zip(docs, metas, distances) if dist < SIMILARITY_THRESHOLD]
    print(f"  After threshold filter: {len(filtered)} chunks\n")
    
    if filtered:
        print("Filtered chunks:")
        for i, (doc, meta, dist) in enumerate(filtered[:3]):
            print(f"  {i+1}. Distance: {dist:.2f}, Source: {meta.get('source_file', 'N/A')}")
            print(f"     Text: {doc[:100]}...\n")
    else:
        print("  ⚠️  NO CHUNKS PASSED THRESHOLD!\n")
    
    # Step 4: Run actual query
    print("Step 4: Running actual query through RAG pipeline...")
    try:
        answer_gen, sources = await rag_service.answer_question(
            tenant_id="default",
            question=question,
            top_k=8,
            mode="fast"
        )
        
        print("\nAnswer from LLM:")
        answer_text = ""
        async for token in answer_gen:
            print(token, end="", flush=True)
            answer_text += token
        
        print(f"\n\nSources returned: {sources}")
        print(f"Sources count: {len(sources)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
