#!/usr/bin/env python3
"""Test query directly against indexed documents"""

import asyncio
from app.services import clients, rag_service

async def main():
    # Initialize clients
    clients.initialize_chroma_client()
    await clients.initialize_http_client()
    
    # Check collection
    collection = rag_service._get_collection("default")
    print(f"Collection: documents_default")
    print(f"Total chunks: {collection.count()}")
    
    # Try a query
    question = "What time should I arrive on my first day?"
    print(f"\nTesting query: {question}")
    
    try:
        answer_gen, sources = await rag_service.answer_question(
            tenant_id="default",
            question=question,
            top_k=8,
            mode="fast"
        )
        
        print("\nAnswer:")
        async for token in answer_gen:
            print(token, end="", flush=True)
        
        print(f"\n\nSources: {sources}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
