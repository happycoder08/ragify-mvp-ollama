#!/usr/bin/env python3
"""
Full pipeline debug for query: "What time should I arrive on my first day?"
Traces through: ChromaDB retrieval -> similarity filter -> hybrid reranking -> MIN_HYBRID_SCORE
"""
import httpx
import json
import asyncio
from app.services.rag_service import (
    get_chroma_collection, embed_texts, _hybrid_rerank_score
)
from app.config import (
    SIMILARITY_THRESHOLD, TOP_K_FAST, RERANKER_TOP_N, 
    ENABLE_RERANKING
)

def trace_full_pipeline():
    print("=" * 80)
    print("FULL RETRIEVAL PIPELINE DEBUG")
    print("=" * 80)
    
    question = "What time should I arrive on my first day?"
    
    # Step 1: Get collection
    print(f"\n[STEP 1] Getting ChromaDB collection...")
    collection = get_chroma_collection()
    print(f"Collection stats: {collection.count()} total chunks")
    
    # Step 2: Embed question
    print(f"\n[STEP 2] Embedding question...")
    print(f"Question: {question}")
    embedded = embed_texts([question])
    print(f"Embedding shape: {len(embedded)} x {len(embedded[0])}")
    
    # Step 3: Query ChromaDB
    print(f"\n[STEP 3] Querying ChromaDB (top_k={TOP_K_FAST})...")
    results = collection.query(
        query_embeddings=embedded,
        n_results=TOP_K_FAST,
        include=["documents", "metadatas", "distances"]
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    print(f"Retrieved {len(docs)} chunks:")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        print(f"  [{i}] dist={dist:.2f} | source={meta.get('source_file')} | chunk={meta.get('chunk')}")
        print(f"      preview: {doc[:100]}...")
    
    # Step 4: Apply similarity threshold
    print(f"\n[STEP 4] Applying SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}...")
    filtered_results = [
        (doc, meta, dist) for doc, meta, dist in zip(docs, metas, distances)
        if dist < SIMILARITY_THRESHOLD
    ]
    print(f"After threshold filter: {len(filtered_results)} chunks remain")
    
    if not filtered_results:
        print("❌ ZERO chunks passed threshold - will return 'no information found'")
        return
    
    for i, (doc, meta, dist) in enumerate(filtered_results):
        print(f"  [{i}] dist={dist:.2f} | source={meta.get('source_file')}")
    
    # Step 5: Hybrid reranking
    print(f"\n[STEP 5] Applying hybrid reranking...")
    print(f"ENABLE_RERANKING={ENABLE_RERANKING}")
    
    if not ENABLE_RERANKING and len(filtered_results) > 1:
        print("Using free lexical+semantic hybrid reranking...")
        
        # Calculate hybrid scores
        scored_results = []
        for doc, meta, dist in filtered_results:
            score = _hybrid_rerank_score(question, doc, dist)
            scored_results.append((doc, meta, dist, score))
            print(f"  score={score:.4f} | dist={dist:.2f} | source={meta.get('source_file')} | chunk={meta.get('chunk')}")
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\nAfter sorting by score:")
        for i, (doc, meta, dist, score) in enumerate(scored_results):
            print(f"  [{i}] score={score:.4f} | dist={dist:.2f}")
        
        # Step 6: Apply MIN_HYBRID_SCORE threshold
        print(f"\n[STEP 6] Applying MIN_HYBRID_SCORE filter...")
        MIN_HYBRID_SCORE = 0.15
        print(f"MIN_HYBRID_SCORE={MIN_HYBRID_SCORE}")
        
        final_results = [
            (doc, meta, dist, score) for doc, meta, dist, score in scored_results 
            if score >= MIN_HYBRID_SCORE
        ]
        
        print(f"After MIN_HYBRID_SCORE filter: {len(final_results)} chunks pass")
        for i, (doc, meta, dist, score) in enumerate(final_results):
            print(f"  [{i}] score={score:.4f} (>= {MIN_HYBRID_SCORE})")
        
        # Step 7: Apply RERANKER_TOP_N limit
        print(f"\n[STEP 7] Applying RERANKER_TOP_N={RERANKER_TOP_N} limit...")
        final_results = final_results[:RERANKER_TOP_N]
        print(f"Final context chunks: {len(final_results)}")
        
        if len(final_results) == 0:
            print("\n❌ NO CHUNKS PASSED FILTERS - Will return 'no information found'")
        else:
            print(f"\n✅ {len(final_results)} chunks will be sent to LLM:")
            for i, (doc, meta, dist, score) in enumerate(final_results):
                print(f"\n  CHUNK {i}:")
                print(f"    Score: {score:.4f}")
                print(f"    Source: {meta.get('source_file')}")
                print(f"    Chunk: {meta.get('chunk')}")
                print(f"    Content: {doc[:200]}...")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    trace_full_pipeline()
