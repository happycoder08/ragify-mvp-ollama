#!/usr/bin/env python3
"""
Diagnostic: Check what chunks are actually being sent to the LLM
"""
import sys
import asyncio
import logging

# Setup logging to see debug output
logging.basicConfig(level=logging.DEBUG)

from app.services.rag_service import query_collection

async def test_retrieval():
    tenant_id = "tenant_default"
    question = "What time should I arrive on my first day?"
    
    print("="*70)
    print(f"Testing retrieval for: {question}")
    print("="*70)
    
    # This will print lots of debug info due to logging.DEBUG
    answer_gen, sources = await query_collection(tenant_id, question, top_k=8, mode="fast")
    
    print("\n[ANSWER]")
    async for token in answer_gen:
        print(token, end="", flush=True)
    
    print(f"\n\n[SOURCES]")
    for src in sources:
        print(f"  - {src}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
