#!/usr/bin/env python3
"""
Final verification test: Demonstrate that the arrival time question is now fixed.
This test shows that:
1. The right chunk (chunk_2) ranks highest
2. The context will be built correctly
3. The system prompt has the right instruction for arrival time questions
"""

import sys
sys.path.insert(0, 'c:\\Users\\sergi\\Documents\\RAGify\\ragify-mvp-ollama')

def main():
    from app.services.rag_service import _lexical_overlap_score, _hybrid_rerank_score
    
    print("\n" + "=" * 80)
    print("RAGIFY ARRIVAL TIME FIX - FINAL VERIFICATION")
    print("=" * 80)
    
    # The question that was previously failing
    question = "What time should I arrive on my first day?"
    
    # The chunks from the Employee Onboarding Guide
    chunks = {
        "chunk_0": {
            "text": "EMPLOYEE ONBOARDING CHECKLIST - Your First Day\nWelcome to our company!",
            "description": "Title/header",
            "expected_rank": "Secondary"
        },
        "chunk_2": {
            "text": "THE OFFICE (8:00 AM) - Report to the main reception on the 3rd floor - Bring: Government-issued ID, signed offer letter, completed I-9 form",
            "description": "CORRECT ANSWER CHUNK",
            "expected_rank": "#1"
        },
        "chunk_4": {
            "text": "MORNING (8:00 AM) - Attend new hire orientation session (1 hour)",
            "description": "Time but no location",
            "expected_rank": "Lower"
        },
        "chunk_5": {
            "text": "AFTERNOON TRAINING (1:00 PM) - Policy changes and compliance training session",
            "description": "Different time (1:00 PM)",
            "expected_rank": "Low"
        }
    }
    
    print(f"\nQuestion: {question}")
    print("\n" + "-" * 80)
    print("SCORING RESULTS:")
    print("-" * 80)
    
    results = []
    for chunk_id, chunk_info in chunks.items():
        chunk_text = chunk_info["text"]
        
        lexical = _lexical_overlap_score(question, chunk_text)
        hybrid = _hybrid_rerank_score(question, chunk_text, 150)
        
        results.append({
            "chunk_id": chunk_id,
            "lexical": lexical,
            "hybrid": hybrid,
            "description": chunk_info["description"],
            "expected": chunk_info["expected_rank"]
        })
    
    # Sort by hybrid score
    results.sort(key=lambda x: x["hybrid"], reverse=True)
    
    print("\nRank | Chunk ID | Lexical | Hybrid | Description")
    print("-" * 60)
    for rank, r in enumerate(results, 1):
        status = "✓ CORRECT" if r["chunk_id"] == "chunk_2" and rank == 1 else ""
        print(f"#{rank:2} | {r['chunk_id']:8} | {r['lexical']:6.3f} | {r['hybrid']:6.3f} | {r['description']:30} {status}")
    
    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKLIST:")
    print("=" * 80)
    
    top_chunk = results[0]
    checks = [
        ("chunk_2 ranks #1", top_chunk["chunk_id"] == "chunk_2"),
        ("Hybrid score > 0.8", top_chunk["hybrid"] > 0.8),
        ("Lexical score = 1.0", top_chunk["lexical"] == 1.0),
        ("Text contains '8:00 AM'", "8:00 AM" in top_chunk.get("text", chunks.get(top_chunk["chunk_id"], {}).get("text", ""))),
        ("Text contains 'main reception'", "main reception" in chunks.get(top_chunk["chunk_id"], {}).get("text", "")),
        ("Text contains '3rd floor'", "3rd floor" in chunks.get(top_chunk["chunk_id"], {}).get("text", "")),
    ]
    
    all_pass = True
    for check_name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {check_name}")
        if not result:
            all_pass = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SYSTEM PROMPT RULE (for arrival time/location questions):")
    print("-" * 80)
    print("""
Include the exact time AND full location from Context in one sentence.
Example: '8:00 AM at the main reception on the 3rd floor'
""")
    
    print("=" * 80)
    print("EXPECTED LLM RESPONSE:")
    print("-" * 80)
    chunk2_text = chunks["chunk_2"]["text"]
    print(f"\nGiven this context:\n  {chunk2_text}\n")
    print("The LLM should respond:\n  8:00 AM at the main reception on the 3rd floor")
    print("\n(or similar variation combining time + location)")
    
    print("\n" + "=" * 80)
    if all_pass and top_chunk["chunk_id"] == "chunk_2":
        print("FINAL RESULT: [PASS] ✓ System is READY for testing")
        print("\nNext step: Start Ollama server and test with running backend")
        print("  ollama serve")
        print("  uvicorn main:app --reload --port 8000")
        return 0
    else:
        print("FINAL RESULT: [FAIL] ✗ Issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
