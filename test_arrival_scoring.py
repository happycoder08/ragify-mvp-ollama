#!/usr/bin/env python3
"""
Quick test to verify arrival time question scoring and prompt are working correctly.
Tests the scoring logic in isolation to ensure chunk 2 ranks highest.
"""

import sys
sys.path.insert(0, 'c:\\Users\\sergi\\Documents\\RAGify\\ragify-mvp-ollama')

from app.services.rag_service import _lexical_overlap_score, _hybrid_rerank_score

def test_arrival_time_scoring():
    """
    Test that the arrival time question retrieves chunk 2 as the top match.
    Chunk 2 contains: "THE OFFICE (8:00 AM) - Report to the main reception on the 3rd floor..."
    """
    
    # The actual chunks from the Employee Onboarding document
    chunks = {
        "chunk_0": "EMPLOYEE ONBOARDING CHECKLIST - Your First Day\nWelcome to our company! Below is your complete onboarding schedule.",
        "chunk_1": "SCHEDULE\nDay 1 includes four main activities:",
        "chunk_2": "THE OFFICE (8:00 AM) - Report to the main reception on the 3rd floor - Bring: Government-issued ID, signed offer letter, completed I-9 form",
        "chunk_3": "HR ORIENTATION (9:30 AM) - Meet with HR team in Conference Room B. Discuss benefits, payroll, and policies (1 hour)",
        "chunk_4": "MORNING (8:00 AM) - Attend new hire orientation session (1 hour) - Led by your team manager",
        "chunk_5": "AFTERNOON TRAINING (1:00 PM) - Policy changes and compliance training session (2 hours)",
        "chunk_6": "DOCUMENTS TO BRING - Government-issued ID, signed offer letter, completed I-9 form, two forms of address verification",
        "chunk_7": "ITEMS YOU WILL RECEIVE - Company laptop, access badge, parking pass, welcome kit with supplies",
    }
    
    question = "What time should I arrive on my first day?"
    
    # Score each chunk
    results = []
    for chunk_id, chunk_text in chunks.items():
        lexical_score = _lexical_overlap_score(question, chunk_text)
        # Use typical vector distance for good semantic match
        hybrid_score = _hybrid_rerank_score(question, chunk_text, 150)
        results.append({
            "chunk_id": chunk_id,
            "lexical": lexical_score,
            "hybrid": hybrid_score,
            "text_preview": chunk_text[:80],
        })
    
    # Sort by hybrid score (descending)
    results.sort(key=lambda x: x['hybrid'], reverse=True)
    
    print("ARRIVAL TIME QUESTION SCORING TEST")
    print("=" * 80)
    print(f"Question: {question}")
    print()
    print("Scoring results (sorted by hybrid score, descending):")
    print("-" * 80)
    
    for rank, result in enumerate(results, 1):
        print(f"\nRank #{rank}: {result['chunk_id']}")
        print(f"  Lexical Score: {result['lexical']:.3f}")
        print(f"  Hybrid Score:  {result['hybrid']:.3f}")
        print(f"  Preview: {result['text_preview']}...")
    
    # Verify chunk_2 is at top
    top_chunk = results[0]['chunk_id']
    top_hybrid = results[0]['hybrid']
    
    print()
    print("=" * 80)
    print("VERIFICATION:")
    if top_chunk == "chunk_2":
        print(f"[PASS] Chunk 2 ranks #1 with hybrid score {top_hybrid:.3f}")
        print("       This chunk contains the correct answer:")
        print("       'THE OFFICE (8:00 AM) - Report to the main reception on the 3rd floor'")
        return True
    else:
        print(f"[FAIL] Expected chunk_2 at top, got {top_chunk} (score {top_hybrid:.3f})")
        return False

if __name__ == "__main__":
    success = test_arrival_time_scoring()
    sys.exit(0 if success else 1)
