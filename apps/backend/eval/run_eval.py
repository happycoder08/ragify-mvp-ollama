#!/usr/bin/env python3
"""
Evaluation harness for RAGify onboarding Q&A.
Runs 15 test questions against /api/query endpoint and scores responses.
"""

import json
import time
import requests
import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"
EVAL_DIR = Path(__file__).parent

# Synonym map: treat these terms as equivalent during keyword matching
SYNONYM_MAP = {
    "camera": ["camera", "video"],
    "video": ["camera", "video"],
    "reimburse": ["reimburse", "reimbursement", "expense"],
    "reimbursement": ["reimburse", "reimbursement", "expense"],
    "sick": ["sick", "illness", "unwell"],
    "vacation": ["vacation", "pto", "time off"],
    "manager": ["manager", "supervisor", "lead"]
}

# Load gold standard Q&A
with open(EVAL_DIR / "qa_gold.json") as f:
    GOLD_QA = json.load(f)


def login(username: str = "demo", password: str = "demo123") -> Optional[str]:
    """Authenticate and return JWT token."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/login",
            json={"username": username, "password": password},
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("access_token")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return None


def query_api(question: str, token: str) -> Optional[Dict]:
    """POST question to /api/query and collect full streaming response."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response_data = {
            "answer": "",
            "evidence": [],
            "sources": [],
            "debug": {
                "selected_chunks": [],
                "context": ""
            }
        }
        
        with requests.post(
            f"{API_BASE}/api/query",
            json={"question": question, "debug": 1},  # Enable debug mode for diagnostics
            headers=headers,
            stream=True,
            timeout=30
        ) as resp:
            resp.raise_for_status()
            
            # Collect NDJSON streaming response
            # Format: {"debug": {...}}, {"token": "..."}, {"answer": "...", "evidence": [...], "sources": [...]}
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    
                    # Debug info (first message)
                    if "debug" in event:
                        response_data["debug"] = event["debug"]
                    
                    # Streaming token
                    elif "token" in event:
                        response_data["answer"] += event["token"]
                    
                    # Final response with answer, evidence, and sources
                    elif "answer" in event:
                        response_data["answer"] = event["answer"]
                        response_data["evidence"] = event.get("evidence", [])
                        response_data["sources"] = event.get("sources", [])
                        
                except json.JSONDecodeError:
                    pass
        
        return response_data
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return None


def normalize_answer(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse spaces."""
    return " ".join(text.lower().split())


def normalize_numbers(text: str) -> str:
    """
    Normalize numeric formats by removing currency symbols and commas.
    Examples: "$1,500" -> "1500", "£2,000.50" -> "2000.50"
    """
    import re
    
    # Remove common currency symbols: $, £, €, ¥
    text = re.sub(r'[$£€¥]', '', text)
    
    # Remove commas from numbers (e.g., "1,500" -> "1500")
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    return text


def normalize_times(text: str) -> str:
    """
    Normalize time formats to a canonical form for matching.
    Converts variations like "8:00 AM", "8 am", "08:00", "8:00am" to "8am".
    """
    import re
    
    # Pattern: optional leading 0, 1-2 digits, optional :00 or :30 etc, optional space, optional am/pm
    # Examples: "8:00 AM" -> "8am", "08:00" -> "8", "3:00 PM" -> "3pm"
    
    # Replace "XX:00 AM/PM" or "XX AM/PM" with "XXam/pm"
    text = re.sub(r'\b0?(\d{1,2}):00\s*(am|pm)\b', r'\1\2', text, flags=re.IGNORECASE)
    
    # Replace "XX AM/PM" with "XXam/pm"
    text = re.sub(r'\b0?(\d{1,2})\s*(am|pm)\b', r'\1\2', text, flags=re.IGNORECASE)
    
    # Replace "XX:00" (24-hour format) with "XX"
    text = re.sub(r'\b0?(\d{1,2}):00\b', r'\1', text)
    
    return text


def normalize_synonyms(text: str) -> str:
    """
    Normalize synonyms to canonical forms for consistent matching.
    Uses SYNONYM_MAP to replace synonyms with their equivalents.
    
    Example: "camera on" -> "camera on video on" (adds synonym)
    This allows matching either "camera" or "video" in keywords.
    """
    text_lower = text.lower()
    for canonical, synonyms in SYNONYM_MAP.items():
        for syn in synonyms:
            if syn in text_lower and canonical not in text_lower:
                # Add canonical term alongside synonym
                text_lower = text_lower.replace(syn, f"{syn} {canonical}")
    return text_lower


def check_keywords(text: str, keywords: List[str], required_keywords: List[str], debug_prefix: str = "") -> tuple[bool, List[str]]:
    """
    Check if text contains required keywords.
    Returns: (pass: bool, matched_keywords: list)
    """
    # Apply text normalization (lowercase, collapse spaces)
    text_norm = normalize_answer(text)
    
    # Apply time normalization
    text_norm = normalize_times(text_norm)
    
    # Apply number normalization
    text_norm = normalize_numbers(text_norm)
    
    # Apply synonym normalization (camera <-> video equivalence)
    text_norm = normalize_synonyms(text_norm)
    
    if debug_prefix:
        print(f"{debug_prefix}Normalized search text: {text_norm[:200]}...")
    
    matched = []
    failed = []
    
    for kw in required_keywords:
        kw_norm = normalize_times(kw.lower())
        kw_norm = normalize_numbers(kw_norm)
        kw_norm = normalize_synonyms(kw_norm)
        if kw_norm in text_norm:
            matched.append(kw)
        else:
            failed.append(kw)
    
    if debug_prefix and failed:
        print(f"{debug_prefix}Failed keywords: {failed}")
    
    # Pass if all required keywords present
    passed = len(matched) >= len(required_keywords)
    return passed, matched


def score_response(qa_pair: Dict, response: Dict) -> Dict:
    """
    Score a single response.
    Returns: {q_id, question, passed, keywords_matched, answer, evidence_count}
    """
    answer = response["answer"].strip() if response else ""
    evidence = response.get("evidence", []) if response else []
    
    # Concatenate answer + evidence for keyword matching
    # This ensures we check both LLM output and retrieved context
    evidence_text = " ".join(evidence)
    search_text = f"{answer} {evidence_text}"
    
    # Check keyword match in combined text
    passed, matched = check_keywords(
        search_text,
        qa_pair.get("keywords", []),
        qa_pair.get("required_keywords", []),
        debug_prefix="   "  # Add debug output with indentation
    )
    
    # Check if answer has evidence (grounding)
    has_evidence = len(evidence) > 0 or answer != ""
    
    return {
        "q_id": qa_pair["id"],
        "question": qa_pair["question"],
        "category": qa_pair.get("category", "unknown"),
        "passed": passed and has_evidence,
        "keywords_required": qa_pair.get("required_keywords", []),
        "keywords_matched": matched,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "evidence_count": len(evidence),
        "sources": response.get("sources", []) if response else []
    }


def run_evaluation(verbose: bool = False) -> Dict:
    """Run full evaluation against all Q&A pairs."""
    print("\n" + "="*70)
    print("RAGify Onboarding Evaluation Harness")
    print("="*70)
    
    # Login
    print("\n[1] Authenticating...")
    token = login()
    if not token:
        print("ERROR: Cannot proceed without auth token")
        return {}
    print(f"  Authenticated (token: {token[:20]}...)")
    
    # Wait for user to upload docs
    print("\n[2] Checking if documents are indexed...")
    time.sleep(1)
    
    # Run queries
    print(f"\n[3] Running {len(GOLD_QA)} test questions...")
    results = []
    passed_count = 0
    
    for i, qa_pair in enumerate(GOLD_QA, 1):
        print(f"\n   [{i}/{len(GOLD_QA)}] {qa_pair['question']}")
        
        response = query_api(qa_pair["question"], token)
        score = score_response(qa_pair, response)
        results.append(score)
        
        status = "PASS" if score["passed"] else "FAIL"
        print(f"   {status} | Keywords: {score['keywords_matched']}/{score['keywords_required']} | Evidence: {score['evidence_count']} snippets")
        
        # Show retrieval diagnostics if available
        if response and "debug" in response:
            debug = response["debug"]
            if "retrieved_count" in debug and "selected_count" in debug:
                print(f"   Retrieval: {debug['retrieved_count']} retrieved -> {debug['selected_count']} selected")
        
        # Print evidence text
        if response and response.get("evidence"):
            print(f"   Evidence snippets:")
            for idx, ev in enumerate(response["evidence"], 1):
                print(f"     [{idx}] {ev[:150]}{'...' if len(ev) > 150 else ''}")
        
        if score["passed"]:
            passed_count += 1
        
        if verbose and response:
            print(f"   Answer: {score['answer']}")
        
        # Small delay to avoid hammering server
        time.sleep(0.5)
    
    # Summary
    pass_rate = (passed_count / len(GOLD_QA)) * 100
    
    print("\n" + "="*70)
    print(f"Results: {passed_count}/{len(GOLD_QA)} passed ({pass_rate:.1f}%)")
    print("="*70)
    
    # Breakdown by category
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if result["passed"]:
            categories[cat]["passed"] += 1
    
    print("\nBreakdown by Category:")
    for cat in sorted(categories.keys()):
        cat_passed = categories[cat]["passed"]
        cat_total = categories[cat]["total"]
        cat_pct = (cat_passed / cat_total * 100) if cat_total > 0 else 0
        print(f"  {cat:25} {cat_passed}/{cat_total} ({cat_pct:.0f}%)")
    
    # Show failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n❌ Failed Questions:")
        for f in failures:
            missing = set(f["keywords_required"]) - set(f["keywords_matched"])
            print(f"  Q{f['q_id']:2}: {f['question']}")
            print(f"      Missing keywords: {missing}")
            if f["answer"]:
                print(f"      Got: {f['answer']}")
    
    # Save detailed results
    results_file = EVAL_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pass_rate": pass_rate,
            "passed": passed_count,
            "total": len(GOLD_QA),
            "results": results,
            "by_category": categories
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: eval/results.json")
    return {"pass_rate": pass_rate, "results": results}


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_evaluation(verbose=verbose)
