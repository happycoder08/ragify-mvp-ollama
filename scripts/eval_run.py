import argparse
import asyncio
import json
import sys
import os
import time
from typing import List, Dict, Any

# Ensure app module matches local checkout
sys.path.append(os.getcwd())

from app.services.rag_service import answer_question
from app.config import RAGIFY_MODE
from app.services import clients

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def _is_clarification_response(full_answer: str, debug_payload: Any) -> bool:
    if isinstance(debug_payload, dict):
        if debug_payload.get("response_type") == "clarification":
            return True
        if debug_payload.get("needs_clarification"):
            return True
        if debug_payload.get("clarification"):
            return True
        if (debug_payload.get("pipeline_marker") or "").upper() == "CLARIFICATION_REQUIRED":
            return True
    stripped = (full_answer or "").strip().lower()
    if "?" in stripped and stripped.startswith(("which", "what", "can you", "could you", "please clarify")):
        return True
    return False


async def run_single_eval(case: Dict[str, Any], suite: str) -> Dict[str, Any]:
    question = case["question"]
    expected = case["expected_behavior"]
    must_cite = case.get("must_cite_source_file")
    must_not_terms = case.get("must_not_include_terms", [])
    
    print(f"Running Case {case['id']}: {question} ... ", end="", flush=True)
    
    start_t = time.time()
    try:
        # Call RAG pipeline
        # Signature: answer_gen, sources, evidence, context_text, debug_payload, decision
        gen, sources, evidence, context, debug, decision = await answer_question(
            tenant_id="default",
            question=question,
            top_k=4,
            mode="full",   # or "fast" if desired
            doc_ids=None,
            debug=1
        )
        
        # Consume generator
        full_answer = ""
        async for chunk in gen:
            full_answer += chunk
            
        duration = time.time() - start_t
        
        # --- ASSERTIONS ---
        failures = []
        
        # 1. Behavior Check
        refusal_phrases = ["does not specify", "cannot find", "no information"]
        is_refusal = (decision and decision.refused) or any(p in full_answer.lower() for p in refusal_phrases)
        is_clarification = _is_clarification_response(full_answer, debug)

        if suite == "demo":
            if expected == "answer":
                if is_refusal:
                    failures.append(f"Expected ANSWER but got REFUSAL. Answer: {full_answer[:50]}...")
                if not evidence:
                    failures.append("Expected evidence_count > 0 but got 0")
                if not sources:
                    failures.append("Expected sources but got none")
            elif expected == "refuse":
                if not is_refusal:
                    failures.append(f"Expected REFUSAL but got ANSWER. Answer: {full_answer[:50]}...")
            elif expected == "clarify":
                if not is_clarification:
                    failures.append("Expected CLARIFICATION but did not see clarification response")
        else:
            if expected == "answer":
                if is_refusal:
                    failures.append(f"Expected ANSWER but got REFUSAL. Answer: {full_answer[:50]}...")
                if not evidence:
                    failures.append("Expected evidence but got None")
                    
            elif expected == "refuse":
                if not is_refusal:
                    failures.append(f"Expected REFUSAL but got ANSWER. Answer: {full_answer[:50]}...")
                
        # 2. Source Citation Check
        if must_cite:
            found_source = False
            for s in sources:
                if must_cite.lower() in s.lower():
                    found_source = True
                    break
            if not found_source:
                failures.append(f"Missing required source citation: {must_cite}. Found: {sources}")
                
        # 3. Forbidden Terms Check
        for term in must_not_terms:
            if term.lower() in full_answer.lower():
                failures.append(f"Answer contained forbidden term: '{term}'")

        success = len(failures) == 0
        
        if success:
            print(f"{GREEN}PASS{RESET} ({duration:.2f}s)")
        else:
            print(f"{RED}FAIL{RESET} ({duration:.2f}s)")
            for f in failures:
                print(f"  - {f}")
                
        return {
            "id": case["id"],
            "question": question,
            "success": success,
            "duration": duration,
            "failures": failures,
            "answer_preview": full_answer[:100]
        }
        
    except Exception as e:
        print(f"{RED}ERROR{RESET}")
        print(f"  - Exception: {e}")
        return {
            "id": case["id"],
            "question": question,
            "success": False,
            "duration": time.time() - start_t,
            "failures": [str(e)],
            "answer_preview": "EXCEPTION"
        }

async def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite.")
    parser.add_argument("--suite", choices=["core", "demo"], default="core", help="Evaluation suite to run.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller subset of cases.")
    args = parser.parse_args()

    limit = 10 if args.quick else None
    if limit:
        print(f"Running in Quick Mode (first {limit} cases)")

    if args.suite == "demo":
        cases_path = os.path.join("demo", "demo_questions.json")
    else:
        cases_path = os.path.join("eval", "eval_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    if limit:
        cases = cases[:limit]
        
    # Initialize clients
    clients.initialize_chroma_client()
    await clients.initialize_http_client()
    
    results = []
    print(f"Starting {args.suite} evaluation on {len(cases)} cases...")
    print("-" * 60)
    
    # Run sequentially to avoid rate limits or state issues in simple harness
    for case in cases:
        res = await run_single_eval(case, args.suite)
        results.append(res)
        
    print("-" * 60)
    print("EVALUATION SUMMARY")
    print("-" * 60)
    print(f"{'ID':<4} | {'Result':<6} | {'Duration':<8} | {'Question'}")
    
    success_count = 0
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        color = GREEN if r["success"] else RED
        print(f"{r['id']:<4} | {color}{status}{RESET}   | {r['duration']:.2f}s     | {r['question'][:50]}")
        if r["success"]:
            success_count += 1
            
    print("-" * 60)
    total = len(results)
    print(f"Total: {total}, Passed: {success_count}, Failed: {total - success_count}")
    
    if success_count < total:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    if os.name == 'nt':
        # Fix for Windows asyncio loop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
